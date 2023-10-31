from dataclasses import dataclass
from typing import Optional
from abc import ABCMeta, abstractmethod
from functools import partial

import torch
from torch import nn

from .lib import ViTAEv2
from .lib import ShapeSpec


class TransformerPureDetector(nn.Module):
    """
        Same as :class:`detectron2.modeling.ProposalNetwork`.
        Use one stage detector and a second stage for instance-wise prediction.
        """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.min_size_test = None
        self.max_size_test = None
        if cfg.MODEL.BACKBONE.NAME == "build_vitaev2_backbone":
            self.min_size_test = cfg.INPUT.MIN_SIZE_TEST
            self.max_size_test = cfg.INPUT.MAX_SIZE_TEST

        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(
            d2_backbone,
            PositionalEncoding2D(N_steps, cfg.MODEL.TRANSFORMER.TEMPERATURE, normalize=True)
        )
        backbone.num_channels = d2_backbone.num_channels
        self.detection_transformer = DETECTION_TRANSFORMER(cfg, backbone)
        bezier_matcher, point_matcher = build_matcher(cfg)

        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {
            "loss_ce": loss_cfg.POINT_CLASS_WEIGHT,
            "loss_texts": loss_cfg.POINT_TEXT_WEIGHT,
            "loss_ctrl_points": loss_cfg.POINT_COORD_WEIGHT,
            "loss_bd_points": loss_cfg.BOUNDARY_WEIGHT,
        }

        enc_weight_dict = {
            "loss_bezier": loss_cfg.BEZIER_COORD_WEIGHT,
            "loss_ce": loss_cfg.BEZIER_CLASS_WEIGHT
        }

        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()}
                )
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()}
            )
            weight_dict.update(aux_weight_dict)

        enc_losses = ["labels", "beziers"]
        if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
            dec_losses = ["labels", "texts", "ctrl_points", "bd_points"]
        else:
            dec_losses = ["labels", "texts", "ctrl_points"]

        self.criterion = SetCriterion(
            self.detection_transformer.num_classes,
            bezier_matcher,
            point_matcher,
            weight_dict,
            enc_losses,
            cfg.MODEL.TRANSFORMER.LOSS.BEZIER_SAMPLE_POINTS,
            dec_losses,
            cfg.MODEL.TRANSFORMER.VOC_SIZE,
            self.detection_transformer.num_points,
            focal_alpha=loss_cfg.FOCAL_ALPHA,
            focal_gamma=loss_cfg.FOCAL_GAMMA
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            output = self.detection_transformer(images)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            output = self.detection_transformer(images)
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_ctrl_points"]
            ctrl_point_text = output["pred_text_logits"]
            bd_points = output["pred_bd_points"]
            results = self.inference(
                ctrl_point_cls,
                ctrl_point_coord,
                ctrl_point_text,
                bd_points,
                images.image_sizes
            )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width, self.min_size_test, self.max_size_test)
                processed_results.append({"instances": r})

            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            gt_classes = targets_per_image.gt_classes

            raw_beziers = targets_per_image.beziers
            raw_ctrl_points = targets_per_image.polyline
            raw_boundary = targets_per_image.boundary
            gt_texts = targets_per_image.texts
            gt_beziers = raw_beziers.reshape(-1, 4, 2) / \
                         torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.detection_transformer.num_points, 2) / \
                             torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_boundary = raw_boundary.reshape(-1, self.detection_transformer.num_points, 4) / \
                          torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)[None, None, :]
            new_targets.append(
                {
                    "labels": gt_classes,
                    "beziers": gt_beziers,
                    "ctrl_points": gt_ctrl_points,
                    "texts": gt_texts,
                    "bd_points": gt_boundary
                }
            )

        return new_targets

    def inference(
            self,
            ctrl_point_cls,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            image_sizes
    ):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []
        # cls shape: (b, nq, n_pts, voc_size)
        ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        if bd_points is not None:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, bd, image_size in zip(
                    scores, labels, ctrl_point_coord, ctrl_point_text, bd_points, image_sizes
            ):
                selector = scores_per_image >= self.test_score_threshold
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]
                bd = bd[selector]

                result = Instances(image_size)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                result.rec_scores = ctrl_point_text_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)
                _, text_pred = ctrl_point_text_per_image.topk(1)
                result.recs = text_pred.squeeze(-1)
                bd[..., 0::2] *= image_size[1]
                bd[..., 1::2] *= image_size[0]
                result.bd = bd
                results.append(result)
            return results
        else:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, image_size in zip(
                    scores, labels, ctrl_point_coord, ctrl_point_text, image_sizes
            ):
                selector = scores_per_image >= self.test_score_threshold
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]

                result = Instances(image_size)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                result.rec_scores = ctrl_point_text_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)
                _, text_pred = ctrl_point_text_per_image.topk(1)
                result.recs = text_pred.squeeze(-1)
                result.bd = [None] * len(scores_per_image)
                results.append(result)
            return results