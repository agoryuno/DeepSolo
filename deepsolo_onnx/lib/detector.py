from typing import Optional, Any, Union
import warnings
import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vitaev2 import build_backbone
from .pos_encoding import PositionalEncoding2D
from .losses import SetCriterion

from ..adet.modeling.model.detection_transformer import DETECTION_TRANSFORMER
from ..adet.modeling.model.matcher import build_matcher


##########################################################################
####### The stuff in this block is slated for removal ####################
##########################################################################
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: list[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def shapes_to_tensor(x: list[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: list[tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: list[torch.Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        padding_constraints: Optional[dict[str, int]] = None,
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad.
            padding_constraints (optional[Dict]): If given, it would follow the format as
                {"size_divisibility": int, "square_size": int}, where `size_divisibility` will
                overwrite the above one if presented and `square_size` indicates the
                square padding size if `square_size` > 0.
        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if padding_constraints is not None:
            square_size = padding_constraints.get("square_size", 0)
            if square_size > 0:
                # pad to square.
                max_size[0] = max_size[1] = square_size
            if "size_divisibility" in padding_constraints:
                size_divisibility = padding_constraints["size_divisibility"]
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: list[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well

            # AG: not sure what this was all about and why it's faster than just doing nothing
            #
            #image_size = image_sizes[0]
            #padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            #batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
            batched_imgs = tensors[0].unsqueeze(0)

        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            device = (
                None if torch.jit.is_scripting() else ("cpu" if torch.jit.is_tracing() else None)
            )
            batched_imgs = tensors[0].new_full(batch_shape, pad_value, device=device)
            batched_imgs = move_device_like(batched_imgs, tensors[0])
            for i, img in enumerate(tensors):
                # Use `batched_imgs` directly instead of `img, pad_img = zip(tensors, batched_imgs)`
                # Tracing mode cannot capture `copy_()` of temporary locals
                batched_imgs[i, ..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: list["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
######################################################################################
######################################################################################
######################################################################################


def detector_postprocess(results, output_height, output_width, min_size=None, max_size=None):
    """
    scale align
    """
    if min_size and max_size:
        # to eliminate the padding influence for ViTAE backbone results
        size = min_size * 1.0
        scale_img_size = min_size / min(output_width, output_height)
        if output_height < output_width:
            newh, neww = size, scale_img_size * output_width
        else:
            newh, neww = scale_img_size * output_height, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        scale_x, scale_y = (output_width / neww, output_height / newh)
    else:
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])

    # scale points
    if results.has("ctrl_points"):
        ctrl_points = results.ctrl_points
        ctrl_points[:, 0::2] *= scale_x
        ctrl_points[:, 1::2] *= scale_y

    if results.has("bd") and not isinstance(results.bd, list):
        bd = results.bd
        bd[..., 0::2] *= scale_x
        bd[..., 1::2] *= scale_y

    return results


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
        breakpoint()
        images = [self.normalizer(x.to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a tuple that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images = self.preprocess_image(batched_inputs)
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