from abc import ABCMeta, abstractmethod
from functools import partial

import numpy as np

import torch
from torch import nn

from timm.models.layers import trunc_normal_

from .utils import ShapeSpec
from ..adet.modeling.swin.swin_transformer import BasicLayer


# Copied over from detectron2.modeling.backbone
#
class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    @property
    def padding_constraints(self) -> dict[str, int]:
        """
        This property is a generalization of size_divisibility. Some backbones and training
        recipes require specific padding constraints, such as enforcing divisibility by a specific
        integer (e.g., FPN) or padding to a square (e.g., ViTDet with large-scale jitter
        in :paper:vitdet). `padding_constraints` contains these optional items like:
        {
            "size_divisibility": int,
            "square_size": int,
            # Future options are possible
        }
        `size_divisibility` will read from here if presented and `square_size` indicates the
        square padding size if `square_size` > 0.

        TODO: use type of Dict[str, int] to avoid torchscipt issues. The type of padding_constraints
        could be generalized as TypedDict (Python 3.8+) to support more types in the future.
        """
        return {}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class ViTAEv2(Backbone):
    def __init__(self,
                img_size=224,
                in_chans=3,
                embed_dims=64,
                token_dims=64,
                downsample_ratios=[4, 2, 2, 2],
                kernel_size=[7, 3, 3, 3],
                RC_heads=[1, 1, 1, 1],
                NC_heads=4,
                dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                RC_op='cat',
                RC_tokens_type='window',
                NC_tokens_type='transformer',
                RC_group=[1, 1, 1, 1],
                NC_group=[1, 32, 64, 64],
                NC_depth=[2, 2, 6, 2],
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                stages=4,
                window_size=7,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                use_checkpoint=False,
                load_ema=True):
        super().__init__()

        self.stages = stages
        self.load_ema = load_ema
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint

        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i], self.NC_depth[i], dpr[startDpr:self.NC_depth[i]+startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], window_size=window_size, use_checkpoint=use_checkpoint)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]

        self.layers = nn.ModuleList(Layers)
        self.num_layers = len(Layers)

        self._freeze_stages()

        self._out_features = ["stage3", "stage4", "stage5"]
        self.init_weights()

    def _freeze_stages(self):

        if self.frozen_stages > 0:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        outs = {}
        b, _, h, w = x.shape
        for idx, layer in enumerate(self.layers):
            x, (h, w) = layer(x, (h, w))

            stage_name = "stage" + str(idx + 2)
            if stage_name in self._out_features:
                outs[stage_name] = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return outs

    def output_shape(self):
        return {
            "stage3": ShapeSpec(channels=128, stride=8),
            "stage4": ShapeSpec(channels=256, stride=16),
            "stage5": ShapeSpec(channels=512, stride=32),
        }
    

def build_backbone(cfg) -> ViTAEv2:
    vitaev2_type = cfg.MODEL.ViTAEv2.TYPE

    assert vitaev2_type == 'vitaev2_s', (f"Wrong ViTAEv2 model type: '{vitaev2_type}'. "
                                         f"Only 'vitaev2_s' is supported.")
    return ViTAEv2(
        in_chans=3,
        RC_tokens_type=['window', 'window', 'transformer', 'transformer'],
        NC_tokens_type=['window', 'window', 'transformer', 'transformer'],
        embed_dims=[64, 64, 128, 256],
        token_dims=[64, 128, 256, 512],
        downsample_ratios=[4, 2, 2, 2],
        NC_depth=[2, 2, 8, 2],
        NC_heads=[1, 2, 4, 8],
        RC_heads=[1, 1, 2, 4],
        mlp_ratio=4.,
        NC_group=[1, 32, 64, 128],
        RC_group=[1, 16, 32, 64],
        use_checkpoint=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        window_size=7,
        drop_path_rate=cfg.MODEL.ViTAEv2.DROP_PATH_RATE,
    )