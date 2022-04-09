# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
#from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
#from detectron2.modeling import ShapeSpec

__all__ = [
    "build_ConvNext_backbone",
    "LayerNorm",
    "Block",
    "ConvNeXt",
]

class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """
        
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=4):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride=4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=4)
        weight_init.c2_msra_fill(self.conv1)
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        return x

class Block(CNNBlockBase):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__(dim,dim,1) #parameters for CNNBlockBase: 'in_channels', 'out_channels', and 'stride'
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

@BACKBONE_REGISTRY.register()
class ConvNeXt(Backbone):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, cfg, input_shape, freeze_at=0):
        super().__init__()
        
        self.in_chans =3
        self.depths = [3, 3, 27, 3]
        self.dims=[192, 384, 768, 1536]
        self.drop_path_rate=0.
        self.layer_scale_init_value=1e-6
        self.head_init_scale=1.
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.out_features = ['res2', 'res3', 'res4', 'res5']
        self.out_strides= [4, 8, 16, 32]
        self._out_feature_strides = {self.out_features[0]:self.out_strides[0], self.out_features[1]:self.out_strides[1], self.out_features[2]:self.out_strides[2], self.out_features[3]:self.out_strides[3]}
        self._out_feature_channels = {self.out_features[0]:self.dims[0], self.out_features[1]: self.dims[1], self.out_features[2]: self.dims[2], self.out_features[3]: self.dims[3]}

        

        if self.out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in self.out_features]
            )
            
            
        #self.stem = nn.Sequential(
        #    nn.Conv2d(self.in_chans, self.dims[0], kernel_size=4, stride=4),
        #    LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
        #)
        self.stem = BasicStem(self.in_chans,self.dims[0], 4, 4)
        self.downsample_layers.append(self.stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=self.dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=self.layer_scale_init_value) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        #self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6) # final norm layer
        self.freeze(freeze_at)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs={}
        for i in range(4):
            x = self.downsample_layers[i](x)    
            x = self.stages[i](x)
            outputs[self.out_features[i]] = x
            #print("Inside ConvNexT, stage i ",i, " x shape : ",x.shape)
        #return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return outputs # global average pooling, (N, C, H, W) -> (N, C)

        
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }
        #return {"stage4": ShapeSpec(channels=768, stride=2)}
        
    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            print(" STEM : ",self.stem)
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@BACKBONE_REGISTRY.register()
def build_convnext_backbone(cfg, input_shape):
    print ("ConvNext: Freezing at : ",cfg.MODEL.BACKBONE.FREEZE_AT)
    return ConvNeXt(cfg, input_shape, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
