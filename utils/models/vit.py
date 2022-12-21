# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
#from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import ml_collections

import platform
from os.path import join as pjoin
from collections import OrderedDict  # pylint: disable=g-importing-member


### CONFIGS ###
def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_MRI_config():
    """Returns a custom implementation of ViT for MRI classification."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16,16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 6
    config.transformer.num_layers = 3
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_MRI_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_MRI_config()
    del config.patches.size
    config.patches.grid = (12, 12)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16,16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config

def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config

def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'R50-ViT-MRI' : get_r50_MRI_config(),
    'ViT-MRI' : get_MRI_config(),
    'testing': get_testing(),
}


### MODELING RESNET ###
class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        if platform.system() != 'Windows':
            conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
            conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
            conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)
    
            gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
            gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])
    
            gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
            gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])
    
            gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
            gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])
        else:
            conv1_weight = np2th(weights[n_block + '/' + n_unit + '/' + "conv1" + '/' + "kernel"], conv=True)
            conv2_weight = np2th(weights[n_block + '/' + n_unit + '/' + "conv2" + '/' + "kernel"], conv=True)
            conv3_weight = np2th(weights[n_block + '/' + n_unit + '/' + "conv3" + '/' + "kernel"], conv=True)
    
            gn1_weight = np2th(weights[n_block + '/' + n_unit + '/' + "gn1" + '/' + "scale"])
            gn1_bias = np2th(weights[n_block + '/' + n_unit + '/' + "gn1" + '/' + "bias"])
    
            gn2_weight = np2th(weights[n_block + '/' + n_unit + '/' + "gn2" + '/' + "scale"])
            gn2_bias = np2th(weights[n_block + '/' + n_unit + '/' + "gn2" + '/' + "bias"])
    
            gn3_weight = np2th(weights[n_block + '/' + n_unit + '/' + "gn3" + '/' + "scale"])
            gn3_bias = np2th(weights[n_block + '/' + n_unit + '/' + "gn3" + '/' + "bias"])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            if platform.system() != 'Windows':
                proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
                proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
                proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])
            else:
                proj_conv_weight = np2th(weights[n_block + '/' + n_unit + '/' + "conv_proj" + '/' +"kernel"], conv=True)
                proj_gn_weight = np2th(weights[n_block + '/' + n_unit + '/' + "gn_proj" + '/' +"scale"])
                proj_gn_bias = np2th(weights[n_block + '/' + n_unit + '/' + "gn_proj" + '/' +"bias"])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, in_channels = 3):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),    
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        return x


### MODELING ###
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor, in_channels = in_channels)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            #query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            query_weight = np2th(weights[ROOT +'/'+ ATTENTION_Q + '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()
            #key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[ROOT + '/' + ATTENTION_K+ '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()
            #value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[ROOT + '/' + ATTENTION_V + '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()
            #out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[ROOT+ '/' + ATTENTION_OUT+ '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()

            #query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            query_bias = np2th(weights[ROOT + '/' + ATTENTION_Q + '/' + "bias"]).view(-1)
            #key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            key_bias = np2th(weights[ROOT+'/'+ ATTENTION_K +'/'+ "bias"]).view(-1)
            #value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            value_bias = np2th(weights[ROOT +'/'+ ATTENTION_V +'/'+ "bias"]).view(-1)
            #out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
            out_bias = np2th(weights[ROOT +'/'+ ATTENTION_OUT +'/'+ "bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            #mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_0 = np2th(weights[ROOT+'/'+ FC_0 +'/'+ "kernel"]).t()
            #mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_weight_1 = np2th(weights[ROOT+'/'+ FC_1+'/' + "kernel"]).t()
            #mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_0 = np2th(weights[ROOT +'/'+ FC_0 +'/'+ "bias"]).t()
            #mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
            mlp_bias_1 = np2th(weights[ROOT +'/'+ FC_1 +'/'+ "bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            #self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.weight.copy_(np2th(weights[ROOT +'/'+ ATTENTION_NORM +'/'+ "scale"]))
            #self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT+'/'+ ATTENTION_NORM+'/'+ "bias"]))
            #self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT+'/'+ MLP_NORM+'/'+ "scale"]))
            #self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT+'/'+ MLP_NORM+'/'+ "bias"]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3, num_classes=21843, loss_weights=None, zero_head=False, vis=False, multi_stage_classification=False, multi_layer_classification=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes if not multi_stage_classification else 2
        self.zero_head = zero_head
        config = CONFIGS[config]
        self.classifier = config.classifier
        self.in_channels = in_channels
        self.transformer = Transformer(config, img_size, in_channels, vis)
        if multi_layer_classification:
            self.head = torch.nn.Sequential(
                Linear(config.hidden_size, 128),
                nn.ReLU(),
                Linear(128, self.num_classes)
                )
        else: 
            self.head = Linear(config.hidden_size, self.num_classes)
        
        if multi_stage_classification:
            self.head2 = Linear(config.hidden_size, self.num_classes)
            
        if 'grid' not in config.patches.keys():
            self.patch_size = config.patches.size
        else:
            self.patch_size = None
        self.loss_weights = loss_weights
        self.multi_stage_classification = multi_stage_classification
        self.multi_layer_classification = multi_layer_classification

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        
        if self.multi_stage_classification:
            if self.training:
                index = torch.flatten(labels) != 2
                x2 = x[index.tolist()]
                logits2 = self.head2(x2[:, 0])
            else :
                pred_labels = logits.argmax(-1)
                index = torch.tensor(pred_labels == 0)
                x2 = x[index.tolist()]
                if x2.shape[0]>0:
                    logits2 = self.head2(x2[:, 0])
                else:
                    logits2 = None
        
        if labels is not None and not self.multi_stage_classification:
            loss_fct = CrossEntropyLoss(weight = self.loss_weights.to(x.device) if torch.is_tensor(self.loss_weights) else None)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, loss
        elif labels is not None and self.multi_stage_classification:
            labels1 = torch.clone(labels)
            labels1[labels1==1] = 0
            labels1[labels1==2] = 1
            labels2 = labels[index]
            loss1_fct = CrossEntropyLoss(weight = self.loss_weights[0].to(x.device) if torch.is_tensor(self.loss_weights[0]) else None)
            loss2_fct = CrossEntropyLoss(weight = self.loss_weights[1].to(x.device) if torch.is_tensor(self.loss_weights[1]) else None)
            loss1 = loss1_fct(logits.view(-1, self.num_classes), labels1.view(-1))
            loss2 = loss2_fct(logits2.view(-1, self.num_classes), labels2.view(-1))
            return logits, logits2, loss1,loss2, index
            
        elif labels is None and self.multi_stage_classification:
            return logits, attn_weights, logits2, index
        else:
            return logits#, attn_weights ################################################################################################################

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                if not self.multi_layer_classification:
                    nn.init.zeros_(self.head.weight)
                    nn.init.zeros_(self.head.bias)
                #else:
                    #nn.init.zeros_(self.head[0].weight)
                    #nn.init.zeros_(self.head[0].bias)
                    #nn.init.zeros_(self.head[2].bias)
                    #nn.init.zeros_(self.head[2].weight)
                  
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
                
            new_weights = np2th(weights["embedding/kernel"], conv=True)
            if self.in_channels == 1 :
                new_weights = new_weights.mean(dim = 1, keepdim = True)
            elif self.in_channels == 2:
                new_weights = new_weights.mean(dim = 1, keepdim = True)
                new_weights = torch.cat([new_weights, new_weights], dim = 1)
                
            if self.patch_size is not None and self.patch_size != (16,16):
                scale_factor = self.patch_size[0] / 16
                new_weights = torch.nn.functional.interpolate(new_weights,scale_factor = scale_factor, mode = 'bilinear')

            self.transformer.embeddings.patch_embeddings.weight.copy_(new_weights)
            
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
                        
class LateFusionVisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3, num_classes=21843, loss_weights=None, zero_head=False, vis=False, multi_stage_classification=False, multi_layer_classification=False):
        super(LateFusionVisionTransformer, self).__init__()
        self.num_classes = num_classes if not multi_stage_classification else 2
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.in_channels = in_channels

        #self.embeddingDatiClinici = torch.nn.Sequential(
        #    Linear(35, 1024),
        #    nn.ReLU(),
        #     nn.Dropout(0.25),
        #    Linear(1024, 512)
        #    )

        self.transformer_t1 = Transformer(config, img_size, in_channels, vis)
        if multi_layer_classification:
            self.head = torch.nn.Sequential(
                Linear(config.hidden_size, 128),
                nn.ReLU(),
                Linear(128, self.num_classes)
                )
        else: 
            self.head = Linear(config.hidden_size, self.num_classes)
        
        self.last_head = torch.nn.Sequential(
            Linear(768, 1024),#+768+35
            nn.ReLU(),
#            nn.Dropout(0.5),
            Linear(1024, self.num_classes)
            )
        
        if multi_stage_classification:
            self.head2 = Linear(config.hidden_size*2, self.num_classes)
            
        if 'grid' not in config.patches.keys():
            self.patch_size = config.patches.size
        else:
            self.patch_size = None
        self.loss_weights = loss_weights
        self.multi_stage_classification = multi_stage_classification
        self.multi_layer_classification = multi_layer_classification

    def forward(self, inp, labels=None):
        x_1, x_2, x_3 = inp
        
        x_t1, attn_weights = self.transformer_t1(x_1)
        #x_t2, attn_weights2 = self.transformer_t1(x_2)
        #x_3o = self.embeddingDatiClinici(x_3)

        #logits_tmp = self.head(x_t1[:, 0])
        #x = torch.cat((logits_tmp,x_2), dim = 1)
        #logits = self.last_head(x)

        x = x_t1[:, 0]#torch.cat((x_t1[:, 0],x_t2[:, 0],x_3), dim = 1)
        logits = self.last_head(x)
                      
        if self.multi_stage_classification:
            raise NotImplementedError()
            if self.training:
                index = torch.flatten(labels) != 2
                x2 = x[index.tolist()]
                logits2 = self.head2(x2[:, 0])
            else :
                pred_labels = logits.argmax(-1)
                index = torch.tensor(pred_labels == 0)
                x2 = x[index.tolist()]
                if x2.shape[0]>0:
                    logits2 = self.head2(x2[:, 0])
                else:
                    logits2 = None
        
        if labels is not None and not self.multi_stage_classification:
            loss_fct = CrossEntropyLoss(weight = self.loss_weights.to(x.device) if torch.is_tensor(self.loss_weights) else None)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, loss
        elif labels is not None and self.multi_stage_classification:
            raise NotImplementedError()
            labels1 = torch.clone(labels)
            labels1[labels1==1] = 0
            labels1[labels1==2] = 1
            labels2 = labels[index]
            loss1_fct = CrossEntropyLoss(weight = self.loss_weights[0].to(x.device) if torch.is_tensor(self.loss_weights[0]) else None)
            loss2_fct = CrossEntropyLoss(weight = self.loss_weights[1].to(x.device) if torch.is_tensor(self.loss_weights[1]) else None)
            loss1 = loss1_fct(logits.view(-1, self.num_classes), labels1.view(-1))
            loss2 = loss2_fct(logits2.view(-1, self.num_classes), labels2.view(-1))
            return logits, logits2, loss1,loss2, index
            
        elif labels is None and self.multi_stage_classification:
            raise NotImplementedError()
            return logits, attn_weights, logits2, index
        else:
            return logits, attn_weights#attn_weights2

    def load_from(self, weights):
        raise NotImplementedError()
        with torch.no_grad():
            if self.zero_head:
                if not self.multi_layer_classification:
                    nn.init.zeros_(self.head.weight)
                    nn.init.zeros_(self.head.bias)
                #else:
                    #nn.init.zeros_(self.head[0].weight)
                    #nn.init.zeros_(self.head[0].bias)
                    #nn.init.zeros_(self.head[2].bias)
                    #nn.init.zeros_(self.head[2].weight)
                  
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
                
            new_weights = np2th(weights["embedding/kernel"], conv=True)
            if self.in_channels == 1 :
                new_weights = new_weights.mean(dim = 1, keepdim = True)
            elif self.in_channels == 2:
                new_weights = new_weights.mean(dim = 1, keepdim = True)
                new_weights = torch.cat([new_weights, new_weights], dim = 1)
                
            if self.patch_size is not None and self.patch_size != (16,16):
                scale_factor = self.patch_size[0] / 16
                new_weights = torch.nn.functional.interpolate(new_weights,scale_factor = scale_factor, mode = 'bilinear')

            self.transformer_t1.embeddings.patch_embeddings.weight.copy_(new_weights)
            
            self.transformer_t1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer_t1.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer_t1.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_t1.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            
            self.transformer_t2.embeddings.patch_embeddings.weight.copy_(new_weights)
            
            self.transformer_t2.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer_t2.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer_t2.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_t2.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer_t1.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer_t1.embeddings.position_embeddings.copy_(posemb)
                self.transformer_t2.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer_t1.embeddings.position_embeddings.copy_(np2th(posemb))
                self.transformer_t2.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer_t1.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
                    
            for bname, block in self.transformer_t2.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer_t1.embeddings.hybrid:
                self.transformer_t1.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                self.transformer_t2.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer_t1.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer_t1.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                
                self.transformer_t2.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer_t2.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer_t1.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
                        
                for bname, block in self.transformer_t2.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_size=35, num_classes=2, loss_weights=None):
        super(MultiLayerPerceptron, self).__init__()
        self.num_classes = num_classes
        self.in_size = in_size
        self.loss_weights = loss_weights
        self.linear = torch.nn.Sequential(
            Linear(in_size, 256),
            nn.ReLU(),
#            nn.Dropout(0.5),
#            Linear(256, 256),
#            nn.ReLU(),
            Linear(256, self.num_classes)
            )
        #self.linear = Linear(in_size, self.num_classes)

    def forward(self, x, labels=None):
        logits = self.linear(x)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight = self.loss_weights.to(x.device) if torch.is_tensor(self.loss_weights) else None)
            loss = loss_fct(logits, labels)
            return logits, loss
        else:
            return logits
