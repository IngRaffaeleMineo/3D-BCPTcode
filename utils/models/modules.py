'''
Copyright (c) R. Mineo, 2021-2023. All rights reserved.
This code was developed by R. Mineo in collaboration with PerceiveLab and other contributors.
For usage and licensing requests, please contact the owner.
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import typing


class Unsqueeze(nn.Module):
    def __init__(self, dim_list):
        super(Unsqueeze, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        for dim in self.dim_list:
            io = io.unsqueeze(dim)
        return io


class Squeeze(nn.Module):
    def __init__(self, dim_list):
        super(Squeeze, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        for dim in self.dim_list:
            io = io.squeeze(dim)
        return io


class View(nn.Module):
    def __init__(self, output_num_dims):
        super(View, self).__init__()
        self.output_num_dims = output_num_dims
        
    def forward(self,io):
        if (self.output_num_dims == 4):
            io = io.view(io.shape[0],io.shape[1],io.shape[2], -1)
        elif (self.output_num_dims == 3):
            io = io.view(io.shape[0],io.shape[1], -1)
        elif (self.output_num_dims == 2):
            io = io.view(io.shape[0], -1)
        elif (self.output_num_dims == 1):
            io = io.view(-1)
        else:
            raise RuntimeError("View module accept only output_num_dims 1/2/3/4")
        return io


class Reshape(nn.Module):
    def __init__(self, output_num_dims):
        super(Reshape, self).__init__()
        self.output_num_dims = output_num_dims
        
    def forward(self,io):
        if (self.output_num_dims == 4):
            io = io.reshape(io.shape[0],io.shape[1],io.shape[2], -1)
        elif (self.output_num_dims == 3):
            io = io.reshape(io.shape[0],io.shape[1], -1)
        elif (self.output_num_dims == 2):
            io = io.reshape(io.shape[0], -1)
        elif (self.output_num_dims == 1):
            io = io.reshape(-1)
        else:
            raise RuntimeError("Reshape module accept only output_num_dims 1/2/3/4")
        return io


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self,i):
        o = F.interpolate(i, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        return o


class Squeeze_3out(nn.Module):
    def __init__(self, dim_list):
        super(Squeeze_3out, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        io1, io2, io3 = io
        for dim in self.dim_list:
            io1 = io1.squeeze(dim)
            io2 = io2.squeeze(dim)
            io3 = io3.squeeze(dim)
        return io1, io2, io3


class Interpolate_3out(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate_3out, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self,i):
        i1, i2, i3 = i
        o1 = F.interpolate(i1, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        o2 = F.interpolate(i2, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        o3 = F.interpolate(i3, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        return o1,o2,o3


class Pad3D(nn.Module):
    def __init__(self, kernel_shape=(1, 1, 1), stride=(1, 1, 1)):
        super(Pad3D, self).__init__()
        self._kernel_shape = kernel_shape
        self._stride = stride

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        return x


class Mean_3out(nn.Module):
    def __init__(self):
        super(Mean_3out, self).__init__()
        
    def forward(self,i):
        i1, i2, i3 = i

        o1 = torch.mean(i1.view(i1.size(0), i1.size(1), i1.size(2)), 2)
        o2 = torch.mean(i2.view(i2.size(0), i2.size(1), i2.size(2)), 2)
        o3 = torch.mean(i3.view(i3.size(0), i3.size(1), i3.size(2)), 2)

        return o1,o2,o3


class Permute(nn.Module):
    def __init__(self, dims_list : typing.Union[tuple, list]):
        self.dims_list = dims_list
        super(Permute, self).__init__()
        
    def forward(self,i):
        o = i.permute(self.dims_list)
        return o


class PrintInShape(nn.Module):
    def __init__(self, stop=True):
        self.stop = stop
        super(PrintInShape, self).__init__()
        
    def forward(self,io):
        print(io.shape)
        if self.stop:
            raise RuntimeError("Stop to view shape.")
        return io


class ClassificationLayer(nn.Module):
    def __init__(self, type, in_dim, out_dim, enable_batchnorm=False, dropout=0.5):
        super(ClassificationLayer, self).__init__()
        self.enable_batchnorm = enable_batchnorm

        if type[0] == 'linear':
            self.fc1 = nn.Linear(in_features=in_dim, out_features=in_dim//2)
        elif type[0] == 'convolutional':
            self.fc1 = nn.Conv3d(in_dim, in_dim//2, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            raise NotImplementedError('You insert ' + type + ' as ClassificationLayer type, but only linear and convolutional type are implemented.')
        
        if self.enable_batchnorm:
            self.bn = nn.BatchNorm1d(num_features=in_dim//2)
        self.a = nn.ReLU()
        self.do = nn.Dropout(p=dropout)

        if type[1] == 'linear':
            self.fc2 = nn.Linear(in_features=in_dim//2, out_features=out_dim)
        elif type[1] == 'convolutional':
            self.fc2 = nn.Conv3d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            raise NotImplementedError('You insert ' + type + ' as ClassificationLayer type, but only linear and convolutional type are implemented.')
        
    def forward(self,i):
        tmp = self.fc1(i)
        if self.enable_batchnorm:
            tmp = self.bn(tmp)
        tmp = self.a(tmp)
        tmp = self.do(tmp)
        o = self.fc2(tmp)
        return o


class MultiheadAttentionMod(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super(MultiheadAttentionMod, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
    
    def forward(self,i):
        o,_ = self.attention(i,i,i)
        return o


class GlobalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, unflatten):
        super(GlobalMultiHeadAttention, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Global Attention
            nn.Flatten(1,3), # torch.Size([batch, time*height*width, channel])
            MultiheadAttentionMod(embed_dim=embed_dim, num_heads=num_heads, batch_first=True), # torch.Size([batch, time*height*width, channel])
            nn.Unflatten(1,unflatten), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, unflatten):
        super(TemporalMultiHeadAttention, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Temporal Attention
            nn.Flatten(2,4), # torch.Size([batch, time, height*width*channel])
            MultiheadAttentionMod(embed_dim=embed_dim, num_heads=num_heads, batch_first=True), # torch.Size([batch, time, height*width*channel])
            nn.Unflatten(2,unflatten), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class SpacialMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, unflatten1, unflatten0):
        super(SpacialMultiHeadAttention, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Temporal Attention
            nn.Flatten(2,3), # torch.Size([batch, time, height*width, channel])
            nn.Flatten(0,1), # torch.Size([batch*time, height*width, channel])
            MultiheadAttentionMod(embed_dim=embed_dim, num_heads=num_heads, batch_first=True), # torch.Size([batch*time, height*width, channel])
            nn.Unflatten(1,unflatten1), # torch.Size([batch*time, height, width, channel])
            nn.Unflatten(0,unflatten0), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, unflatten):
        super(TemporalTransformer, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Temporal Attention
            nn.Flatten(2,4), # torch.Size([batch, time, height*width*channel])
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=num_heads*64, dropout=0.1, batch_first=True),
                                    num_layers=num_layers), # torch.Size([batch, time, height*width*channel])
            nn.Unflatten(2,unflatten), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class SpacialTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, unflatten1, unflatten0):
        super(SpacialTransformer, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Temporal Attention
            nn.Flatten(2,3), # torch.Size([batch, time, height*width, channel])
            nn.Flatten(0,1), # torch.Size([batch*time, height*width, channel])
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=num_heads*64, dropout=0.1, batch_first=True),
                                    num_layers=num_layers), # torch.Size([batch*time, height*width, channel])
            nn.Unflatten(1,unflatten1), # torch.Size([batch*time, height, width, channel])
            nn.Unflatten(0,unflatten0), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class LastLayer(nn.Module):
    def __init__(self, type, in_dim, out_dim={'label':2}, dropout={'label_dropout':0.5}):
        super(LastLayer, self).__init__()

        self.labelClassification = ClassificationLayer(type=type,in_dim=in_dim,out_dim=out_dim['label'],dropout=dropout['label_dropout'])
        
    def forward(self,i):
        o_label = self.labelClassification(i)
        
        return o_label


class doppiaAngolazione_datiClinici_keyframe_LateFusion(nn.Module):
    def __init__(self, enable_doppiaAngolazione, enable_datiClinici, enable_keyframe, net_img_3d, net_img_2d, fusion_dim, in_dim_datiClinici, num_classes):
        super(doppiaAngolazione_datiClinici_keyframe_LateFusion, self).__init__()
        self.enable_doppiaAngolazione = enable_doppiaAngolazione
        self.enable_datiClinici = enable_datiClinici
        self.enable_keyframe = enable_keyframe
        #self.enable_batchnorm = enable_batchnorm

        self.net_img_3d = net_img_3d

        if self.enable_keyframe:
            self.net_img_2d = net_img_2d

        if self.enable_datiClinici:
            self.fc_datiClinici = nn.Linear(in_dim_datiClinici, fusion_dim)
            #if self.enable_batchnorm:
            #   self.bn = nn.BatchNorm1d(fusion_dim)

        self.n = 1
        if self.enable_doppiaAngolazione:
            self.n += 1
        if self.enable_datiClinici:
            self.n += 1
        if self.enable_keyframe:
            self.n += 1
            if self.enable_doppiaAngolazione:
                self.n += 1

        self.classifier_label = nn.Linear(self.n*fusion_dim, self.n*fusion_dim//2)
        self.classifier2_label = nn.Linear(self.n*fusion_dim//2, num_classes)
        
    def forward(self, inputs):
        if self.enable_doppiaAngolazione:
            if self.enable_datiClinici:
                if self.enable_keyframe:
                    imgs_3d, doppiaAngolazione_3d, datiClinici, imgs_2d, doppiaAngolazione_2d = inputs
                else:
                    imgs_3d, doppiaAngolazione_3d, datiClinici = inputs
            else:
                if self.enable_keyframe:
                    imgs_3d, doppiaAngolazione_3d, imgs_2d, doppiaAngolazione_2d = inputs
                else:
                    imgs_3d, doppiaAngolazione_3d = inputs
        else:
            if self.enable_datiClinici:
                if self.enable_keyframe:
                    imgs_3d, datiClinici, imgs_2d = inputs
                else:
                    imgs_3d, datiClinici = inputs
            else:
                if self.enable_keyframe:
                    imgs_3d, imgs_2d = inputs
                else:
                    #imgs = inputs
                    raise RuntimeError("All optional input branch disabled, so don't use this module.")
        
        output = self.net_img_3d(imgs_3d)
        o_img_label = output
            
        union_label = o_img_label

        if self.enable_doppiaAngolazione:
            output_doppiaAngolazione = self.net_img_3d(doppiaAngolazione_3d)
            o_img_label = output_doppiaAngolazione
            
            union_label = torch.cat( (union_label, o_doppiaAngolazione_label) , dim=1)

        if self.enable_datiClinici:
            o_datiClinici = self.fc_datiClinici(datiClinici)
            #if self.enable_batchnorm:
            #    o_datiClinici = self.bn(o_datiClinici)
            union_label = torch.cat( (union_label, o_datiClinici) , dim=1)

        if self.enable_keyframe:
            output_keyframe = self.net_img_2d(imgs_2d)
            o_keyframe_label = output_keyframe
            
            union_label = torch.cat( (union_label, o_keyframe_label) , dim=1)
            
            if self.enable_doppiaAngolazione:
                output_keyframe = self.net_img_2d(doppiaAngolazione_2d)
                o_doppiaAngolazione_keyframe_label = output_keyframe
                
                union_label = torch.cat( (union_label, o_doppiaAngolazione_keyframe_label) , dim=1)

        o_label = self.classifier2_label(F.relu(self.classifier_label(union_label)))

        return o_label


def reduceInputChannel(model_name, model):
    if (model_name == 'resnet3d') or (model_name == 'resnet3d_pretrained'):
        weight = model.stem[0].weight.clone()

        model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        with torch.no_grad():
            model.stem[0].weight = nn.Parameter(weight.mean(dim=1).unsqueeze(1))
    else:
        raise NotImplementedError("reduceInputChannel not implemented in actual model.")