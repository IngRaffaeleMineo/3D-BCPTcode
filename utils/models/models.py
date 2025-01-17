'''
Copyright (c) R. Mineo, 2021-2023. All rights reserved.
This code was developed by R. Mineo in collaboration with PerceiveLab and other contributors.
For usage and licensing requests, please contact the owner.
'''

import torch
import torchvision
from torch import nn
from collections import OrderedDict

from utils.models import modules, i3d, s3d

from utils.models import mvcnn, gvcnn, vit, swin_transformer

def get_model(num_classes, model_name, enable_datiClinici, in_dim_datiClinici, enable_doppiaAngolazione, enable_keyframe, reduceInChannel, enableGlobalMultiHeadAttention, enableTemporalMultiHeadAttention, enableSpacialTemporalTransformerEncoder, numLayerTransformerEncoder, numHeadMultiHeadAttention, loss_weights):
    ''' returns the classifier '''

    if enableGlobalMultiHeadAttention and (model_name != "resnet3d" and model_name != "resnet3d_pretrained"):
        raise NotImplementedError("Multihead attention is available only in resnet3d/resnet3d_pretrained models.")
    if enableTemporalMultiHeadAttention and (model_name != "resnet3d" and model_name != "resnet3d_pretrained"):
        raise NotImplementedError("Temporal multihead attention is available only in resnet3d and resnet3d_pretrained models.")
    if enableSpacialTemporalTransformerEncoder and (model_name != "resnet3d" and model_name != "resnet3d_pretrained"):
        raise NotImplementedError("Spacial temporal trasformer encoder is available only in resnet3d and resnet3d_pretrained models.")

    if enable_doppiaAngolazione and (model_name == 'MVCNN'):
        raise RuntimeError("MVCNN needs two input views but not two input branch.")
    
    if enable_datiClinici or enable_doppiaAngolazione or enable_keyframe:
        if enable_datiClinici and (in_dim_datiClinici is None):
            raise RuntimeError("IN dim dati clinici non specificata")
        fusion_dim = 256
        num_classes_in = num_classes
        num_classes = fusion_dim
    
    if model_name == 'resnet3d':
        model = torchvision.models.video.r3d_18(pretrained=False, progress=True)
        if enableGlobalMultiHeadAttention:
            model.avgpool = nn.Sequential( # torch.Size([batch, 512, 8, 16, 16])
                # Global attention
                modules.GlobalMultiHeadAttention(embed_dim=512, num_heads=numHeadMultiHeadAttention, unflatten=(8,16,16)), # torch.Size([batch, 512, 8, 16, 16])
                # Reduce dims
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 128, 4, 8, 8]),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 2, 4, 4])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([4, 512])
            )
            output_feature_dim = 512
        elif enableTemporalMultiHeadAttention:
            model.avgpool = nn.Sequential( # torch.Size([batch, 512, 8, 16, 16])
                # Reduce feature
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(1,1,1)), # torch.Size([batch, 128, 8, 16, 16])
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(1,1,1)), # torch.Size([batch, 16, 8, 16, 16])
                # Temporal attention
                modules.TemporalMultiHeadAttention(embed_dim=16*16*16, num_heads=numHeadMultiHeadAttention, unflatten=(16,16,16)), # torch.Size([batch, 16, 8, 16, 16])
                # Reduce dims
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 4, 8, 8])
                nn.ReLU(),
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 2, 4, 4])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([4, 512])
            )
            output_feature_dim = 512
        elif enableSpacialTemporalTransformerEncoder:
            model.avgpool = nn.Sequential( # torch.Size([batch, 512, 8, 16, 16])
                # Reduce feature
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(1,1,1)), # torch.Size([batch, 128, 8, 16, 16])
                # Spacial Transformer
                modules.SpacialTransformer(d_model=128, num_heads=numHeadMultiHeadAttention, num_layers=numLayerTransformerEncoder, unflatten1=(16,16), unflatten0=(-1,8)), # torch.Size([batch*8, 16*16, 128])
                # Reduce feature
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(1,1,1)), # torch.Size([batch, 16, 8, 16, 16])
                # Temporal Transformer
                modules.TemporalTransformer(d_model=16*16*16, num_heads=numHeadMultiHeadAttention, num_layers=numLayerTransformerEncoder, unflatten=(16,16,16)), # torch.Size([batch, 8, 16*16*16])
                # Reduce dims
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 4, 8, 8])
                nn.ReLU(),
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 2, 4, 4])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([4, 512])
            )
            output_feature_dim = 512
        else:
            output_feature_dim = 512
        model.fc = modules.LastLayer(('linear', 'linear'),output_feature_dim,{'label':num_classes})
    elif model_name == 'resnet3d_pretrained':
        model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        if enableGlobalMultiHeadAttention:
            model.avgpool = nn.Sequential( # torch.Size([batch, 512, 8, 16, 16])
                # Global attention
                modules.GlobalMultiHeadAttention(embed_dim=512, num_heads=numHeadMultiHeadAttention, unflatten=(8,16,16)), # torch.Size([batch, 512, 8, 16, 16])
                # Reduce dims
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 128, 4, 8, 8]),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 2, 4, 4])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([4, 512])
            )
            output_feature_dim = 512
        elif enableTemporalMultiHeadAttention:
            model.avgpool = nn.Sequential( # torch.Size([batch, 512, 8, 16, 16])
                # Reduce feature
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(1,1,1)), # torch.Size([batch, 128, 8, 16, 16])
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(1,1,1)), # torch.Size([batch, 16, 8, 16, 16])
                # Temporal attention
                modules.TemporalMultiHeadAttention(embed_dim=16*16*16, num_heads=numHeadMultiHeadAttention, unflatten=(16,16,16)), # torch.Size([batch, 16, 8, 16, 16])
                # Reduce dims
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 4, 8, 8])
                nn.ReLU(),
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 2, 4, 4])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([4, 512])
            )
            output_feature_dim = 512
        elif enableSpacialTemporalTransformerEncoder:
            model.avgpool = nn.Sequential( # torch.Size([batch, 512, 8, 16, 16])
                # Reduce feature
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(1,1,1)), # torch.Size([batch, 128, 8, 16, 16])
                # Spacial Transformer
                modules.SpacialTransformer(d_model=128, num_heads=numHeadMultiHeadAttention, num_layers=numLayerTransformerEncoder, unflatten1=(16,16), unflatten0=(-1,8)), # torch.Size([batch*8, 16*16, 128])
                # Reduce feature
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(1,1,1)), # torch.Size([batch, 16, 8, 16, 16])
                # Temporal Transformer
                modules.TemporalTransformer(d_model=16*16*16, num_heads=numHeadMultiHeadAttention, num_layers=numLayerTransformerEncoder, unflatten=(16,16,16)), # torch.Size([batch, 8, 16*16*16])
                # Reduce dims
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 4, 8, 8])
                nn.ReLU(),
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, 16, 2, 4, 4])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([4, 512])
            )
            output_feature_dim = 512
        else:
            output_feature_dim = 512
        model.fc = modules.LastLayer(('linear', 'linear'),output_feature_dim,{'label':num_classes})
    elif model_name == 'resnetmc':
        model = torchvision.models.video.mc3_18(pretrained=False, progress=True)
        model.fc = modules.LastLayer(('linear', 'linear'),512,{'label':num_classes})
    elif model_name == 'resnetmc_pretrained':
        model = torchvision.models.video.mc3_18(pretrained=True, progress=True)
        model.fc = modules.LastLayer(('linear', 'linear'),512,{'label':num_classes})
    elif model_name == 'resnet21d':
        model = torchvision.models.video.r2plus1d_18(pretrained=False, progress=True)
        model.fc = modules.LastLayer(('linear', 'linear'),512,{'label':num_classes})
    elif model_name == 'resnet21d_pretrained':
        model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
        model.fc = modules.LastLayer(('linear', 'linear'),512,{'label':num_classes})
    elif model_name == 'i3d':
        #model = i3d.InceptionI3d(num_classes=400, spatial_squeeze=False, final_endpoint='Logits', in_channels=3, dropout_keep_prob=0.0)
        #model.logits = nn.Sequential(
        #    modules.Pad3D(),
        #    modules.LastLayer(('convolutional','convolutional'),1024,{'label':num_classes}),
        #    modules.Squeeze_3out([-1,-1]),
        #    modules.Interpolate_3out(scale_factor=1, mode='linear', align_corners=False),
        #    modules.Squeeze_3out([-1])
        #)
        model = i3d.InceptionI3d(num_classes=400, final_endpoint='Predictions', in_channels=3)
        model.replace_logits(num_classes)
    elif model_name == 'i3d_pretrained':
        #model = i3d.InceptionI3d(num_classes=400, spatial_squeeze=False, final_endpoint='Logits', in_channels=3, dropout_keep_prob=0.0)
        #model.load_state_dict(torch.load('utils/models/I3D_rgb_imagenet.pt'))
        #model.logits = nn.Sequential(
        #    modules.Pad3D(),
        #    modules.LastLayer(('convolutional','convolutional'),1024,{'label':num_classes},multibranch_dropout),
        #    modules.Squeeze_3out([-1,-1]),
        #    modules.Interpolate_3out(scale_factor=1, mode='linear', align_corners=False),
        #    modules.Squeeze_3out([-1])
        #)
        model = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Predictions', in_channels=3, dropout_keep_prob=0.5)
        model.load_state_dict(torch.load('utils/models/I3D_rgb_imagenet.pt'))
        model.replace_logits(num_classes)
    elif model_name == 's3d':
        #model = s3d.S3D(num_class=400, last_mean=False)
        #model.fc = nn.Sequential(
        #    modules.LastLayer(('convolutional','convolutional'),1024,{'label':num_classes},multibranch_dropout),
        #    modules.Mean_3out()
        #)
        model = s3d.S3D(num_class=400, last_mean=True)
        model.fc = nn.Sequential(nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),)
    elif model_name == 's3d_pretrained':
        #model = s3d.S3D(num_class=400, last_mean=False)
        model = s3d.S3D(num_class=400, last_mean=True)
        weight_dict = torch.load('utils/models/S3D_kinetics400.pt')
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    raise RuntimeError(' size? ' + name, param.size(), model_dict[name].size())
            else:
                raise RuntimeError(' name? ' + name)
        weight_dict = torch.load('utils/models/S3D_kinetics400.pt')
        #model.fc = nn.Sequential(
        #    modules.LastLayer(('convolutional','convolutional'),1024,{'label':num_classes},multibranch_dropout),
        #    modules.Mean_3out()
        #)
        model.fc = nn.Sequential(nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),)
    elif model_name == 'MVCNN':
        model = mvcnn.Model()
    elif model_name == 'GVCNN':
        model = gvcnn.GVCNN(nclasses=2, num_views=2)
    elif model_name == 'ViT_B_16':
        model = vit.VisionTransformer(config='ViT-B_16', img_size=128, in_channels=3, zero_head=True, num_classes=2, loss_weights = loss_weights, multi_stage_classification=False, multi_layer_classification=False)
    elif model_name == 'ViT_3D':
        model = vit.VisionTransformer(config='ViT-B_16', img_size=256, in_channels=3, zero_head=True, num_classes=2, loss_weights = loss_weights, multi_stage_classification=False, multi_layer_classification=False)#ViT-MRI
    elif model_name == 'VideoSwinTransformer': # batch*3*frames*224*224
        model = swin_transformer.SwinTransformer3D(pretrained=None,
                                                        pretrained2d=True,
                                                        embed_dim=128,
                                                        depths=[2, 2, 18, 2],
                                                        num_heads=[4, 8, 16, 32],
                                                        patch_size=(2,4,4),
                                                        window_size=(16,7,7),
                                                        drop_path_rate=0.4,
                                                        patch_norm=True)
        # https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py
        checkpoint = torch.load('utils\\models\\swin_base_patch244_window1677_sthv2.pth')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        #dummy_x = torch.rand(1, 3, 32, 224, 224)
        #logits = model(dummy_x)
        #print(logits.shape) # torch.Size([1, 1024, 16, 7, 7])
        model = nn.Sequential(model,
                                nn.AdaptiveAvgPool3d((5,2,2)),
                                modules.View(2),
                                nn.Linear(1024*5*2*2,1024),
                                nn.ReLU(),
                                nn.Linear(1024,2))
    else:
        raise RuntimeError('Model name not found!')

    if reduceInChannel:
        modules.reduceInputChannel(model_name, model)
    

    if enable_datiClinici or enable_doppiaAngolazione or enable_keyframe:
        model_aggiuntivo = None
        if enable_keyframe:
            model_aggiuntivo = torchvision.models.resnext50_32x4d(pretrained=True, progress=True) # weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
            if reduceInChannel:
                weight2 = model_aggiuntivo.conv1.weight.clone()
                model_aggiuntivo.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    model_aggiuntivo.conv1.weight = nn.Parameter(weight2.mean(dim=1).unsqueeze(1))
            model_aggiuntivo.fc = modules.LastLayer(('linear', 'linear'),False,0,512*4,{'label':num_classes})
        
        model = modules.doppiaAngolazione_datiClinici_keyframe_LateFusion(enable_doppiaAngolazione=enable_doppiaAngolazione, enable_datiClinici=enable_datiClinici, enable_keyframe=enable_keyframe, net_img_3d=model, net_img_2d=model_aggiuntivo, fusion_dim=fusion_dim, in_dim_datiClinici=in_dim_datiClinici, num_classes=num_classes_in)

    return model