# 3D-BCPTcode (3D - Basic Classification Parallelized Training code)

# How to run
The code expects ...:
```
{
 "num_fold": #N, 
 "fold0": { "train": [ {"image": #path,
                        "image2": #path,
                        "label": #class},
                       ...
                       {"image": #path,
                        "image2": #path,
                        "label": #class}
                      ],
            "val": [ {"image": #path,
                        "image2": #path,
                        "label": #class},
                      ...],
            "test": [{"image": #path,
                      "image2": #path,
                      "label": #class},
                      ...]}, 
 "fold1": { "train": [],
            "val": [],
            "test": []},
 ....
 "fold#N": { "train": [],
            "val": [],
            "test": []},
}

```

## Pre-requisites:
- NVIDIA GPU (Tested on Nvidia Tesla T4 GPUs )
- [Requirements](requirements.txt)
- Please download this 3 pretrained models into utils\models: \
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt (rename as i3d_rgb_imagenet.pt) \
https://drive.google.com/uc?export=download&id=1HJVDBOQpnTMDVUM3SsXLy0HUkf_wryGO \
https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth \

## Train Example
To start training use "python train.py" (or if distribuited "python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env train.py" --root_dir='data\MY_DATASET' --split_path='data\MY_DATASET_BNCV5F.json'

## Test Example
To start evaluation only use "python test.py" (or if distribuited "python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env test.py" --logdir='logs\2022-01-01_00-00-00_resnet3d_pretrained' --start_tornado_server=1 --enable_explainability=1

<!--- ## Notes --->

Acknowledgements \
https://github.com/MECLabTUDA/M3d-Cam \
https://github.com/jacobgil/pytorch-grad-cam
