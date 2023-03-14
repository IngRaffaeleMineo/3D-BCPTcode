<div align="center">

# 3D-BCPTcode: A 3D Basic Classification Parallelized Training code
Raffaele Mineo

<!---[![Paper]()]()-->

</div>

# How to run
The code expects a JSON file in the format support by MONAI, passed via the --split_path argument, with the following structure:
```
{
 "num_fold": <N>, 
 "fold0": { "train": [ {"image": <path>,
                        "image2": <path>,
                        "label": <class>},
                       ...
                       {"image": <path>,
                        "image2": <path>,
                        "label": <class>}
                      ],
            "val": [ {"image": <path>,
                        "image2": <path>,
                        "label": <class>},
                      ...],
            "test": [{"image": <path>,
                      "image2": <path>,
                      "label": <class>},
                      ...]}, 
 "fold1": { "train": [...],
            "val": [...],
            "test": [...]},
...
}
```
`<N>`, `<path>` and `<class>` fields should be filled as appropriate.
Each path should point to a `.npy`file containing a 3D (2D+T) tensor, representing a video.

## Pre-requisites:
- NVIDIA GPU (Tested on Nvidia Tesla T4 GPUs )
- [Requirements](requirements.txt)
- Please download this 3 pretrained models into utils\models: \
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt (rename as i3d_rgb_imagenet.pt) \
https://drive.google.com/uc?export=download&id=1HJVDBOQpnTMDVUM3SsXLy0HUkf_wryGO \
https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth

## Train Example
 To start training, simply run (using default arguments):
 
 ```python train.py --root_dir='<dataset_path>' --split_path='<split_json_path>'```
 
To start distributed training, use:

```
python -m torch.distributed.launch --nproc_per_node=<N_GPUS> --use_env train.py --root_dir='<dataset_path>' --split_path='<split_json_path>'
```

## Test Example
To start evaluation, simply run (using default arguments):

```python test.py --logdir='<log_path>' --start_tornado_server=1 --enable_explainability=1```

Log directories are automatically created upon training inside a `logs` directory.

To start distributed testing, use:

```
python -m torch.distributed.launch --nproc_per_node=<N_GPUS> --use_env test.py --logdir='<log_path>' --start_tornado_server=1 --enable_explainability=1
```

## Notes
Please, remember to insert the following acknowledgement in your code, thanks: `This code is taken from https://github.com/IngRaffaeleMineo/3D-BCPTcode and modified to our purposes.`

## Acknowledgements
https://github.com/MECLabTUDA/M3d-Cam \
https://github.com/jacobgil/pytorch-grad-cam
