# CoDL

## Introduction

This is the official implementation of **CoDL: Efficient CPU-GPU Co-execution for Deep Learning Inference on Mobile Devices** in The 20th Annual International Conference on Mobile Systems, Applications and Services, which is a novel framework for co-execution of deep learning models on mobile devices. 

CoDL can fully utilize the heterogeneous processors to accelerate each operator of a model, which makes it different from available inference frameworks. CoDL integrates two novel techniques: 1) **hybrid-type-friendly data sharing**, which allows each processor to use its efficient data type for inference. To reduce data sharing overhead, we also propose hybrid-dimension partitioning and operator chain methods; 2) **non-linearity- and concurrency-aware latency prediction**, which can direct proper operator partitioning by building an extremely light-weight but accurate latency predictor for different processors.

We evaluate CoDL on a variety of smartphones and deep learning models. The inference speed of CoDL achieves 680ms on RetinaFace, 140ms on YOLOv2, 137ms on VGG-16, 244ms on PoseNet, and 267ms on Fast Style Transfer in our Snapdragon 855 platform (Xiaomi 9).

Below is the list of all the models and their performance on CoDL.

| Platform | Model | Inference Time |
| :---: | :---: | :---: |
| Snapdragon 855 | RetinaFace | 680ms |
|  | YOLOv2 | 140ms |
|  | VGG-16 | 137ms |
|  | PoseNet | 244ms |
|  | Fast Style Transfer | 267ms |
| Snapdragon 865 | RetinaFace | 551ms |
|  | YOLOv2 | 123ms |
|  | VGG-16 | 121ms |
|  | PoseNet | 201ms |
|  | Fast Style Transfer | 251ms |
| Snapdragon 888 | RetinaFace | 558.37ms |
|  | YOLOv2 | 119.82ms |
|  | VGG-16 | 107.94ms |
|  | PoseNet | 225.63ms |
|  | Fast Style Transfer | 227.83ms |
| Kirin990 (buffer-based) | RetinaFace | 804ms |
|  | YOLOv2 | 155ms |
|  | VGG-16 | 141ms |
|  | PoseNet | 257ms |
|  | Fast Style Transfer | 679ms |

Any questions are welcome. Our paper can be found [here](CoDL.pdf).

## Installation

1. For building execution files, please read and follow the instruction in [codl-mobile/README.md](codl-mobile/README.md).
2. For evaluating in your smartphones, please read and follow the instruction in [codl-eval-tools/README.md](codl-eval-tools/README.md).

## Citation

```
@inproceedings{jia2022codl,
    author = {Jia, Fucheng and Zhang, Deyu and Cao, Ting and Jiang, Shiqi and Liu, Yunxin and Ren, Ju and Zhang, Yaoxue},
    title = {CoDL: Efficient CPU-GPU Co-execution for Deep Learning Inference on Mobile Devices},
    year = {2022},
    publisher = {ACM},
    url = {https://doi.org/10.1145/3498361.3538932},
    doi = {10.1145/3498361.3538932},
    booktitle = {The 20th Annual International Conference on Mobile Systems, Applications and Services (MobiSys '22)},
}
```