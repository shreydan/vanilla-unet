# Vanilla U-NET

U-Net implementation in PyTorch from scratch.

requirements:
```
torch
torchvision
```

Note:

- For cropping as required in the paper, the method I used is center-crop
- Crop enabled (default) : the skip connections will be cropped as per the upscaled image and the final result will be smaller than the input image.
- Crop disabled : the upscaled images will be padded as per the skip connections and the final result will be the same size as the input image. 

## Paper

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Architecture

![U-Net Architecture](architecture.png)