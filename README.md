# ssn-pytorch
PyTorch implementation of Superpixel Sampling Networks  
paper: https://arxiv.org/abs/1807.10174  
original code: https://github.com/NVlabs/ssn_superpixels

# Requirements
- PyTorch >= 1.4
- scikit-image

# Usage
## inference
SSN_pix
```
python inference --image /path/to/image
```
SSN_deep
```
python inference --image /path/to/image --weight /path/to/pretrained_weight
```

## training
```
python train.py --root /path/to/BSDS500
```
