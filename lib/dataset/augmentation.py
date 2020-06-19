import cv2
import numpy as np
import random


class Compose:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        for aug in self.augmentations:
            data = aug(data)
        return data


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            # call copy() to avoid negative stride error in torch.from_numpy
            data = [d[:, ::-1].copy() for d in data]
        return data


class RandomScale:
    def __init__(self, scale_range=(0.75, 3.0)):
        self.scale_range = scale_range

    def __call__(self, data):
        rand_factor = np.random.normal(1, 0.75)
        scale = np.min((self.scale_range[1], rand_factor))
        scale = np.max((self.scale_range[0], scale))
        data = [
            cv2.resize(d, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR if d.dtype == np.float32 else cv2.INTER_NEAREST)
            for d in data]
        return data


class RandomCrop:
    def __init__(self, crop_size=(200, 200)):
        self.crop_size = crop_size

    def __call__(self, data):
        height, width = data[0].shape[:2]
        c_h, c_w = self.crop_size
        assert height >= c_h and width >= c_w, f"({height}, {width}) v.s. ({c_h}, {c_w})"
        left = random.randint(0, width - c_w)
        top = random.randint(0, height - c_h)
        data = [d[top:top+c_h, left:left+c_w] for d in data]
        return data
