"""
MindSpore版本的视频数据增强
"""
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip, label=None, file_id=None):
        for t in self.transforms:
            clip = t(clip)
        return clip, label


class CenterCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, clip):
        """
        Args:
            clip: list of images (H, W, C)
        Returns:
            cropped clip
        """
        if len(clip) == 0:
            return clip

        h, w = clip[0].shape[:2]
        th, tw = self.size

        # 如果目标尺寸大于原始尺寸，则使用原始尺寸
        if th >= h:
            th = h
        if tw >= w:
            tw = w

        # 计算中心裁剪的起始位置
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))

        return [img[i:i + th, j:j + tw] for img in clip]


class RandomCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, clip):
        if len(clip) == 0:
            return clip
        
        h, w = clip[0].shape[:2]
        th, tw = self.size
        
        # 随机裁剪位置
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        
        return [img[i:i+th, j:j+tw] for img in clip]


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, clip):
        if np.random.random() < self.prob:
            return [cv2.flip(img, 1) for img in clip]
        return clip


class ToTensor:
    def __call__(self, clip):
        """
        将图像列表转换为Tensor
        
        Args:
            clip: list of images (H, W, C) in RGB format
        Returns:
            tensor: (T, C, H, W) - 与PyTorch版本保持一致
        """
        if len(clip) == 0:
            return Tensor(np.array([]))
        
        # 转换为numpy数组
        # clip: list of (H, W, C) -> (T, H, W, C)
        clip_array = np.array(clip)
        
        # (T, H, W, C) -> (T, C, H, W) - 修正为与PyTorch一致
        clip_array = np.transpose(clip_array, (0, 3, 1, 2))
        
        # 转换为float32
        clip_array = clip_array.astype(np.float32)
        
        return Tensor(clip_array, dtype=ms.float32)


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, clip):
        """
        调整图像大小
        
        Args:
            clip: list of images
        Returns:
            resized clip
        """
        return [cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR) 
                for img in clip]


def normalize_video(video_tensor):
    """
    标准化视频tensor
    
    Args:
        video_tensor: (C, T, H, W)
    Returns:
        normalized tensor
    """
    if hasattr(video_tensor, 'asnumpy'):
        video_array = video_tensor.asnumpy()
    else:
        video_array = video_tensor
    
    # 归一化到[-1, 1]
    video_array = video_array / 127.5 - 1.0
    
    return Tensor(video_array, dtype=ms.float32)
