import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F
import torch
from typing import Tuple

class LetterboxResize(T.Transform):
    """Custom transform to pad to square and resize while preserving aspect ratio."""
    def __init__(self, target_size: int):
        super().__init__()
        self.target_size = target_size

    def _get_params(self, flat_inputs):
        return {}

    def _transform(self, inpt, params):
        img = inpt  # Only handling image (not boxes here)

        if not F.is_tensor_image(img):
            raise TypeError("Expected a tensor image.")

        _, h, w = img.shape
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        img = F.resize(img, [new_h, new_w], antialias=True)

        # Compute padding
        pad_left = (self.target_size - new_w) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_top = (self.target_size - new_h) // 2
        pad_bottom = self.target_size - new_h - pad_top

        # Pad to target size
        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

        return img

# Create transform pipeline
def get_v2_transform_pipeline(target_size=320):
    return T.Compose([
        T.ToImageTensor(),  # Converts from PIL/ndarray to Tensor [C, H, W]
        T.ConvertImageDtype(torch.float32),  # Normalize to [0, 1]
        LetterboxResize(target_size),
    ])
