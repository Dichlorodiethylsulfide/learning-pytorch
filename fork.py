import torch

# Used for improving the performance of the Robust Video Matting model
# This code is used in one of my projects, which uses PyTorch to key out people without a green screen.

# Module that loads and applies the Robust Video Matting model
class RVMModule(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.model = torch.jit.load(str(model_path))
        self.model.to(device)
        self.model = torch.jit.script(self.model)

    # Syntax copied from https://github.com/PeterL1n/RobustVideoMatting/blob/master/model/model.py
    def forward(self, camera, 
                r1: torch.Optional[torch.Tensor] = None,
                r2: torch.Optional[torch.Tensor] = None,
                r3: torch.Optional[torch.Tensor] = None,
                r4: torch.Optional[torch.Tensor] = None,
                downsample_ratio: float = 1):
        outputs = self.model(camera, r1, r2, r3, r4, downsample_ratio)
        return outputs

# Wrapper that forks the work and returns a future
class RVMForkModule(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.rvm = RVMModule(model_path, device)
        
    def forward(self, camera,
                r1: torch.Optional[torch.Tensor] = None,
                r2: torch.Optional[torch.Tensor] = None,
                r3: torch.Optional[torch.Tensor] = None,
                r4: torch.Optional[torch.Tensor] = None,
                downsample_ratio: float = 1):
        fut = torch.jit.fork(self.rvm, camera, r1, r2, r3, r4, downsample_ratio)
        return fut
