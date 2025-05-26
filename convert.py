import torch
import torch.nn.functional as F
from torchvision.ops import nms
import numpy as np
from onnx2torch import convert
import onnx


class YOLOv5Wrapper(torch.nn.Module):
    def __init__(self, model, conf_threshold=0.25, iou_threshold=0.45):
        super().__init__()
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = 18

    def forward(self, x):
        with torch.no_grad():
            out = self.model(x)  # output shape: (1, 22, 2100)
            if isinstance(out, (list, tuple)):
                out = out[0]

            out = out.permute(0, 2, 1)  # shape: (1, 2100, 22)
            boxes = out[..., :4]  # cx, cy, w, h
            class_logits = out[..., 4:]  # shape: (1, 2100, 18)
            scores = torch.sigmoid(class_logits)

            results = []
            for b in range(x.shape[0]):
                boxes_b = boxes[b]
                scores_b = scores[b]

                # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
                cx, cy, w, h = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # For each class, apply confidence threshold and NMS
                all_detections = []
                for cls in range(self.num_classes):
                    class_scores = scores_b[:, cls]
                    mask = class_scores > self.conf_threshold
                    if mask.sum() == 0:
                        continue

                    selected_boxes = boxes_xyxy[mask]
                    selected_scores = class_scores[mask]
                    selected_labels = torch.full_like(selected_scores, cls, dtype=torch.int64)

                    keep = nms(selected_boxes, selected_scores, self.iou_threshold)
                    all_detections.append(torch.cat([
                        selected_boxes[keep],
                        selected_scores[keep].unsqueeze(1),
                        selected_labels[keep].unsqueeze(1).float()
                    ], dim=1))

                if all_detections:
                    detections = torch.cat(all_detections, dim=0)  # [x1, y1, x2, y2, conf, class]
                else:
                    detections = torch.zeros((0, 6), device=x.device)

                results.append(detections)

            return results




onnx_model = onnx.load("best.onnx")
torch_model = convert(onnx_model)

wrapped_model = YOLOv5Wrapper(torch_model)

# Dummy input test
dummy_input = torch.randn(1, 3, 320, 320)
results = wrapped_model(dummy_input)

# [x1, y1, x2, y2, confidence, class_id]
print(results)
