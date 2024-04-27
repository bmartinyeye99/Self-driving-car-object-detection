import torch
from torchmetrics import R2Score
from torchvision.ops import box_iou

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def iou_wh(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] +
        boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_pred, boxes_labels):

    box1_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
    box1_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
    box1_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3] / 2
    box1_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def plot_image(image_list, boxes_list):
    for image_tensor, boxes_tensor in zip(image_list, boxes_list):
        image_np = image_tensor.permute(1, 2, 0).numpy()

        img_height, img_width = image_tensor.shape[1], image_tensor.shape[2]

        _fig, ax = plt.subplots(1)
        ax.imshow(image_np)

        for box in boxes_tensor:
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2

            rect = patches.Rectangle(
                (upper_left_x * img_width, upper_left_y * img_height),
                box[2] * img_width,
                box[3] * img_height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        plt.show()


class Statistics:
    def __init__(self, device):
        self.ground_truths = []
        self.predictions = []
        self.losses = []

        self.r2_score = R2Score().to(device)

    def step(self, ground_truth, prediction, loss):

        ground_truth, prediction, loss = ground_truth, prediction, loss.unsqueeze(
            0)

        self.ground_truths.append(ground_truth)
        self.predictions.append(prediction)
        self.losses.append(loss)

        self.r2_score.update(prediction.view(-1), ground_truth.view(-1))

    def get_average_loss(self):
        total_loss = torch.cat(self.losses).mean()
        return total_loss.item()

    def get_r2_score(self):
        return self.r2_score.compute().item()

    def get_average_iou(self):
        predictions = torch.cat(self.predictions, dim=0)
        ground_truths = torch.cat(self.ground_truths, dim=0)

        # Process IoU in batches to manage GPU memory usage
        batch_size = 100  # Set batch size based on your GPU memory capacity
        total_iou = 0
        count = 0

        for i in range(0, predictions.shape[0], batch_size):
            pred_batch = predictions[i:i + batch_size]
            gt_batch = ground_truths[i:i + batch_size]

            # Calculate IoU for the current batch
            iou = box_iou(pred_batch, gt_batch)
            total_iou += iou.sum().item()  # Summing all IoUs in the batch
            count += iou.numel()  # Counting the total number of IoU calculations

        # Return the average IoU
        return total_iou / count if count != 0 else 0

    def get_metrics(self):
        metrics = []
        metrics.append(['Average Loss', self.get_average_loss()])
        metrics.append(['R2 Score', self.get_r2_score()])
        metrics.append(['Average IoU', self.get_average_iou()])

        return metrics
