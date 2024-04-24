import torch
from torchmetrics import R2Score
from torchvision.ops import box_iou

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(image_tensor, boxes_tensor):
    image_np = image_tensor.permute(1, 2, 0).numpy()

    img_height, img_width = image_tensor.shape[1], image_tensor.shape[2]

    _fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    for box in boxes_tensor:
        # Format class_id, x_min, y_min, x_max, y_max
        x_min = box[-4] * img_width
        y_min = box[-3] * img_height
        x_max = box[-2] * img_width
        y_max = box[-1] * img_height

        width = x_max - x_min
        height = y_max - y_min

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
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
