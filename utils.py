import torch
from torch.nn import functional
from torchmetrics import R2Score

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.detection import MeanAveragePrecision


def plot_image(image_tensor, true_boxes, boxes_tensor):
    image_tensor = image_tensor.cpu()

    image_np = image_tensor.permute(1, 2, 0).numpy()

    img_height, img_width = image_tensor.shape[1], image_tensor.shape[2]

    _fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    for box in boxes_tensor:
        box = box[2:]

        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * img_width, upper_left_y * img_height),
            box[2] * img_width,
            box[3] * img_height,
            linewidth=1,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)

    for box in true_boxes:
        box = box[2:]

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
    def __init__(self, device, cfg, calculate_map=False):
        self.losses = []

        self.all_pred_boxes = []
        self.all_true_boxes = []

        self.cfg = cfg
        self.C = cfg.C
        self.S = cfg.S
        self.calculate_map = calculate_map

        self.r2_score = R2Score().to(device)

        if calculate_map:
            self.mAP = MeanAveragePrecision(
                box_format='xywh', iou_type="bbox").to(device)

        self.device = device

        self.idx = 0

    def step(self, prediction, ground_truth, loss):
        # Format class_id form 0 zo C, obj/no obj, bbox
        prediction = prediction.clone()

        # Format obj/no obj, bbox, class_id
        ground_truth = ground_truth.clone()

        if self.calculate_map:
            self.update_map(ground_truth, prediction)

        self.losses.append(loss.item())

        presence = ground_truth[..., 0] == 1
        self.r2_score.update((prediction[..., -4:] * presence.unsqueeze(-1).float(
        )).view(-1), (ground_truth[..., -4:] * presence.unsqueeze(-1).float()).view(-1))

    def update_map(self, ground_truth, prediction):
        ground_truth = transform_to_prediction_format(ground_truth, self.C)

        true_bboxes = cell_boxes_to_boxes(ground_truth, self.cfg)
        bboxes = cell_boxes_to_boxes(prediction, self.cfg)

        bboxes = torch.tensor(bboxes)
        true_bboxes = torch.tensor(true_bboxes)

        pred = []
        target = []

        for box in bboxes:
            pred.append({
                "boxes": box[..., 2:],
                "scores": box[..., 1],
                "labels": box[..., 0].int(),
            })

        for box in true_bboxes:
            target.append({
                "boxes": box[..., 2:],
                "scores": box[..., 1],
                "labels": box[..., 0].int(),
            })

        self.mAP.update(pred, target)

    def get_average_loss(self):
        total_loss = sum(self.losses) / len(self.losses)
        return total_loss

    def get_r2_score(self):
        return self.r2_score.compute().item()

    def get_map(self):
        return self.mAP.compute()

    def get_metrics(self):
        metrics = []

        metrics.append(['Average Loss', self.get_average_loss()])
        metrics.append(['R2 Score', self.get_r2_score()])

        if self.calculate_map:
            result = self.get_map()
            metrics.append(['Average mAP 50', result['map_50'].item()])
            metrics.append(['Average mAP 70', result['map_75'].item()])

        return metrics


def transform_to_prediction_format(ground_truth, C):
    new_g = ground_truth.clone()

    # Only on present obj
    presence = ground_truth[..., 0] == 1
    class_ids = ground_truth[..., -1].long()

    one_hot = functional.one_hot(class_ids, num_classes=C) * \
        presence.unsqueeze(-1).float()

    # Append the one hot encoding of classes at the start
    # of elements in the last dim

    return torch.cat((one_hot.float(), new_g[..., :-1]), dim=-1)


def cell_boxes_to_boxes(out, cfg):
    converted_pred = convert_cell_boxes(out, cfg.S, cfg.C).reshape(
        out.shape[0], cfg.S * cfg.S, -1)
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(cfg.S * cfg.S):
            bboxes.append([x.item()
                           for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def convert_cell_boxes(predictions, S, C):
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]

    bboxes = predictions[..., -4:]

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    x = 1 / S * (bboxes[..., :1] + cell_indices)
    y = 1 / S * (bboxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * bboxes[..., 2:4]

    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    confidence = predictions[..., C].unsqueeze(-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)

    converted_preds = torch.cat(
        (predicted_class, confidence, converted_bboxes), dim=-1
    )

    return converted_preds
