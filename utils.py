from typing import Counter
import torch
from torch.nn import functional
from torchmetrics import R2Score

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

    return intersection / (box1_area + box2_area - intersection + 1e-10)


def plot_image(image_tensor, boxes_tensor, true_boxes):
    image_tensor = image_tensor.cpu()
    # boxes_tensor = boxes_tensor.cpu()

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
            edgecolor="r",
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
            edgecolor="green",
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
        self.iou_threshold = cfg.iou_threshold
        self.threshold = cfg.threshold
        self.calculate_map = calculate_map

        self.r2_score = R2Score().to(device)

        self.device = device

        self.idx = 0

    def step(self, prediction, ground_truth, loss):
        # Format class_id form 0 zo C, obj/no obj, bbox
        prediction = prediction.clone()

        # Format obj/no obj, bbox, class_id
        ground_truth = ground_truth.clone()

        if self.calculate_map:
            self.add_boxes(ground_truth, prediction)

        self.losses.append(loss.item())

        presence = ground_truth[..., 0] == 1

        self.r2_score.update((prediction[..., -4:] * presence.unsqueeze(-1).float(
        )).view(-1), (ground_truth[..., -4:] * presence.unsqueeze(-1).float()).view(-1))

    def add_boxes(self, ground_truth, prediction):
        ground_truth = transform_to_prediction_format(ground_truth, self.C)

        true_bboxes = cell_boxes_to_boxes(ground_truth, self.cfg)
        bboxes = cell_boxes_to_boxes(prediction, self.cfg)

        batch_size = prediction.shape[0]

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=self.iou_threshold,
                threshold=self.threshold
            )

            for nms_box in nms_boxes:
                self.all_pred_boxes.append([self.idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > self.threshold:
                    self.all_true_boxes.append([self.idx] + box)

            self.idx += 1

    def get_average_loss(self):
        total_loss = sum(self.losses) / len(self.losses)
        return total_loss

    def get_r2_score(self):
        return self.r2_score.compute().item()

    def get_map(self):
        return mean_average_precision(
            self.all_pred_boxes, self.all_true_boxes, self.iou_threshold, self.C
        )

    def get_metrics(self):
        metrics = []

        metrics.append(['Average Loss', self.get_average_loss()])
        metrics.append(['R2 Score', self.get_r2_score()])

        if self.calculate_map:
            metrics.append(['Average mAP', self.get_map()])

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
    converted_pred[..., 0] = converted_pred[..., 0].long()
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
    predictions = predictions.reshape(batch_size, S, S, C + 5)

    bboxes = predictions[..., -4:]
    scores = predictions[..., C]

    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes * (1 - best_box)

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]

    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    confidence = predictions[..., C].unsqueeze(-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)

    converted_preds = torch.cat(
        (predicted_class, confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def non_max_suppression(bboxes, iou_threshold, threshold):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold, num_classes
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
