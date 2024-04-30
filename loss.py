import torch
import torch.nn as nn


class SIoULoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()

        self.eps = eps

    def forward(self, box1, box2):
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps

        iou = inter / union

        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5

        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) + self.eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma

        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold,
                                sin_alpha_2, sin_alpha_1)

        angle_cost = torch.sin(torch.arcsin(sin_alpha) * 2)
        rho_x = (s_cw / (cw + self.eps)) ** 2
        rho_y = (s_ch / (ch + self.eps)) ** 2

        gamma = angle_cost - 2

        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)

        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + \
            torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        loss = (iou - 0.5 * (distance_cost + shape_cost)) + self.eps

        return torch.abs(loss.mean())


class YoloLoss(nn.Module):
    def __init__(self, C, box_loss):
        super().__init__()

        self.box_loss = box_loss
        self.no_object_loss = nn.BCEWithLogitsLoss()
        self.object_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.CrossEntropyLoss()

        self.C = C

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_no_obj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target):
        predictions = predictions.clone()
        target = target.clone()

        obj_pred = predictions[..., self.C:self.C + 1]
        obj_target = target[..., 0:1]

        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.no_object_loss(
            obj_pred[no_obj], obj_target[no_obj])

        # Object loss
        object_loss = self.object_loss(obj_pred[obj], obj_target[obj])

        # Box coordinates
        box_loss = self.box_loss(
            predictions[..., -4:][obj], target[..., 1:5][obj])

        # Class
        class_loss = self.class_loss(
            (predictions[..., :self.C][obj]), (target[..., 5][obj].long()),
        )

        loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_no_obj * no_object_loss
            + self.lambda_class * class_loss
        )

        return loss
