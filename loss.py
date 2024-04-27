import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.box_loss = nn.HuberLoss()
        self.no_object_loss = nn.BCEWithLogitsLoss()
        self.object_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

        self.sigmoid = nn.Sigmoid()

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

        # # No object loss
        no_object_loss = self.no_object_loss(
            (obj_pred[no_obj]), (obj_target[no_obj]),
        )

        # # Object loss
        box_pred = predictions[..., -4:]

        iou = intersection_over_union(
            box_pred[obj], target[..., 1:5][obj]).detach()

        object_loss = self.object_loss(self.sigmoid(
            obj_pred[obj]), (iou * obj_target[obj]))

        # # Box coordinates
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
