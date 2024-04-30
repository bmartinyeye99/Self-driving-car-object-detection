
import torch
from torchsummary import summary
from model import Model
from datamodule import DataModule
from resnet50_yolo_model import ResNet50
from trainer import Trainer
from yolo_model import Yolov1


class Experiment:
    def __init__(self, cfg, originalYolo=False, resnet50=False):
        # Create model & datamodule
        self.datamodule = DataModule(cfg)
        self.model = self._create_model(cfg, originalYolo, resnet50)

        # Create trainer
        self.trainer = Trainer(cfg, self.model)

    def _create_model(self, cfg, originalYolo, resnet50):

        if not originalYolo and not resnet50:
            model = Model(cfg)

        elif originalYolo:
            model = Yolov1(cfg)

        elif resnet50:
            model = ResNet50(cfg)

        device = torch.device(cfg.device)
        model = model.to(device, memory_format=torch.channels_last)

        print("Creating model: ")
        summary(
            model,
            input_size=(cfg.chin, cfg.image_width, cfg.image_height),
            batch_size=cfg.batch_size,
            device=cfg.device
        )
        return model

    def train(self):
        self.trainer.setup(datamodule=self.datamodule)
        self.trainer.fit()
