
import torch
from torchsummary import summary
from model import Model
from datamodule import DataModule
from trainer import Trainer


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        # Create model & datamodule
        self.datamodule = DataModule()
        self.model = self._create_model(cfg)

        # Create trainer
        self.trainer = Trainer(cfg, self.model)

    def _create_model(self, cfg):
        model = Model(chin=cfg.chin, channels=cfg.channels,
                      num_hidden=cfg.num_hidden, dropout_rate=cfg.dropout_rate, negative_slope=cfg.negative_slope)

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
