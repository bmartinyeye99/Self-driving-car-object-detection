import torch
from torchsummary import summary
from datamodule import DataModule


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        # Create model & datamodule
        self.datamodule = DataModule()