import torch
import torchvision.transforms as TF
from dataset import NumpyToTensor, create_dataset


class DataModule:
    def __init__(self):

        self.input_transform = TF.Compose([
            NumpyToTensor(),
        ])

        self.dataset_train, self.dataset_val, self.dataset_test, self.dataset_draw = create_dataset(
            self.input_transform)

    def setup(self, cfg):
        self.dataloader_train = torch.utils.data.dataloader.DataLoader(
            self.dataset_train,
            batch_size=cfg.batch_size,    # Batch size hyper-parameter
            shuffle=True,                 # Iterate over samples in random order
            pin_memory=cfg.pin_memory,    # Should be faster when pin memory set to true
            num_workers=cfg.num_workers   # Parallel processing of input samples
        )

        self.dataloader_valid = torch.utils.data.dataloader.DataLoader(
            self.dataset_val,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers
        )

        self.dataloader_test = torch.utils.data.dataloader.DataLoader(
            self.dataset_test,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers
        )

        self.dataloader_draw = torch.utils.data.dataloader.DataLoader(
            self.dataset_draw,
            batch_size=1,                 # Batch size hyper-parameter
            shuffle=True,                 # Iterate over samples in random order
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers   # Parallel processing of input samples
        )
