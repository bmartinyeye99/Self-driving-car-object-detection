import torch
import torchvision.transforms as TF
from dataset import NumpyToTensor, create_dataset
import albumentations as A


class DataModule:
    def __init__(self, stride, classes):

        self.input_transform = TF.Compose([
            NumpyToTensor(),
        ])

        self.augmentation_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.ChannelShuffle(p=0.1),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4))

        self.dataset_train, self.dataset_val, self.dataset_test, self.dataset_draw = create_dataset(
            stride, classes, self.input_transform, self.augmentation_transform)

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
