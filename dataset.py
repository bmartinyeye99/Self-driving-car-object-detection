import subprocess
import torch
from typing import Callable
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import zipfile
import os

scratch_directory = '.scratch/'
data_directory = f"{scratch_directory}data"

annotation_file = 'labels_trainval.csv'
annotation_location = f"{data_directory}/{annotation_file}"
images_path = f"{data_directory}/images"

zip_file = 'self-driving-cars.zip'
dataset_name = 'alincijov/self-driving-cars'
kaggle_command = f"kaggle datasets download -d {
    dataset_name} -p {scratch_directory}/"


image_original_width, image_original_height = 480, 300


class NumpyToTensor(Callable):
    def __call__(self, x):
        x = np.transpose(x, axes=(2, 0, 1))     # HWC -> CHW
        x = torch.from_numpy(x) / 255.0         # <0;255>UINT8 -> <0;1>

        return x.float()                        # cast as 32-bit flow


def unzip():
    with zipfile.ZipFile(f".scratch/{zip_file}", 'r') as zip_ref:
        zip_ref.extractall(data_directory)


def load_dataset():
    # Download dataset if it isn't already
    if not os.path.exists(data_directory):
        print(f"'{data_directory}' folder does not exist. Downloading data...")

        if not os.path.exists(f"{scratch_directory}{zip_file}"):
            subprocess.run(kaggle_command, shell=True, check=True)

        unzip()

    return pd.read_csv(annotation_location)


def create_dataset(cfg, image_transform=None, augmentation_transform=None):
    print("Creating dataset...")

    df = load_dataset()

    df = df.groupby('frame').agg(list).reset_index()

    # Splitting the dataset into 90% for training + validation (70% + 20%) and 10% for testing
    train_val, df_test = train_test_split(df, test_size=0.1, random_state=42)

    df_train, df_val = train_test_split(
        train_val, test_size=0.2, random_state=42)

    df_draw = pd.concat([df_test[:5], df_test[-5:]])

    print('Training dataset shape: ', df_train.shape)
    print('Validation dataset shape: ', df_val.shape)
    print('Testing dataset shape: ', df_test.shape)

    dataset_train = CustomDataset(
        df_train, cfg, image_transform, augmentation_transform)
    dataset_val = CustomDataset(
        df_val, cfg, image_transform)
    dataset_test = CustomDataset(
        df_test, cfg, image_transform)
    dataset_draw = CustomDataset(
        df_draw, cfg, image_transform)

    print("Dataset splitted (70%, 20%, 10%)!")

    return dataset_train, dataset_val, dataset_test, dataset_draw


class CustomDataset(Dataset):
    def __init__(self, annotations, cfg, image_transform=None, augmentation_transform=None):
        self.annotations = annotations
        self.image_transform = image_transform
        self.augmentation_transform = augmentation_transform

        self.S = cfg.S
        self.C = cfg.C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_location = str(self.annotations.iloc[idx, 0])

        boxes = self.__setup_boxes(idx)

        image = self._load_image(img_location)

        if (self.augmentation_transform is not None):
            transformed = self.augmentation_transform(
                image=image, bboxes=boxes)
            image = transformed['image']
            boxes = transformed['bboxes']

        if (self.image_transform is not None):
            image = self.image_transform(image)

        label_matrix = torch.zeros((self.S, self.S, 6), dtype=torch.float)

        for box in boxes:
            x, y, width, height, class_id = box

            i, j = int(self.S * y), int(self.S * x)

            cell = label_matrix[i, j]

            # One object per cell
            if cell[0] == 0:
                cell[0] = 1

                x_cell, y_cell = self.S * x - j, self.S * y - i

                width_cell, height_cell = (
                    width * self.S,
                    height * self.S,
                )

                cell[1:5] = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                cell[5] = class_id - 1

        return image, label_matrix

    def _load_image(self, file_name):
        file_path = f"{images_path}/{file_name}"

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)      # <H;W;C>
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __setup_boxes(self, idx):

        xmin = self.annotations.iloc[idx, 1]
        xmax = self.annotations.iloc[idx, 2]
        ymin = self.annotations.iloc[idx, 3]
        ymax = self.annotations.iloc[idx, 4]

        # Convert to midpoint format and normalize

        x = [
            ((xmin + xmax + 1e-10) / 2 / image_original_width)
            for xmin, xmax in zip(xmin, xmax)
        ]

        y = [
            ((ymin + ymax + 1e-10) / 2 / image_original_height)
            for ymin, ymax in zip(ymin, ymax)
        ]
        width = [
            ((xmax - xmin + 1e-10) / image_original_width)
            for xmin, xmax in zip(xmin, xmax)
        ]
        height = [
            ((ymax - ymin + 1e-10) / image_original_height)
            for ymin, ymax in zip(ymin, ymax)
        ]

        class_ids = self.annotations.iloc[idx, 5]

        return list(zip(x, y, width, height, class_ids))
