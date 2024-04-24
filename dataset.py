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


def create_dataset(image_transform=None, augmentation_transform=None):
    print("Creating dataset...")

    df = load_dataset().groupby('frame').agg(list).reset_index()

    # Splitting the dataset into 90% for training + validation (70% + 20%) and 10% for testing
    train_val, df_test = train_test_split(df, test_size=0.1, random_state=42)

    df_train, df_val = train_test_split(
        train_val, test_size=0.2, random_state=42)

    df_draw = pd.concat([df_test[:5], df_test[-5:]])

    print('Training dataset shape: ', df_train.shape)
    print('Validation dataset shape: ', df_val.shape)
    print('Testing dataset shape: ', df_test.shape)

    dataset_train = CustomDataset(
        df_train, image_transform, augmentation_transform)
    dataset_val = CustomDataset(df_val, image_transform)
    dataset_test = CustomDataset(df_test, image_transform)
    dataset_draw = CustomDataset(df_draw, image_transform)

    print("Dataset splitted (70%, 20%, 10%)!")

    return dataset_train, dataset_val, dataset_test, dataset_draw


class CustomDataset(Dataset):
    def __init__(self, annotations, image_transform=None, augmentation_transform=None):
        self.annotations = annotations
        self.image_transform = image_transform
        self.augmentation_transform = augmentation_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_location = str(self.annotations.iloc[idx, 0])

        # Normalized xmin xmax ymin	ymax
        xmin = [x / image_original_width for x in self.annotations.iloc[idx, 1]]
        xmax = [x / image_original_width for x in self.annotations.iloc[idx, 2]]
        ymin = [x / image_original_height for x in self.annotations.iloc[idx, 3]]
        ymax = [x / image_original_height for x in self.annotations.iloc[idx, 4]]

        class_ids = self.annotations.iloc[idx, 5]

        # Format x_min, y_min, x_max, y_max, class_id
        boxes = [[x_min, y_min, x_max, y_max, class_ids] for class_ids, x_min,
                 x_max, y_min, y_max in zip(class_ids, xmin, xmax, ymin, ymax)]

        image = self._load_image(img_location)

        if (self.augmentation_transform is not None):
            transformed = self.augmentation_transform(
                image=image, bboxes=boxes)
            image = transformed['image']
            boxes = transformed['bboxes']

        if (self.image_transform is not None):
            image = self.image_transform(image)

        # Final format class_id, x_min, y_min, x_max, y_max
        boxes = [[box[4], box[0], box[1], box[2], box[3]] for box in boxes]

        boxes = torch.tensor(boxes)

        return image, boxes

    def _load_image(self, file_name):
        file_path = f"{images_path}/{file_name}"

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)      # <H;W;C>
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
