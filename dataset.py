import os
from typing import Callable
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa
import torchvision.transforms as TF




class NumpyToTensor(Callable):
    def __call__(self, x):
        x = np.transpose(x, axes=(2, 0, 1))       # HWC -> CHW
        x = torch.from_numpy(x) / 255.0         # <0;255>UINT8 -> <0;1>
        return x.float()

class SelfDrivingCarDataset(Dataset):
    def __init__(self, annotations, image_transform=None):
        self.annotations = annotations
        self.image_transform = image_transform

    def __len__(self):
        return len(self.annotations)

    def __len__(self):
        return len(self.annotations)


    # This method retrieves a sample from the dataset given its index idx.
    # It loads the  image corresponding to the index, applies any specified
    # transformations, and returns the image and its associated bounding boxes
    def __getitem__(self, idx):
        img_location = str(self.annotations.iloc[idx, 0])
        print(img_location)

        # Load image
        image = self._load_image(img_location)
        if (self.image_transform is not None):
            image = self.image_transform(image)

        x0 = self.annotations.iloc[idx, 2]
        y0 = self.annotations.iloc[idx, 3]
        x1 = self.annotations.iloc[idx, 4]
        y1 = self.annotations.iloc[idx, 5]

        bbox = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)

        bbox = self._adjust_bbox(bbox, 480, 300)
        if pd.notnull(cell_value):
            bbox2 = self._adjust_bbox(bbox2, 480, 300)

        return image, bbox, bbox2

    def _load_image(self, file_path):
        image = cv2.imread(file_path,
                           cv2.IMREAD_COLOR)      # <H;W;C>
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def create_dataset(input_transform):
    annotation_dir = ".scratch/data/labels_trainval.csv"
    root_dir = ".scratch/data/"
    print("Creating dataset...")

    data = pd.read_csv(annotation_dir)

    # Count the occurrences of each value
    value_counts = data['frame'].value_counts()
    print(value_counts)

    # Keep only unique frames
    result_df = data[data['frame'].isin(value_counts[value_counts == 1].index)]
    print("Dataset created")

    # Combine root_dir col to df - place filepath to 0th index so __getitem__ found the dir of the image
    result_df.insert(0, 'filepath', root_dir + "images/" + result_df['frame'])

    new_columns = [ 'x1_2', 'y1_2', 'x2_2', 'y2_2', 'class_id_2']

    # Adding new columns
    for column in new_columns:
        result_df[column] = None

    # Splitting the dataset into 90% for training + validation (70% + 20%) and 10% for testing
    train_val, df_test = train_test_split(
        result_df, test_size=0.1, random_state=42)

    df_train, df_val = train_test_split(
        train_val, test_size=0.2, random_state=42)

    df_draw = pd.concat([df_test[:5], df_test[-5:]])

    print('Training dataset shape: ', df_train.shape)
    print('Validation dataset shape: ', df_val.shape)
    print('Testing dataset shape: ', df_test.shape)

    dataset_train = SelfDrivingCarDataset(df_train,input_transform)
    dataset_val = SelfDrivingCarDataset(df_val,input_transform)
    dataset_test = SelfDrivingCarDataset(df_test,input_transform)
    dataset_draw = SelfDrivingCarDataset(df_draw,input_transform)

    print("Dataset splitted (70%, 20%, 10%)!")
    print(result_df.columns)

    return dataset_train, dataset_val, dataset_test, dataset_draw

input_transform = TF.Compose([
            NumpyToTensor()
        ])
create_dataset(input_transform)


