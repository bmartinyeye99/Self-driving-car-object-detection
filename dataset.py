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

def create_dataset():
    dir = ".scratch/data/labels_trainval.csv"
    root_dir = ".scratch/data/"
    print("Creating dataset...")

    data = pd.read_csv(dir)

    # Count the occurrences of each value
    value_counts = data['frame'].value_counts()
    print(value_counts)

    # Keep only unique frames
    result_df = data[data['frame'].isin(value_counts[value_counts == 1].index)]
    print("Dataset created")

    # Combine root_dir col to df
    result_df['filepath'] = root_dir + "images/" + result_df['frame']

    new_columns = ['label2', 'x1_2', 'y1_2', 'x2_2', 'y2_2']

    # Adding new columns
    for column in new_columns:
        result_df[column] = None

    # Splitting the dataset into 90% for training + validation (70% + 20%) and 10% for testing
    train_val, df_test = train_test_split(
        result_df, test_size=0.1, random_state=42)

    df_train, df_val = train_test_split(
        train_val, test_size=0.2, random_state=42)

    df_train = augment_data(df_train)


def blend_images(image_path1, image_path2, alpha=0.7):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Blend the images
    blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended


def augment_data(df):
    print('Augmenting data....')
    root = '.scratch/data/augmentations'

    if not os.path.exists(root):
        os.makedirs(root)

    root = root + '/'

    augmented_df = []

    df2 = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for row, row2 in zip(df.itertuples(index=True, name='Pandas'), df2.itertuples(index=True, name='Pandas')):
        # Load the image
        file_name = row.frame
        filepath = row.filepath
     #   print(filepath)
        file_name2 = row2.frame
        filepath2 = row2.filepath

        # Check if filepath is None
        if filepath is None:
            print(f"Error: Filepath is None for {file_name}")
            continue

        # Read the image
        image = cv2.imread(filepath)

        # Check if image is None
        if image is None:
            print(f"Error: Failed to read image for {file_name}")
            continue


        x1, y1, x2, y2 = row.xmin, row.ymin, row.xmax, row.ymax
        x1_2, y1_2, x2_2, y2_2 = row2.xmin, row2.ymin, row2.xmax, row2.ymax

        # Define the bounding box
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        ], shape=image.shape)


        # Augmenters
        flip_augmenter = iaa.Fliplr(1.0)  # flip
        rotate_augmenter = iaa.Affine(rotate=10)  # Rotate by 10 degrees

        flipped = root + 'flipped_' + file_name

        # Apply flip
        image_flipped, bbs_flipped = flip_augmenter(
            image=image, bounding_boxes=bbs)

        if not os.path.exists(flipped):
            cv2.imwrite(flipped, image_flipped)

        rotated = root + 'rotated_' + file_name

        # Apply rotation
        image_rotated, bbs_rotated = rotate_augmenter(
            image=image, bounding_boxes=bbs)

        if not os.path.exists(rotated):
            cv2.imwrite(rotated, image_rotated)

        rotated_flipped = root + 'rotated_flipped_' + file_name

        # Apply rotation and then flip
        image_rotated_flipped, bbs_rotated_flipped = flip_augmenter(
            image=image_rotated, bounding_boxes=bbs_rotated)

        if not os.path.exists(rotated_flipped):
            cv2.imwrite(rotated_flipped, image_rotated_flipped)

        blend_image = root + 'blended_' + \
            file_name.replace(".jpg", "") + '_' + file_name2

        if not os.path.exists(blend_image):
            cv2.imwrite(blend_image, blend_images(filepath, filepath2))

        bbs_flipped = bbs_flipped.bounding_boxes[0]
        bbs_rotated = bbs_rotated.bounding_boxes[0]
        bbs_rotated_flipped = bbs_rotated_flipped.bounding_boxes[0]

        # blended image
        bl_image = {
            'filepath': blend_image,
            'label': row.class_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'file': row.frame,
            'label2': row2.class_id,
            'x1_2': x1_2,
            'y1_2': y1_2,
            'x2_2': x2_2,
            'y2_2': y2_2,
        }

        # flipped image
        fl_row = {
            'filepath': flipped,
            'label': row.class_id,
            'x1': bbs_flipped.x1,
            'y1': bbs_flipped.y1,
            'x2': bbs_flipped.x2,
            'y2': bbs_flipped.y2,
            'file': row.frame
        }

        rt_row = {
            'filepath': rotated,
            'label': row.class_id,
            'x1': bbs_rotated.x1,
            'y1': bbs_rotated.y1,
            'x2': bbs_rotated.x2,
            'y2': bbs_rotated.y2,
            'file': row.frame
        }

        fl_rt_row = {
            'filepath': rotated_flipped,
            'label': row.class_id,
            'x1': bbs_rotated_flipped.x1,
            'y1': bbs_rotated_flipped.y1,
            'x2': bbs_rotated_flipped.x2,
            'y2': bbs_rotated_flipped.y2,
            'file': row.frame
        }

        augmented_df.append(bl_image)
        augmented_df.append(fl_row)
        augmented_df.append(rt_row)
        augmented_df.append(fl_rt_row)

    aug_df = pd.DataFrame(augmented_df)

    df = pd.concat([df, aug_df], ignore_index=True)
    print('Augmentation finished!')

    print('Augmented shape: ', aug_df.shape)
    #print(df)
    return df

