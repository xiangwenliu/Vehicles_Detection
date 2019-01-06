"""
split 3/4 as training data,1/4 as test data
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from images import read_image, draw_bbox
from data_convert import get_relevant_frames, get_boxes

labels = pd.read_csv("labels_resized.csv")
labels.head()

paths = labels[["Frame", "Mask"]].as_matrix()
paths = paths[:, 0] + "!" + paths[:, 1]
paths = np.unique(paths)

def temp(img_path: str, mask_path: str) -> None:
    """Open IMAGE and MASK and plot"""
    img = read_image(img_path)
    mask = read_image(mask_path, gray=True)

    result = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(result)

image_paths = labels["Frame"].unique()
np.random.shuffle(image_paths)
split_idx = int(image_paths.shape[0] * 0.75)

print("train index 0 ~ {} (size: {})".format(split_idx - 1, split_idx))
print("test index {} ~ {} (size: {})".format(split_idx,
                                             image_paths.shape[0] - 1, image_paths.shape[0] - split_idx))

train_paths = image_paths[:split_idx]
test_paths = image_paths[split_idx:]

print(train_paths.shape, test_paths.shape)

train_csv = labels[labels['Frame'].isin(train_paths)].reset_index(drop=True)
test_csv = labels[labels['Frame'].isin(test_paths)].reset_index(drop=True)

assert train_csv["Frame"].unique().shape[0] == train_paths.shape[0]
assert test_csv["Frame"].unique().shape[0] == test_paths.shape[0]

columns = ["Frame", "Mask"]

train_csv[columns].to_csv("train.csv", index=False, header=False)
test_csv[columns].to_csv("test.csv", index=False, header=False)