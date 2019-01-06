import argparse
import os
import shutil
import time
from collections import namedtuple
from multiprocessing.pool import Pool
import cv2
import numpy as np
import pandas as pd
from images import read_image,read_image_and_resize



parser = argparse.ArgumentParser()

parser.add_argument(
        "--src_csv",
        default="~/tensorflowproject/data/crowdai/labels_crowdai.csv",
        help="/path/to/labels.csv")

parser.add_argument(
    "--data_dir",
    default="~/tensorflowproject/data/crowdai/object-detection-crowdai",
    help="Directory where training datasets are located")

parser.add_argument(
    "--save_dir",
    default="data_resize",
    help="path to the directory in which resize image will be saved")

parser.add_argument(
    "--target_width", default=960, help="new target width (default: 960)")

parser.add_argument(
    "--target_height", default=640, help="target height (default: 640)")

parser.add_argument(
    "--target_csv",
    default="labels_resized.csv",
    help="target csv filename")

FLAGS = parser.parse_args()
Box = namedtuple("Box", ["left_top", "right_bot"])

def get_boxes(df):
    """Given relevant DATAFRAME return a list of BOX"""
    boxes = []
    for _, items in df.iterrows():

        left_top = items["xmin"], items["ymin"]
        right_bot = items["xmax"], items["ymax"]

        boxes.append(Box(left_top, right_bot))
    return boxes


def create_clean_dir(dirname):

    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    assert os.path.exists(dirname) is False

    os.mkdir(dirname)

    assert not os.listdir(dirname)


def adjust_bbox(bboxframe,
                src_size,
                dst_size):
    W, H = src_size
    W_new, H_new = dst_size

    bboxframe = bboxframe.copy()

    bboxframe['xmin'] = (bboxframe['xmin'] * W_new / W).astype(np.int16)
    bboxframe['xmax'] = (bboxframe['xmax'] * W_new / W).astype(np.int16)
    bboxframe['ymin'] = (bboxframe['ymin'] * H_new / H).astype(np.int16)
    bboxframe['ymax'] = (bboxframe['ymax'] * H_new / H).astype(np.int16)

    return bboxframe


def get_relevant_frames(image_path,
                        dataframe):
    cond = dataframe["Frame"] == image_path
    return dataframe[cond].reset_index(drop=True)


def get_mask(image, bbox_frame):
    H, W, _ = image.shape

    mask = np.zeros((H, W))

    for _, row in bbox_frame.iterrows():
        W_beg, W_end = row['xmin'], row['xmax']
        H_beg, H_end = row['ymin'], row['ymax']

        mask[H_beg:H_end, W_beg:W_end] = 255

    return mask


def create_mask(image_WH,
                image_path,
                dataframe):

    W, H = image_WH
    mask = np.zeros((H, W))

    bbox_frame = get_relevant_frames(image_path, dataframe)

    for _, row in bbox_frame.iterrows():
        W_beg, W_end = row['xmin'], row['xmax']
        H_beg, H_end = row['ymin'], row['ymax']

        mask[H_beg:H_end, W_beg:W_end] = 255

    return mask


def generate_mask_pipeline(image_WH,
                           image_path,
                           dataframe,
                           save_dir):

    filename = os.path.basename(image_path)
    full_path = os.path.join(save_dir, filename)
    mask = create_mask(image_WH, image_path, dataframe)

    cv2.imwrite(full_path, mask)


def main(FLAGS):
    new_WH = (FLAGS.target_width, FLAGS.target_height)
    data = pd.read_csv(FLAGS.src_csv)
    # Only consider car and truck images
    data = data[data["Label"].isin(["Car", "Truck"])].reset_index(drop=True)

    # 123.jpg -> object-detection-crowdai/123.jpg
    data["Frame"] = data["Frame"].map(
        lambda x: os.path.join(FLAGS.data_dir, x))

    # IF dir exists, clean it
    create_clean_dir(FLAGS.save_dir)
    print("Cleaned {} directory".format(FLAGS.save_dir))

    print("Resizing begins")
    start = time.time()
    pool = Pool()
    pool.starmap_async(read_image_and_resize,
                       [(image_path, new_WH, FLAGS.save_dir)
                        for image_path in data["Frame"].unique()])

    pool.close()
    pool.join()
    end = time.time()

    print("Time elapsed: {}".format(end - start))
    print("Resizing ends")

    print("Adjusting dataframe")

    # Read any image file to get the WIDTH and HEIGHT
    image_path = data["Frame"][0]
    image = read_image(image_path)

    H, W, _ = image.shape
    src_size = (W, H)

    labels = adjust_bbox(data, src_size, new_WH)

    # object-.../123.jpg -> data_resize/123.jpg
    labels["Frame"] = labels["Frame"].map(
        lambda x: os.path.join(FLAGS.save_dir, os.path.basename(x)))

    create_clean_dir("mask")
    print("Cleaned {} directory".format("mask"))
    print("Masking begin")
    start = time.time()

    pool = Pool()
    tasks = [(new_WH, image_path, labels, "mask")
             for image_path in labels["Frame"].unique()]
    pool.starmap_async(generate_mask_pipeline, tasks)
    pool.close()
    pool.join()
    end = time.time()
    print("Masking ends. Time elapsed: {}".format(end - start))

    labels["Mask"] = labels["Frame"].map(
        lambda x: os.path.join("mask", os.path.basename(x)))
    labels.to_csv(FLAGS.target_csv, index=False)

    print("Adjustment saved to {}".format(FLAGS.target_csv))



if __name__ == '__main__':
    main(FLAGS)