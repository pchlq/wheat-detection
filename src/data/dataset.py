import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class WheatDataset(Dataset):
    def __init__(self, image_ids, dataframe, path, transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
        self.path = path

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]
        image = cv2.imread(f"{self.path}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # DETR takes in data in coco format
        boxes = records[["x", "y", "w", "h"]].values

        # Area of bb
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels = np.zeros(len(boxes), dtype=np.int32)

        sample = {"image": image, "bboxes": boxes, "labels": labels}

        if self.transforms:
            sample = self.transforms(**sample)
            image = sample["image"]
            boxes = sample["bboxes"]
            labels = sample["labels"]

        # Normalizing BBOXES
        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(
            sample["bboxes"], rows=h, cols=w
        )
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.long)
        target["image_id"] = torch.tensor([index])
        target["area"] = area

        return image, target, image_id


def split_folds(df_path):
    """ 
    splitting train set on 5 folds
    """

    seed = 100  # TOCONFIG
    marking = pd.read_csv(df_path / "train.csv")
    bboxs = np.stack(marking["bbox"].apply(lambda x: np.fromstring(x[1:-1], sep=",")))
    for i, column in enumerate(["x", "y", "w", "h"]):
        marking[column] = bboxs[:, i]
    marking.drop(columns=["bbox"], inplace=True)

    # Creating Folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    df_folds = marking[["image_id"]].copy()
    df_folds.loc[:, "bbox_count"] = 1
    df_folds = df_folds.groupby("image_id").count()
    df_folds.loc[:, "source"] = (
        marking[["image_id", "source"]].groupby("image_id").min()["source"]
    )
    df_folds.loc[:, "stratify_group"] = np.char.add(
        df_folds["source"].values.astype(str),
        df_folds["bbox_count"].apply(lambda x: f"_{x // 15}").values.astype(str),
    )
    df_folds.loc[:, "fold"] = 0

    for fold_number, (train_index, val_index) in enumerate(
        skf.split(X=df_folds.index, y=df_folds["stratify_group"])
    ):
        df_folds.loc[df_folds.iloc[val_index].index, "fold"] = fold_number

    print(df_folds.head())

    df_folds.to_csv(df_path / "df_folds.csv")
    marking.to_csv(df_path / "marking.csv", index=False)
