import sys

sys.path.insert(1, "./detr/")
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

import argparse
import yaml

import cv2
import numpy as np
import pandas as pd
import time
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models_.detr as model_detr
from logger import logging
from project import Project
from data.dataset import WheatDataset
from data.datatransform.transform import get_train_transforms, get_valid_transforms
from utils.handlers import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


def main(config):

    project = Project()

    if (project.inputs_dir / "df_folds.csv").is_file():
        pass
    else:
        print("splitting dataset..")
        data.split_folds(project.inputs_dir)

    fold = config["val_fold"]
    logging.info(f"val fold = {fold}")

    df_folds = pd.read_csv(project.inputs_dir / "df_folds.csv")
    marking = pd.read_csv(project.inputs_dir / "marking.csv")

    df_train = df_folds[df_folds["fold"] != fold]
    df_valid = df_folds[df_folds["fold"] == fold]

    train_dataset = WheatDataset(
        image_ids=df_train["image_id"].values,
        dataframe=marking,
        path=project.inputs_dir / "train",
        transforms=get_train_transforms(),
    )

    valid_dataset = WheatDataset(
        image_ids=df_valid["image_id"].values,
        dataframe=marking,
        path=project.inputs_dir / "train",
        transforms=get_valid_transforms(),
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = model_detr.DETRModel(
        num_classes=config["num_classes"], num_queries=config["num_queries"]
    )

    model = model.to(device)
    matcher = HungarianMatcher()
    weight_dict = weight_dict = {"loss_ce": 1, "loss_bbox": 1, "loss_giou": 1}
    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(
        config["num_classes"] - 1,
        matcher,
        weight_dict,
        eos_coef=config["null_class_coef"],
        losses=losses,
    )
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    best_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        train_loss = train_fn(
            train_data_loader, model, criterion, optimizer, config=config, epoch=epoch
        )
        valid_loss = eval_fn(valid_data_loader, model, criterion)

        print(
            f"|EPOCH {epoch+1}| TRAIN_LOSS {train_loss.avg}| VALID_LOSS {valid_loss.avg}|"
        )

        logging.info(
            f"|EPOCH {epoch+1}| TRAIN_LOSS {train_loss.avg}| VALID_LOSS {valid_loss.avg}|"
        )

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print(f"New best model in epoch {epoch+1}")
            torch.save(model.state_dict(), project.checkpoint_dir / f"detr_best_{fold}.pth")


def train_fn(data_loader, model, criterion, optimizer, epoch, config):

    model.train()
    criterion.train()
    loss_handler = AverageMeter()

    pbar = tqdm(total=len(data_loader) * config["batch_size"])
    pbar.set_description(
        "Epoch {}, lr: {:.2e}".format(epoch + 1, get_learning_rate(optimizer))
    )

    for i, (images, targets, image_ids) in enumerate(data_loader):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        losses.backward()

        batch_size = len(images)

        if (i + 1) % config["step"] == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_handler.update(losses.item())

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_lr = get_learning_rate(optimizer)

        pbar.update(batch_size)
        pbar.set_postfix(loss="{:.5f}".format(loss_handler.avg))

    pbar.close()
    return loss_handler


def eval_fn(data_loader, model, criterion):
    model.eval()
    criterion.eval()
    loss_handler = AverageMeter()

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            loss_handler.update(losses.item())
            tk0.set_postfix(loss=loss_handler.avg)

    return loss_handler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


if __name__ == "__main__":
    """
    Example of usage:
    >>> python train.py --config config/detr.yaml

    """

    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--config", required=True, help="config file")
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)
