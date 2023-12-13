import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np


def load_file(path):
    return np.load(path).astype(np.float32)


# Compose data
def compose_data(mean, std):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert numpy array to tensor
        transforms.Normalize(mean, std),  # Use mean and std from preprocessing notebook
        transforms.RandomAffine(  # Data Augmentation
            degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
        transforms.RandomResizedCrop((224, 224), scale=(0.35, 1), antialias=True)

    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert numpy array to tensor
        transforms.Normalize([mean], [std]),  # Use mean and std from preprocessing notebook
    ])

    return train_transforms, val_transforms


def create_dataset(mean, std):
    train_transforms, val_transforms = compose_data(mean, std)
    train_dataset = torchvision.datasets.DatasetFolder(
        "Processed/train/",
        loader=load_file, extensions="npy", transform=train_transforms)

    val_dataset = torchvision.datasets.DatasetFolder(
        "Processed/val/",
        loader=load_file, extensions="npy", transform=val_transforms)

    return train_dataset, val_dataset


def create_loader(train_dataset, val_dataset):
    batch_size = 64
    num_workers = 4

    np.unique(train_dataset.targets, return_counts=True), np.unique(val_dataset.targets, return_counts=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader

