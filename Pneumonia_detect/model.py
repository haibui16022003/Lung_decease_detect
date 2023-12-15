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


class PneumoniaResnet18Model(pl.LightningModule):
    def __init__(self, weight=1):
        super.__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # Change out_features of Resnet18 last fully connected layer from 512 -> 1
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, data):
        pred = self.model(data)
        return pred

    def training_step(self, batch, batch_index):
        xray, label = batch
        label = label.float()       # Convert label to float to compute Loss
        pred = self(xray)[:, 0]
        loss = self.loss_fn(pred, label)    # Compute Loss

        # Log Loss and batch accuracy
        self.log("Train Loss", loss)
        self.log("Step Train Accuracy", self.train_acc(torch.sigmoid(pred), label.int()))

        return loss

    def training_epoch_end(self):
        self.log("Train Accuracy", self.train_acc.compute())

    def validation_step(self, batch, batch_index):
        xray, label = batch
        label = label.float()  # Convert label to float to compute Loss
        pred = self(xray)[:, 0]
        loss = self.loss_fn(pred, label)  # Compute Loss

        # Log Loss and batch accuracy
        self.log("Val Loss", loss)
        self.log("Step Val Accuracy", self.val_acc(torch.sigmoid(pred), label.int()))

        return loss

    def validation_epoch_end(self):
        self.loh("Val Accuracy", self.val_acc.compute())

    def configure_optimizers(self):
        return [self.optimizer]
