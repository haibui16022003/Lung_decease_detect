from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

ROOT_PATH = Path("rsna-pneumonia-detection-challenge/stage_2_train_images/")
PROCESSED_PATH = Path("Processed/")

train_labels = pd.read_csv("rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")
train_labels = train_labels.drop_duplicates()


# Preprocessing training data
def preprocess(labels):
    """Preprocessing training data"""
    sums = 0
    sums_squared = 0

    for c, patient_id in enumerate(labels.patientId):
        dcm_path = ROOT_PATH / patient_id
        dcm_path = dcm_path.with_suffix(".dcm")

        # Read and Standardize dicom array
        dcm = pydicom.read_file(dcm_path).pixel_array / 255

        # Resizing picture from 1024x1024 -> 224x224 -> convert to float16
        dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
        label = labels.Target.iloc[c]

        # Split train and val data
        status = "train" if c < 24000 else "val"

        # Save
        current_save_path = PROCESSED_PATH/status/str(label)
        current_save_path.mkdir(parents=True, exist_ok=True)
        np.save(current_save_path/patient_id, dcm_array)

        # Normalizer sum of image
        normalizer = dcm_array.shape[0] * dcm_array.shape[1]
        if status == "train":
            sums += np.sum(dcm_array) / normalizer
            sums_squared += (np.power(dcm_array, 2).sum()) / normalizer

    mean = sums / 24000
    std = np.sqrt(sums_squared / 24000 - (mean**2))
    return mean, std

