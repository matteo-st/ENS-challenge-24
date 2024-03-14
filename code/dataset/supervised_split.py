import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import os
import cv2

# Adjust to the correct path relative to the script's location
#project_dir = Path("/mnt/c/Users/matte/iCloudDrive/Documents/Studies/IASD/College-De-France/Challenge")
project_dir = Path("~/ENS-challenge-24")
data_dir = project_dir / "data"  # Assuming 'data' is a sibling directory to 'code', and you are in 'code/dataset'
x_train_dir = data_dir / "x_train"
supervised_dir = data_dir / "supervised"
supervised_x_train_dir = supervised_dir / "x_train"

# Create directories if they do not exist
#supervised_x_train_dir.mkdir(parents=True, exist_ok=True)
if not os.path.exists(supervised_dir):
    os.makedirs(supervised_x_train_dir)
if not os.path.exists(supervised_dir):
    os.makedirs(supervised_x_train_dir)
#Load labels
labels_train = pd.read_csv(data_dir / "y_train.csv", index_col=0).T

# Filter out labeled data (rows not all zero)
supervised_files = labels_train[(labels_train != 0).any(axis=1)]
print(f"Number of labeled images : {supervised_files.shape}")
      
# Copy labeled images to the supervised folder
for file in supervised_files.index:
    image_file = x_train_dir / file
    print(image_file)
    
    if image_file.exists():
        print("bwe")
        shutil.copy(image_file, supervised_x_train_dir)

# Save the labeled portion of y_train.csv
supervised_files.T.to_csv(supervised_dir / "y_train.csv")

print("Supervised dataset creation completed.")
