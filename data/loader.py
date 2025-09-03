import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(base_path="UCI HAR Dataset/UCI HAR Dataset"):
    print("Loading data...")
    # Read signals
    def read_signals(folder, dataset="train"):
        signals = []
        signal_files = [
            "body_acc_x_", "body_acc_y_", "body_acc_z_",
            "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
            "total_acc_x_", "total_acc_y_", "total_acc_z_"
        ]
        for signal in signal_files:
            filename = os.path.join(base_path, dataset, "Inertial Signals", signal + dataset + ".txt")
            data = pd.read_csv(filename, sep=r'\s+', header=None).values
            signals.append(data)
        return np.transpose(np.array(signals), (1, 2, 0))

    X_train = read_signals(base_path, "train")
    X_test  = read_signals(base_path, "test")

    # Read labels
    y_train = pd.read_csv(os.path.join(base_path, "train", "y_train.txt"), sep=r'\s+', header=None).values.ravel()
    y_test  = pd.read_csv(os.path.join(base_path, "test", "y_test.txt"), sep=r'\s+', header=None).values.ravel()
    
    # Convert labels from 1-indexed to 0-indexed (1-6 -> 0-5)
    y_train = y_train - 1
    y_test = y_test - 1

    return (X_train, y_train), (X_test, y_test)