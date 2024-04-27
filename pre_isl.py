import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

def process_dataset_isl(dataset_dir):
    image_paths = []
    labels = []

    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.isdir(folder_path):
            if folder_name.isalpha() and len(folder_name) == 1:
                for file_name in tqdm(os.listdir(folder_path), desc=f'Processing {folder_name}'):
                    if file_name.endswith('.jpg'):
                        image_path = os.path.join(folder_path, file_name)
                        image_paths.append(image_path)
                        labels.append(folder_name)

    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = [label_map[label] for label in labels]

    X = []
    y = []
    for image_path, label in zip(image_paths, encoded_labels):
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (400, 400))
            X.append(image)
            y.append(label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    X = np.array(X)
    X = X.astype('float32') / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train, num_classes=len(unique_labels))
    y_test = to_categorical(y_test, num_classes=len(unique_labels))

    return X_train, X_test, y_train, y_test, unique_labels

X_train_isl, X_test_isl, y_train_isl, y_test_isl, unique_labels_isl = process_dataset_isl('D:\A to Z sign detection language\Data\ISL')

np.save('X_train_isl.npy', X_train_isl)
np.save('X_test_isl.npy', X_test_isl)
np.save('y_train_isl.npy', y_train_isl)
np.save('y_test_isl.npy', y_test_isl)

print("ISL Dataset:")
print(f"X_train shape: {X_train_isl.shape}, y_train shape: {y_train_isl.shape}")
print(f"X_test shape: {X_test_isl.shape}, y_test shape: {y_test_isl.shape}")
print(f"Number of unique labels: {len(unique_labels_isl)}")
