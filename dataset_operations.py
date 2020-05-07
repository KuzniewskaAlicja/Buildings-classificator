from typing import Tuple
import cv2 
from os import (path, listdir)
import numpy as np

def load_dataset(dataset_path) -> Tuple[np.ndarray, np.ndarray, list]:
    train_data, train_labels = [], []
    
    class_names = [class_name for class_name in listdir(dataset_path) if path.isdir(path.join(dataset_path, class_name))]
    for idx, class_name in enumerate(class_names):
        for img_name in listdir(path.join(dataset_path, class_name)):
            train_data.append(cv2.imread(path.join(dataset_path, class_name, img_name), cv2.IMREAD_GRAYSCALE))
            train_labels.append(idx)
    
    return (np.asarray(train_data), np.asarray(train_labels), class_names)

def resize_data(data: np.ndarray, image_width, image_height = None, proportional_height = False) -> np.ndarray:
    for idx, img in enumerate(data):
        if proportional_height:
            image_size = (image_width, int((image_width / img.shape[1]) * img.shape[0]))
        else:
            image_size = (image_width, image_height)

        data[idx] = cv2.resize(img, image_size, fx = 0, fy = 0, interpolation= cv2.INTER_AREA)
    
    return data

def descriptor2histogram(descriptor, vocab_model) -> np.ndarray:
    features_words = vocab_model.predict(descriptor)
    histogram = np.zeros(vocab_model.n_clusters, dtype = np.float32)
    unique, counts = np.unique(features_words, return_counts = True)
    histogram[unique] += counts
    histogram /= histogram.sum()
    return histogram

def apply_feature_transform(data: np.ndarray, descriptor, vocab_model) -> np.ndarray:
    data_transformed = []
    for image in data:
        _, image_descriptor = descriptor.detectAndCompute(image, None)
        bow_features_histogram = descriptor2histogram(image_descriptor, vocab_model)
        data_transformed.append(bow_features_histogram)

    return np.asarray(data_transformed)
