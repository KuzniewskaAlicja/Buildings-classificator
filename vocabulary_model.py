import cv2
import numpy as np
from sklearn.cluster import KMeans
from pickle import dump

import dataset_operations

def visual_features_vocabulary(train_data) -> np.ndarray:
    descriptor = cv2.AKAZE_create()
    descriptor_list = []
    for img in train_data:
        _, des = descriptor.detectAndCompute(img, None)
        descriptor_list.extend(des)

    return np.asarray(descriptor_list)

def clustering_visual_features(descriptors, nb_words):
    model_path = './model/vocabulary'
    model = KMeans(n_clusters = nb_words, random_state= 42, n_jobs= - 1)
    model.fit(descriptors)

    dump(model, open(f'{model_path}/vocab_model_{nb_words}.p', 'wb'))

def main():
    train_data, _, _ = dataset_operations.load_dataset('./dataset')
    test_data, _, _ = dataset_operations.load_dataset('./test_data')

    train_data = dataset_operations.process_data(train_data, 700, proportional_height= True)
    test_data = dataset_operations.process_data(test_data, 700, proportional_height= True)

    nb_words = 440
    descriptors = visual_features_vocabulary(train_data)
    clustering_visual_features(descriptors, nb_words)

if __name__ == "__main__":
    main()


