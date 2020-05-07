import cv2
import numpy as np
from pickle import (dump, load)
from os import path
from typing import Tuple

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import (LinearSVC, SVC)

import dataset_operations


def tuning_parameters() -> Tuple[list, list]:
    clfs = [['Linear SVM', LinearSVC()], 
            ['SVM', SVC()], 
            ['Decision Tree', DecisionTreeClassifier()]]

    params_tree = {'criterion':['gini', 'entropy'],
                   'max_depth':[10, 20, 30, 40, 50, 60, 90, 150],
                   'splitter':['best', 'random'],
                   'max_features':[5, 10, 20, 30],
                   'random_state':[42, 50, 100]}

    params_svc_lin = {'C':[0.01, 0.1, 0.5],
                      'penalty':['l1', 'l2'],
                      'multi_class':['ovr', 'crammer_singer'],
                      'random_state':[42, 50]}

    params_svc = {'C':[0.001, 0.1, 1, 5, 10, 20, 50, 100, 1000],
                  'gamma':[0.001, 0.01, 0.1, 10, 100, 1000, 'scale'],
                  'degree':[0.001, 0.1, 1, 3, 5, 10, 30],
                  'decision_function_shape':['ovo', 'ovr'], 
                  'kernel':['linear', 'rbf'],
                  'random_state':[42, 50, 100]}
    
    return (clfs, [params_svc_lin, params_svc, params_tree])

def grid_search(train_data, train_label, test_data, test_label, vocab_model):
    clfs_path = './model/classifier'
    classifiers, clfs_params = tuning_parameters()
    descriptor = cv2.AKAZE_create()

    train_data = apply_feature_transform(train_data, descriptor, vocab_model)
    test_data = apply_feature_transform(test_data, descriptor, vocab_model)

    for idx, (name, clf) in enumerate(classifiers):
        grid_clf = GridSearchCV(estimator = clf, cv = 5, param_grid = clfs_params[idx], n_jobs = -1)
        grid_clf.fit(train_data, train_label)
        accuracy_test = accuracy_score(grid_clf.predict(test_data), test_label)
        accuracy_train = accuracy_score(grid_clf.predict(train_data), train_label)

        print(f'{10 * "-"} {name} {10 * "-"}\n'
              f'test accuracy: {accuracy_test}\n'
              f'train accuracy: {accuracy_train}\n')

        if accuracy_test > 0.85:
            dump(grid_clf, open(f'{clfs_path}/clf_{name.lower().replace(" ", "_")}_acc_{accuracy_test * 100 :.2f}.p', 'wb'))

def apply_feature_transform(data: np.ndarray, descriptor, vocab_model) -> np.ndarray:
    data_transformed = []
    for image in data:
        _, image_descriptor = descriptor.detectAndCompute(image, None)
        bow_features_histogram = descriptor2histogram(image_descriptor, vocab_model)
        data_transformed.append(bow_features_histogram)

    return np.asarray(data_transformed)

def descriptor2histogram(descriptor, vocab_model) -> np.ndarray:
    features_words = vocab_model.predict(descriptor)
    histogram = np.zeros(vocab_model.n_clusters, dtype = np.float32)
    unique, counts = np.unique(features_words, return_counts = True)
    histogram[unique] += counts
    histogram /= histogram.sum()
    return histogram

def main():
    train_data, train_labels, _ = dataset_operations.load_dataset('./dataset')
    test_data, test_labels, _ = dataset_operations.load_dataset('./test_data')

    train_data = dataset_operations.process_data(train_data, 700, proportional_height= True)
    test_data = dataset_operations.process_data(test_data, 700, proportional_height= True)

    vocab_model = load(open('./model/vocabulary/vocab_model_440.p', 'rb'))

    grid_search(train_data, train_labels, test_data, test_labels, vocab_model)

if __name__ == "__main__":
    main()
