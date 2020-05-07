import cv2
import matplotlib.pyplot as plt
import numpy as np
from pickle import load
from sklearn.metrics import (plot_confusion_matrix, accuracy_score)

import dataset_operations

def main():
    model_dir = './model/vocabulary'
    clf_dir = './model/classifier'
    train_path = './dataset'
    test_path = './test_data'

    train_data, train_labels, class_names = dataset_operations.load_dataset(train_path)
    test_data, test_labels, _ = dataset_operations.load_dataset(test_path)

    train_data = dataset_operations.resize_data(train_data, 700, proportional_height=True)
    test_data = dataset_operations.resize_data(test_data, 700, proportional_height=True)

    clf = load(open(clf_dir + '/clf_svm_acc_89.41.p', 'rb'))
    vocab_model = load(open(model_dir + '/vocab_model_440.p', 'rb'))
    descriptor = cv2.AKAZE_create()

    # Displaying result
    train_data_transform = dataset_operations.apply_feature_transform(train_data, descriptor, vocab_model)
    test_data_transform = dataset_operations.apply_feature_transform(test_data, descriptor, vocab_model)

    predicted_test = clf.predict(test_data_transform)
    predicted_train = clf.predict(train_data_transform)
    print(f'{10 * "-"} {str(clf.best_estimator_)[:3].lower()} accuracy {10 * "-"}\n'
          f'Test Data: {accuracy_score(predicted_test, test_labels) * 100:.2f} %\n'
          f'Train Data: {accuracy_score(predicted_train, train_labels)* 100:.2f} %')
    
    plot_confusion_matrix(clf, test_data_transform, test_labels, 
                          display_labels=class_names, normalize='true',
                          cmap='Oranges')
    plt.gca().set_title('Confusion matrix on the test data')
    plt.show()

if __name__ == "__main__":
    main()

