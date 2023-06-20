import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    # Load dataset
    input_file = '0-Datasets/transfusion-Clear.data'
    names = ['R', 'F', 'M', 'T', 'C']
    target_names = ['Não', 'Sim']
    df = pd.read_csv(input_file, names=names)
    df = df.rename({'C': 'target'}, axis=1)

    # Separate X and y data
    X = df.drop('target', axis=1)
    y = df.target
    print("Total samples: {}".format(X.shape[0]))

    # Split the data - 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Balanceamento de Classe
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM classifier using polynomial kernel
    svm = SVC(kernel='rbf') # poly, rbf, linear

    # Perform cross-validation
    cv_results = cross_validate(svm, X_train, y_train, cv=10)
    print("Cross Validation SVM: {:.2f}%".format(np.mean(cv_results['test_score']) * 100))

    # Fit the model on the whole training data
    svm.fit(X_train, y_train)

    # Predict using test dataset
    y_hat_test = svm.predict(X_test)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy SVM from scikit-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from scikit-learn: {:.2f}%".format(f1))

    # Get test confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - SVM scikit-learn")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - SVM scikit-learn normalized")
    plt.show()


if __name__ == "__main__":
    main()