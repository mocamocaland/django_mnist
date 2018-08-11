import pickle
import numpy as np
from PIL import Image
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def read():
    with open('mnist.pickle', 'rb')as file:
        clf = pickle.load(file)
    return clf

def create_and_save():
    mnist = datasets.fetch_mldata('MNIST original', data_home='image/')
    X = mnist.data / 255
    y = mnist.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=50000, test_size=0
    )

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    with open('mnist', 'wb') as file:
        pickle.dump(clf, file)
    return clf

try:
    clf = read()
except FileNotFoundError:
    clf = create_and_save()

def predict(img_array):
    result = clf.predict(img_array)
    return str(int(result[0]))


if __name__ == '__main__':
    for i in range(0, 10):
        file_name = '{}.png'.format(i)
        img = Image.open(file_name)
        img = np.asarray(img)/255
        img_array = img.reshape(1, 784)
        result = predict(img_array)
        print(result)

score = clf.score(X_test, y_test)
print(score)