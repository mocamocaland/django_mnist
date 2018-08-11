from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

mnist = datasets.fetch_mldata('MNIST original', data_home='image/')
X = mnist.data / 255
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=1000, test_size=300
)

clf = svm.SVC()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)