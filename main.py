#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
mnist = fetch_openml('mnist_784',  as_frame=False)

plt.figure(figsize=(20, 4))
for index, img in zip(range(1, 9), mnist.data[:8]):
    plt.subplot(1, 8, index)
    plt.imshow(np.reshape(img, (28, 28)), cmap=plt.cm.gray)
    plt.title(f'Image No.{index} . ')
plt.show()
print(mnist.target[:8])

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.25, random_state=42)

perceptron_model = Perceptron(max_iter=5000, random_state=43) #Adjustable parameter max_iter
perceptron_model.fit(X_train, y_train)

y_pred = perceptron_model.predict(X_test)
accuracy = f1_score(y_test, y_pred, average='micro')
print(f"Accuracy: {accuracy:.2f}")

index = 0
index_errors = []

for label, predict in zip(y_test, y_pred):
    if label != predict:
        index_errors.append(index)
    index+=1

plt.figure(figsize=(18, 4))

for i, img_i in zip(range(1, 6), index_errors[8:14]):
    plt.subplot(1, 6, i)
    plt.imshow(np.reshape(X_test[img_i],(28, 28)), cmap=plt.cm.gray)
    plt.title("Origin: " + str(y_test[img_i]) + " Pred: " + str(y_pred[img_i]))
plt.show()