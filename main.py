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
