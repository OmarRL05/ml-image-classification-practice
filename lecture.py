import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def load_data(dfname='mnist_784'):
    """
    Charges the MNIST dataset, then shows the first eight images & its labels
    Returns the object charged with data.
    """
    mnist = fetch_openml(dfname, as_frame=False, parser='auto')
    plt.figure(figsize=(20, 4))
    for index, img in zip(range(1, 9), mnist.data[:8]):
        plt.subplot(1, 8, index)
        plt.imshow(np.reshape(img, (28, 28)), cmap=plt.cm.gray)
        plt.title(f'Image No.{index} . ')
    plt.show()
    print(f"Primeras 8 etiquetas de {dfname}: {mnist.target[:8]}")

    return mnist