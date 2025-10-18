import numpy as np
import matplotlib.pyplot as plt
#Local Imports
from lecture import load_data
from perceptron_ml import perceptron_start

def main():
    """
    Main Function wich carry the data process & model training.
    """
    print("Loading MNIST Perceptron Project...")

    #Loading Data
    mnist_data = load_data()
    #Training Model
    X_test, y_test, y_pred = perceptron_start(mnist_data)

    #Finding Errors
    index = 0
    index_errors = []
    for label, predict in zip(y_test, y_pred):
        if label != predict:
            index_errors.append(index)
        index += 1

    #Showing Errors
    plt.figure(figsize=(18, 4))
    for i, img_i in zip(range(1, 6), index_errors[8:14]):
        plt.subplot(1, 6, i)
        plt.imshow(np.reshape(X_test[img_i], (28, 28)), cmap=plt.cm.gray)
        plt.title("Origin: " + str(y_test[img_i]) + " Pred: " + str(y_pred[img_i]))
    plt.show()

# Execution Control
if __name__ == "__main__":
    main()