#ML Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score

def perceptron_start(mnist_data, test_size=0.25, max_iter=5000):
    """
    Divides data, trains Perceptron Model, makes predictions,
    calculates accuracy & returns testing and predictions data.
    """
    X_train, X_test, y_train, y_test = train_test_split(mnist_data.data, mnist_data.target, test_size=test_size, random_state=42)

    perceptron_model = Perceptron(max_iter=max_iter, random_state=43)
    perceptron_model.fit(X_train, y_train)

    y_pred = perceptron_model.predict(X_test)

    accuracy = f1_score(y_test, y_pred, average='micro')
    print(f"------ Perceptron Model Evaluation ------")
    print(f"F1-micro Accuracy: {accuracy:.2f}")
    print("------------------------------------------")

    return X_test, y_test, y_pred