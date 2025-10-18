# ML Image Classification Practice Project


## Description
This ML Image Clasification uses a SLP Model that is trained by using a MNIST Kaggel DataSet wich has about 70,000 images of written numbers. The task the SLP Model solves is to recive an image and clasificate it by its correct number. It was trained by using a simple supervised learning.

By developing this ML project i learned how to use external packages such as numpy, sklearn and matplotlib. It was a complete challenge to discover how to apply them by being new at the leanguage, but its awesome to see the way different packages conect to each other creating a world of possibilities. I loved working with even this simple level of a ML Model, and im exited to continue learning and working with them.

## Installation
### Setup
__Prerequisites:__ Ensure you have Python installed.
1. Create a Virtual Environment

```
   python -m venv venv
```
2. __Install Dependences:__ (from ```requirements.txs```)
```
   pip install -r requirements.txt
```

### Run the Project
1. Execute the __```main.py```__ script.
```
    python main.py
```

## Using Modules

1. __Load the Data:__
*```load_data``` function retrieves the specified dataset (default is MNIST).*

Example:
```
from lecture import load_data

# Load the MNIST data
mnist_data = load_data() 
# mnist_data is the object containing .data (images) and .target (labels)
```

2. __Train and Predict with Perceptron:__
*```perceptron_start``` function handles the entire machine learning pipeline: data splitting, model training, prediction, and evaluation.*

Example:
```
from perceptron_ml import perceptron_start

# Assume 'mnist_data' was loaded previously
X_test, y_test, y_pred = perceptron_start(mnist_data)

# X_test: Test images
# y_test: True labels for test images
# y_pred: Model's predictions
```




