# Logistic Regression Implementation on Iris Dataset

## Overview

This project involves performing Exploratory Data Analysis (EDA) and implementing a Logistic Regression model on the Iris dataset. The Iris dataset is a classic dataset in machine learning and contains features of iris flowers along with their species.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries Used](#libraries-used)
3. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Binary Classification](#binary-classification)
6. [Logistic Regression Implementation](#logistic-regression-implementation)
7. [Model Training and Evaluation](#model-training-and-evaluation)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [References](#references)

## Introduction

The Iris dataset consists of 150 samples from three species of Iris flowers: setosa, versicolor, and virginica. Each sample contains four features: sepal length, sepal width, petal length, and petal width. The goal of this project is to explore the dataset and build a logistic regression model to classify the species based on these features.

## Libraries Used

- **NumPy**: For numerical computations.
- **Pandas**: For data analysis and manipulation.
- **Seaborn**: For statistical data visualization.
- **Matplotlib**: For creating static, interactive, and animated visualizations.
- **Scikit-learn**: For machine learning utilities.

## Data Loading and Preprocessing

### Steps:

1. **Load the Iris Dataset**:
   - The Iris dataset is loaded using Scikit-learn's `datasets.load_iris()` function.

2. **Create a DataFrame**:
   - Convert the dataset into a Pandas DataFrame for easier analysis and manipulation.

3. **Basic Statistics**:
   - Print basic statistics (count, mean, std, min, max, etc.) for each feature using `DataFrame.describe()`.

4. **Visualizations**:
   - Visualize the data using pair plots to explore relationships between features, categorized by species.
   - Analyze species distribution by printing the distribution of species in the dataset.
   - Calculate the correlation matrix between features and plot it as a heatmap to identify relationships between variables.

### Code:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame
df = pd.DataFrame(data=np.c_[X, y], columns=feature_names + ['species'])
df['species'] = df['species'].map({i: target_names[i] for i in range(len(target_names))})

# Basic Statistics
print(df.describe())

# Visualizations
sns.pairplot(df, hue='species')
plt.show()

# Species Distribution
species_counts = df['species'].value_counts()
print(species_counts)

# Correlation Analysis
numeric_df = df.drop('species', axis=1)
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```

## Exploratory Data Analysis (EDA)

### Insights from EDA:

1. **Petal Length and Width**: Significant separability between species, especially for setosa.
2. **Setosa**: Smaller petal lengths and widths compared to versicolor and virginica.
3. **Versicolor and Virginica**: More overlap in distributions but discernible differences in petal dimensions.

## Binary Classification

### Methodology:

1. **Subset Selection**:
   - For binary classification, only a subset of the dataset is used (Setosa and Versicolor).

2. **Data Assignment**:
   - `X`: Feature data for the selected subset.
   - `y`: Target labels for the selected subset.

### Code:

```python
# Binary Classification
X = iris.data[:100]  # Using only two species for binary classification
y = iris.target[:100]
```

## Logistic Regression Implementation

### Methodology:

1. **Sigmoid Function**:
   - Maps the output of the linear combination of features to the range [0, 1].

2. **Cost Function**:
   - Calculates the logistic regression cost (or loss) using cross-entropy loss.

3. **Gradient Descent**:
   - Optimizes the weights by minimizing the cost function through iterative updates.

### Code:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, weights):
    m = len(y)
    h = sigmoid(X.dot(weights))
    cost = (-y).dot(np.log(h)) - (1 - y).dot(np.log(1 - h))
    return cost / m

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        weights = weights - (learning_rate/m) * X.T.dot(sigmoid(X.dot(weights)) - y)
        cost_history[i] = cost_function(X, y, weights)

    return weights, cost_history
```

## Model Training and Evaluation

### Steps:

1. **Data Preparation**:
   - Add a bias term to the feature matrix.

2. **Initialization**:
   - Set initial weights to zero for all features, including the bias term.

3. **Training**:
   - Train the model using gradient descent.

4. **Prediction and Accuracy**:
   - Predict target labels for the test data and evaluate the model’s performance.

### Code:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add Bias Term
m, n = X_train_scaled.shape
X_train_with_bias = np.append(np.ones((m, 1)), X_train_scaled, axis=1)
weights = np.zeros(n + 1)
iterations = 2000
learning_rate = 0.1

# Train the Model
weights, cost_history = gradient_descent(X_train_with_bias, y_train, weights, learning_rate, iterations)

# Prediction
def predict(X, weights):
    X_with_bias = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    return [1 if i > 0.5 else 0 for i in sigmoid(X_with_bias.dot(weights))]

y_pred = predict(X_test_scaled, weights)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

## Results

- **Accuracy**: The logistic regression model achieved an accuracy of 100% on the test set, indicating excellent performance.

## Conclusion

This project demonstrates the effectiveness of logistic regression for binary classification on the Iris dataset. The combination of thorough EDA and careful preprocessing contributed to the model's high accuracy. Future work could explore multi-class classification and other advanced techniques to further improve model performance.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

Feel free to contribute, raise issues, or suggest improvements!

---

**Author**: Varun Kumar Singh

**Contact**: Varun_2303res73@iitp.ac.in / Varunsingh3913@gmail.com
