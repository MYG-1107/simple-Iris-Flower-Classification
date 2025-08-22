# Simple Iris Flower Classification
This project introduces machine learning (ML) by building a model to classify iris flowers into three species (Setosa, Versicolor, Virginica) based on measurements like petal length and width. It uses the classic Iris dataset and Python with scikit-learn, ideal for beginners to understand the ML workflow.
Prerequisites

Python 3.8+ (or use Google Colab: colab.research.google.com)
Libraries: Install via !pip install scikit-learn pandas matplotlib seaborn in Colab or locally.
No prior ML knowledge required.

## Project Steps
1. Import Libraries and Load Data
The Iris dataset contains 150 samples with 4 features (sepal length, sepal width, petal length, petal width) and a label (species).
<!-- import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns -->

## Load dataset
<!--
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target  # 0: Setosa, 1: Versicolor, 2: Virginica
print(data.head())  # View first 5 rows -->

2. Data Exploration and Preprocessing
Visualize data to spot patterns and split into training (80%) and testing (20%) sets.
## Visualize relationships
sns.pairplot(data, hue='species')
plt.show() 

## Features (X) and labels (y)
X = data.drop('species', axis=1)
y = data['species']

## Train-test split
<!--X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) -->

Why? Training data teaches the model; testing data checks if it generalizes.
3. Train the Model
Use a Random Forest classifier (an ensemble of decision trees) to learn patterns.
## Create and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

Flow: The model builds rules (e.g., "If petal length > 2.5, likely not Setosa").
4. Evaluate the Model
Test on unseen data to measure performance.
## Predict on test data
y_pred = model.predict(X_test)

## Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")  # Typically 95%+

Note: High accuracy means the model works well. If low, try more data or a different model.
5. Make Predictions
Use the model to classify a new flower.
### New flower: [sepal length, sepal width, petal length, petal width]
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Likely Setosa
prediction = model.predict(new_flower)
species_names = ['Setosa', 'Versicolor', 'Virginica']
print(f"Predicted species: {species_names[prediction[0]]}")

<i>Try different measurements to see varying predictions.
What You’ll Learn</i>

ML Workflow: Data → Preprocessing → Training → Evaluation → Prediction.
AI Context: This is supervised ML (uses labeled data). AI extends to neural networks, etc.
Pitfalls: Overfitting (model memorizes training data). Always split data to test generalization.
