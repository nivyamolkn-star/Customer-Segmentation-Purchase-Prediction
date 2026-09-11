# Customer Segmentation & Purchase Prediction

##  Project Overview

This project combines **Unsupervised Learning** and **Supervised Learning** to analyze customer data.

Two Machine Learning techniques are used:

* **K-Means Clustering** – Customer Segmentation
* **Logistic Regression** – Purchase Prediction

The dataset used in this project is **manually created** for educational and Machine Learning practice purposes.

## Objectives

* Segment customers into different groups using K-Means
* Analyze customers based on age, income, and spending score
* Visualize customer segments
* Predict whether a customer will make a purchase
* Evaluate the Logistic Regression model using accuracy

##  Dataset

The dataset is **created manually within the Python program** using a dictionary and converted into a Pandas DataFrame.

```python
data = {
    "Age": [...],
    "Annual_Income": [...],
    "Spending_Score": [...],
    "Purchased": [...]
}
```

### Features

| Feature        | Description                   |
| -------------- | ----------------------------- |
| Age            | Age of the customer           |
| Annual_Income  | Annual income of the customer |
| Spending_Score | Customer spending score       |
| Purchased      | Purchase outcome (0 or 1)     |

> **Note:** This dataset is manually created and does not contain real customer information. It is used only for learning and demonstration purposes.

## Machine Learning Methods

### 1. K-Means Clustering

K-Means clustering is used to divide customers into **3 groups** based on:

* Age
* Annual Income
* Spending Score

```python
Kmeans = KMeans(n_clusters=3, random_state=42)
clusters = Kmeans.fit_predict(X_cluster)
```

The resulting customer segments are displayed using a scatter plot.

### 2. Logistic Regression

Logistic Regression is used to predict whether a customer will make a purchase.

Input features:

```text
Age
Annual Income
Spending Score
```

Target:

```text
Purchased
```

##  Project Workflow

```text
Manually Created Dataset
          ↓
    DataFrame Creation
          ↓
     K-Means Clustering
          ↓
   Customer Segmentation
          ↓
   Cluster Visualization
          ↓
      Train-Test Split
          ↓
   Logistic Regression
          ↓
    Purchase Prediction
          ↓
    Accuracy Evaluation
```

##  Visualization

The project generates a scatter plot for customer segmentation.

* **X-axis:** Annual Income
* **Y-axis:** Spending Score
* **Color:** Customer Cluster

##  Model Evaluation

The Logistic Regression model is evaluated using **accuracy score**.

The output displays:

```text
LOGISTIC REGRESSION ACCURACY: ...
```

The exact accuracy is calculated when the program is executed.

##  Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

##  Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/nivyamolkn-star/Customer-Segmentation-Purchase-Prediction.git
```

Navigate to the project directory:

```bash
cd Customer-Segmentation-Purchase-Prediction
```

Run the Python file:

```bash
python customer_segmentation.py
```



##  Learning Outcomes

Through this project, I learned:

* K-Means clustering
* Customer segmentation
* Data visualization using Matplotlib
* Logistic Regression
* Train-test splitting
* Classification
* Accuracy evaluation
* Combining supervised and unsupervised learning techniques

## Disclaimer

This project is created **for educational purposes only**.

The dataset is manually generated and does not represent real customer data.

##  Author

**Nivyamol K.N.**

Machine Learning & Data Science Learner
