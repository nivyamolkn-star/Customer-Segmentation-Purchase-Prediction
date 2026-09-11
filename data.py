
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

data={
    "Age":[19,21,20,23,31,45,46,47,54,57,59,60,63,67,70,18,22,25,39,42,49,55,61,28,33,36,50,53,56,65],
    "Annual_Income":[15,16,17,18,19,25,26,27,30,32,35,36,40,42,45,14,20,22,28,30,33,37,41,19,21,24,29,34,38,44],
    "Spending_Score":[39, 81, 6, 77, 40, 14, 57, 75, 40, 18, 60, 70, 44,50, 46, 85, 90, 77, 40, 24, 15, 60, 66, 79, 61, 55,18, 72, 58, 38],
    "Purchased":[1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,1,1,0,0,1,1,0]
}

df=pd.DataFrame(data)

print("FIRST 5 ROWS OF DATASET:")
print(df.head())
print("\n")

X_cluster=df[["Age","Annual_Income","Spending_Score"]]

Kmeans=KMeans(n_clusters=3,random_state=42)
clusters=Kmeans.fit_predict(X_cluster)

df["Cluster"]=clusters
print("\nCLUSTER LABELS ASSIGED:")
print(df[["Age","Annual_Income","Spending_Score","Cluster"]].head())

plt.scatter(df["Annual_Income"],df["Spending_Score"],c=df["Cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()

X=df[["Age","Annual_Income","Spending_Score"]]
y=df["Purchased"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)

print("\nLOGISTIC REGRESSION ACCURACY:",accuracy)
