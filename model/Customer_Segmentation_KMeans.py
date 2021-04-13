#Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#Using the Elbow Method to find the optimal numbeer of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11) :
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state= 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")


#Difference WCSS
length = len(wcss)
diff_wcss = []
for i in range(0, length) :
  if i == length-1 :
      break
  else :
      difference = wcss[i] - wcss[i+1]
      diff_wcss.append(difference)
avg = sum(diff_wcss)/len(diff_wcss)

#Training the K - Means model on the dataset
for i in range(0, len(diff_wcss)) :
    if diff_wcss[i] < avg :
        cluster = i+1
        break
kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(x)

#Scaling Feature
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(x)

#Saving model as a pickle
import pickle
pickle.dump(y_kmeans, open("kmeans_model.sav", 'wb'))
pickle.dump(sc, open("scaled_model.sav", 'wb'))

#Visualising the Clusters
number_of_colors = cluster
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
for i in range(0, len(color)) :
  plt.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1], s = 100, c = color[i], label = 'Cluster')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1 - 100)')
plt.legend()