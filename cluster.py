import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Live.csv")   

#model = KMeans(n_clusters=3) first we randomly select cluster
"""
number however we can see 3 is not the best cluster number when
we look plot created by matplotlib. So we learn the best cluster 
number as 4. Therefor changed 3 with 4. 
"""

model = KMeans(n_clusters = 4)
model.fit(df)
labels = model.predict(df) 

print(np.unique(labels, return_counts=True))

silhouettes = []
ks = list(range(2, 12))

for n_cluster in ks:
    kmeans = KMeans(n_clusters = n_cluster).fit(df)
    label = kmeans.labels_
    sil_coeff = silhouette_score(df, label, metric = "euclidean")
    print("For n_cluesters = {}, The Silhouette Coefficient = {}"
          .format(n_cluster, sil_coeff))
    silhouettes.append(sil_coeff)
    

plt.figure(figsize = (12,8)) 
plt.subplot(211)
plt.scatter(ks, silhouettes, marker = "x", c = "r")
plt.plot(ks, silhouettes)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()

df["labels"] = labels
print(df.labels.value_counts())

print("Group 0 comments mean:", df[df["labels"] == 0]["num_comments"].mean())
print("Group 1 comments mean:", df[df["labels"] == 1]["num_comments"].mean())
print("Group 2 comments mean:", df[df["labels"] == 2]["num_comments"].mean())
print("Group 3 comments mean:", df[df["labels"] == 3]["num_comments"].mean())

print("Group 0 shares mean:", df[df["labels"] == 0]["num_shares"].mean())
print("Group 1 shares mean:", df[df["labels"] == 1]["num_shares"].mean())
print("Group 2 shares mean:", df[df["labels"] == 2]["num_shares"].mean())
print("Group 3 shares mean:", df[df["labels"] == 3]["num_shares"].mean())

status_type = df[["status_type_photo", "status_type_video", "status_type_status"]].idxmax(axis = 1)

df = pd.concat([df["labels"], status_type.rename("status_type")], axis=1)
print(df.groupby(["labels", "status_type"])["status_type"].count())

