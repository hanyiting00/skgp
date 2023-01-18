# 非凸数据集
# 谱聚类算法 与 kmeans 算法对比
from sklearn import datasets
from sklearn import cluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.figure(figsize=[6, 6])
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
plt.scatter(noisy_circles[0][:, 0], noisy_circles[0][:, 1], marker='.')
plt.title("non-convex datasets")
plt.show()
# k=2训练数据,k-means聚类算法
random_state = 170
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(noisy_circles[0])
plt.scatter(noisy_circles[0][:, 0], noisy_circles[0][:, 1], marker='.', c=y_pred)
plt.title("k-means clustering")
plt.show()
# spectralClustering谱聚类算法
y_pred = cluster.SpectralClustering(n_clusters=2, affinity="nearest_neighbors").fit_predict(noisy_circles[0])
plt.scatter(noisy_circles[0][:, 0], noisy_circles[0][:, 1], marker='.', c=y_pred)
plt.title("spectralClustering")
plt.show()
