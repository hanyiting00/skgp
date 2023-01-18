# dbscan 聚类-基于密度的空间聚类算法
# 聚类距离簇边界最近的点
# 核心点：核心点的半径范围内的样本个数≥最少点数
# 边界点：边界点的半径范围内的样本个数小于最少点数大于0
# 噪声点：噪声点的半径范围的样本个数为0
# 优点：能够识别任意形状的样本. 该算法将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇，它将簇定义为密度相连的点的最大集合。
# 缺点：需指定最少点个数，半径(或自动计算半径)
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 定义模型
model = DBSCAN(eps=0.40, min_samples=20)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    # 创建这些样本的散布
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()