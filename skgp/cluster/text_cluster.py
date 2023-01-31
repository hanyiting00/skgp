from numpy import unique
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification

from skgp.cluster.base import tfidf_features, count_features

"""
features_method: 提取文本特征的方法，目前支持 tfidf 和 count 两种。
method: 指定使用的方法，默认为 k-means，可选 'k-means', 'dbscan'
n_clusters: k-means 参数，类簇数量
max_iter: k-means 参数，最大迭代次数，确保模型不收敛的情况下可以退出循环
eps: dbscan 参数，邻域距离
min_samples: dbscan 参数，核心对象中的最少样本数量
"""


def text_cluster(docs, features_method='tfidf', method="dbscan", n_clusters=3, max_iter=100, eps=0.5,
                 min_samples=2, tokenizer=list):
    if features_method == 'tfidf':
        features, names = tfidf_features(docs, tokenizer)
    elif features_method == 'count':
        features, names = count_features(docs, tokenizer)
    else:
        raise ValueError('features_method error')

    if method == 'k-means':
        model = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        model.fit(features)
        yhat = model.predict(features)
        clusters = unique(yhat)
        return clusters

    elif method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        # 模型拟合与聚类预测
        yhat = model.fit_predict(features)
        # 检索唯一群集
        clusters = unique(yhat)
        return clusters

    else:
        raise ValueError("method invalid, please use 'k-means' or 'dbscan'")


# docs, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
#                               random_state=4)
# method:dbscan、k-means
# text_cluster(docs, method="k-means")
# text_cluster(docs, features_method="count", method="dbscan", eps=0.4, min_samples=20)

# corpus = ["判断unicode是否是汉字。", "判断unicode是否是汉字2。", "全角符号转半角符号。", "一些基于自然语言处理的预处理过程也会在本文中出现。"]
# corpus = ["判断unicode是否是汉字。"]
# clusters = text_cluster(corpus, features_method="count", method="k-means", n_clusters=3)
# clusters = text_cluster(corpus, features_method="count", method="dbscan", eps=0.4, min_samples=20)
# print(clusters)
