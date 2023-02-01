from skgp.cluster.base import tfidf_features, count_features
from skgp.cluster.dbscan import DBSCAN
from skgp.cluster.kmeans import KMeans

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

    # feature to doc
    f2d = {k: v for k, v in zip(docs, features)}

    if method == 'k-means':
        km = KMeans(k=n_clusters, max_iter=max_iter)
        clusters = km.train(features)
    elif method == 'dbscan':
        ds = DBSCAN(eps=eps, min_pts=min_samples)
        clusters = ds.train(features)
    else:
        raise ValueError("method invalid, please use 'k-means' or 'dbscan'")

    clusters_out = {}
    for label, examples in clusters.items():
        c_docs = []
        for example in examples:
            doc = [d for d, f in f2d.items() if list(example) == f]
            c_docs.extend(doc)

        clusters_out[label] = list(set(c_docs))

    return clusters_out

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
