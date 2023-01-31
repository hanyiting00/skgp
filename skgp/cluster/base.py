from collections import Counter
import math


def count_features(corpus, tokenizer=list):
    """词频特征

        :param corpus: list of str
        :param tokenizer: function for tokenize, default is `jiagu.cut`
        :return:
            features: list of list of float
            names: list of str
        """
    tokens = [tokenizer(x) for x in corpus]
    vocab = [x[0] for x in Counter([x for s in tokens for x in s]).most_common()]
    features = []
    for sent in tokens:
        counter = Counter(sent)
        feature = [counter.get(x, 0) for x in vocab]
        features.append(feature)

    return features, vocab


def tfidf_features(corpus, tokenizer=list):
    """ 文本的 tfidf 特征
    :param corpus: list of str
    :param tokenizer: function for tokenize
    :return:
        features: list of list of float
        names: list of str
    """
    tokens = [tokenizer(x) for x in corpus]

    vocab = [x[0] for x in Counter([x for s in tokens for x in s]).most_common()]
    idf_dict = dict()
    total_doc = len(corpus)
    for word in vocab:
        num = sum([1 if (word in s) else 0 for s in corpus])
        if num == total_doc:
            idf = math.log(total_doc / num)
        else:
            idf = math.log(total_doc / (num + 1))
        idf_dict[word] = idf

    features = []
    for sent in tokens:
        counter = Counter(sent)
        feature = [counter.get(x, 0) / len(sent) * idf_dict.get(x, 0) for x in vocab]
        features.append(feature)

    return features, vocab


if __name__ == '__main__':
    corpus = ["判断unicode是否是汉字。", "全角符号转半角符号。", "一些基于自然语言处理的预处理过程也会在本文中出现。"]
    X, names = count_features(corpus, tokenizer=list)
    print(X)
    print(names)

    X, names = tfidf_features(corpus, tokenizer=list)
    print(X)
    print(names)
