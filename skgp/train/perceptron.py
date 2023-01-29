import gzip
import pickle
import random
from collections import defaultdict


class AveragedPerceptron(object):
    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)  # 生成一个默认dict,不存在的值返0.0
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight  # 每次预测一个特征值时该值都会重置
        # Do a secondary alphabetic sort, for stability
        return max(self.classes, key=lambda label: (scores[label], label))  # 返回得分最高的词性标签，如果得分相同取字母大的

    def update(self, truth, guess, features):
        """Update the feature weights."""

        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w  # 累加:(此时的i - 上次更新该权值时的i)*权值
            self._tstamps[param] = self.i  # 记录更新此权值时的i
            self.weights[f][c] = w + v  # 更新权值

        self.i += 1  # 一个word对应于一个features,每处理一个word后i值+1
        if truth == guess:
            return None
        for f in features:  # 遍历特征值,对每个特征值都加入当前判断正确和错误的词性,以及各自权值
            weights = self.weights.setdefault(f, {})  # 如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值,并将键值加入字典中,注意和get方法的区别
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)
        return None

    def average_weights(self):
        """Average weights from all iterations."""
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged  # 向字典中加入key-value
            self.weights[feat] = new_feat_weights
        return None


class Perceptron:
    def __init__(self, loc=None):
        self.START = ['-START-', '-START2-']
        self.END = ['-END-', '-END2-']
        self.model = AveragedPerceptron()

        if loc is not None:
            self.load(loc)

    def train(self, sentences, save_loc=None, nr_iter=5, shuf=True):
        """Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param shuf:
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        """
        self._make_tagdict(sentences)
        for iter_ in range(nr_iter):
            c = 0  # 预测正确的个数
            n = 0  # 总个数
            for words, tags in sentences:
                prev, prev2 = self.START
                context = self.START + words + self.END
                for i, word in enumerate(words):
                    feats = self._get_features(i, word, context, prev, prev2)
                    guess = self.model.predict(feats)
                    self.model.update(tags[i], guess, feats)

                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            if shuf:
                random.shuffle(sentences)  # 将序列的所有元素随机排序,shuffle()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法

            print("Iter {0}: {1}/{2}={3}".format(iter_, c, n, (float(c) / n) * 100))
            self.save(save_loc)

        self.model.average_weights()
        self.save(save_loc)

    def predict(self, words):
        prev, prev2 = self.START
        labels = []
        context = self.START + words + self.END
        for i, word in enumerate(words):
            features = self._get_features(i, word, context, prev, prev2)
            tag = self.model.predict(features)
            labels.append(tag)
            prev2 = prev
            prev = tag
        return labels

    def save(self, loc='model/ap.model', zip=True):
        if not zip:
            pickle.dump((self.model.weights, self.model.classes), open(loc, 'wb'))
        else:
            pickle.dump((self.model.weights, self.model.classes), gzip.open(loc, 'wb'))

    def load(self, loc="model/ap.model", zip=True):
        if not zip:
            self.model.weights, self.model.classes = pickle.load(open(loc, 'rb'))
        else:
            self.model.weights, self.model.classes = pickle.load(gzip.open(loc, 'rb'))

    def _get_features(self, i, word, context, prev, prev2):
        """Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        """

        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i - 1])
        add('i-1 suffix', context[i - 1][-3:])
        add('i-2 word', context[i - 2])
        add('i+1 word', context[i + 1])
        add('i+1 suffix', context[i + 1][-3:])
        add('i+2 word', context[i + 2])
        return features

    def _make_tagdict(self, sentences):
        '''Make a tag dictionary for single-tag words.'''
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                self.model.classes.add(tag)


def train(filepath='data/train.txt', model='model/ap.model', nr_iter=1):
    tagger = Perceptron()
    print('Reading corpus...')  # 读取语料
    training_data = []
    sentence = ([], [])
    fin = open(filepath, 'r', encoding='utf-8')
    for index, line in enumerate(fin):
        line = line.strip()
        if line == '':
            training_data.append(sentence)
            sentence = ([], [])
        else:
            params = line.split()
            if len(params) != 2:
                continue
            sentence[0].append(params[0])
            sentence[1].append(params[1])
    fin.close()
    print('training corpus size : %d', len(training_data))
    print('Start training...')
    tagger.train(training_data, save_loc=model, nr_iter=nr_iter)


def eval(filepath='data/test.txt', model='model/ap.model'):
    tagger = Perceptron(model)

    print('Start testing...')
    right = 0.0
    total = 0.0
    sentence = ([], [])
    fin = open(filepath, 'r', encoding='utf-8')
    for index, line in enumerate(fin):
        line = line.strip()
        if line == '':
            words = sentence[0]
            tags = sentence[1]
            outputs = tagger.predict(words)
            assert len(tags) == len(outputs)
            total += len(tags)
            for o, t in zip(outputs, tags):
                if o == t: right += 1
            sentence = ([], [])
        else:
            params = line.split()
            if len(params) != 2:
                continue
            sentence[0].append(params[0])
            sentence[1].append(params[1])
    fin.close()
    print("Precision : %f", right / total)


def predict(model="model/ap.model"):
    tagger = Perceptron(model)
    while True:
        text = input('>')
        words = list(text)
        labels = tagger.predict(words)

        for word, label in zip(words, labels):
            print(word, label)


if __name__ == '__main__':
    # train()
    eval()
    predict(model="model/ap.model")
