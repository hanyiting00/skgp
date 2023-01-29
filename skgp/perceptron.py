import gzip
import pickle
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


class Perceptron:
    def __init__(self, loc=None):
        self.START = ['-START-', '-START2-']
        self.END = ['-END-', '-END2-']
        self.model = AveragedPerceptron()

        if loc is not None:
            self.load(loc)

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
    # eval()
    predict(model="model/ner.model")
