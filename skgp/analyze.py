import os

from skgp import perceptron
from skgp.segment.jieba_segment import JiebaSegment
from skgp.textrank import Keywords, Summarize


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class Analyze(object):
    def __init__(self):
        self.segment = JiebaSegment()
        self.ner_model = None
        self.keywords_model = None
        self.summarize_model = None

    def init(self):
        self.init_ner()

    def init_ner(self):
        if self.ner_model is None:
            self.ner_model = perceptron.Perceptron(add_curr_dir("model/ner.model"))

    def seg(self, sentence, cut_all=False):
        return self.segment.seg(sentence, cut_all)

    def pos(self, words):
        labels = self.segment.pos(words)
        return labels

    def ner(self, words):  # 传入的是词语
        self.init_ner()
        labels = self.ner_model.predict(words)
        return labels

    def keywords(self, text, topkey=5):  # 关键字抽取
        if self.keywords_model is None:
            self.keywords_model = Keywords(tol=0.0001, window=2)
        return self.keywords_model.keywords(text, topkey)

    def summarize(self, text, topsen=5):  # 文本摘要
        if self.summarize_model is None:
            self.summarize_model = Summarize(tol=0.0001)
        return self.summarize_model.summarize(text, topsen)
