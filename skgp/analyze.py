import os

from skgp.segment.jieba_segment import JiebaSegment


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class Analyze(object):
    def __init__(self):
        self.segment = JiebaSegment()
        self.ner_model =None


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
        labels = []
        return labels