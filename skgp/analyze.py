import os

from skgp.segment.jieba_segment import JiebaSegment


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class Analyze(object):
    def __init__(self):
        self.segment = JiebaSegment()

    def seg(self, sentence, cut_all=False):
        return self.segment.seg(sentence, cut_all)

    def pos(self, words):
        labels = self.segment.pos(words)
        return labels
