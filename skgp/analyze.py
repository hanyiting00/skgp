import os

from skgp import perceptron, findword
from skgp.segment.jieba_segment import JiebaSegment
from skgp.sentiment.bayes import Bayes
from skgp.textrank import Keywords, Summarize
from skgp.cluster.text_cluster import text_cluster as cluster


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class Analyze(object):
    def __init__(self):
        self.segment = JiebaSegment()
        self.ner_model = None
        self.kg_model = None
        self.keywords_model = None
        self.summarize_model = None

        self.sentiment_model = Bayes()

    def init(self):
        self.init_ner()

    def load_userdict(self, userdict="../dict/custom.dict"):
        self.segment.load_userdict(userdict)

    def init_ner(self):
        if self.ner_model is None:
            self.ner_model = perceptron.Perceptron(add_curr_dir("model/ner.model"))

    def init_kg(self):
        if self.kg_model is None:
            self.kg_model = perceptron.Perceptron(add_curr_dir("model/kg.model"))

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

    def findword(self, input_file, output_file, min_freq=10, min_mtro=80, min_entro=3):  # 发现新词
        findword.new_word_find(input_file, output_file, min_freq, min_mtro, min_entro)

    def knowledge(self, text):  # 知识图谱关系，传入的是文本
        self.init_kg()
        words = self.seg(text)
        labels = self.kg_model.predict(words)
        return self.lab2spo(words, labels)

    def lab2spo(self, words, epp_labels):
        subject_list = []  # 存放实体的列表
        object_list = []
        index = 0
        for word, ep in zip(words, epp_labels):
            if ep[0] == 'B' and ep[2:] == '实体':
                subject_list.append([word, ep[2:], index])
            elif (ep[0] == 'I' or ep[0] == 'E') and ep[2:] == '实体':
                if len(subject_list) == 0:
                    continue
                subject_list[len(subject_list) - 1][0] += word

            if ep[0] == 'B' and ep[2:] != '实体':
                object_list.append([word, ep[2:], index])
            elif (ep[0] == 'I' or ep[0] == 'E') and ep[2:] != '实体':
                if len(object_list) == 0:
                    return []
                object_list[len(object_list) - 1][0] += word

            index += 1

        spo_list = []
        if len(subject_list) == 0 or len(object_list) == 0:
            pass
        elif len(subject_list) == 1:
            entity = subject_list[0]
            for obj in object_list:
                predicate = obj[1][:-1]
                spo_list.append([entity[0], predicate, obj[0]])
        else:
            for obj in object_list:
                entity = []
                predicate = obj[1][:-1]
                direction = obj[1][-1]
                for sub in subject_list:
                    if direction == '+':
                        if sub[2] > obj[2]:
                            entity = sub
                            break
                    else:
                        if sub[2] < obj[2]:
                            entity = sub

                if not entity:
                    continue

                spo_list.append([entity[0], predicate, obj[0]])

        return spo_list

    def sentiment(self, text):  # 情感分析
        words = self.seg(text)
        ret, prob = self.sentiment_model.classify(words)
        return ret, prob

    def text_cluster(self, docs, features_method='tfidf', method="k-means", k=3, max_iter=100, eps=0.5, min_pts=2):
        return cluster(docs, features_method, method, k, max_iter, eps, min_pts, self.seg)
