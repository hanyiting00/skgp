import re
import os
from math import log

re_eng = re.compile('[a-zA-Z0-9]', re.U)
re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
re_skip = re.compile("(\r\n|\s)", re.U)


class Segment:
    def __init__(self):
        self.vocab = {}
        self.max_word_len = 0
        self.max_freq = 0
        self.total_freq = 0
        self.initialized = False

    def init(self, vocab_path="dict/custom.dict"):
        self.load_vocab(os.path.join(os.path.dirname(__file__), vocab_path))
        self.initialized = True

    def load_vocab(self, vocab_path):
        fin = open(vocab_path, 'r', encoding='utf-8')
        for index, line in enumerate(fin):
            line = line.strip()
            if line == '':
                continue
            word_freq_tag = line.split('\t')
            if len(word_freq_tag) == 1:
                word = word_freq_tag[0]
                self.add_vocab(word)
            elif len(word_freq_tag) == 2:
                word = word_freq_tag[0]
                freq = int(word_freq_tag[1])
                self.add_vocab(word, freq)
        fin.close()

    def add_vocab(self, word=None, freq=None, tag=None):
        if freq is None:
            freq = self.max_freq

        if word not in self.vocab:
            self.vocab[word] = 0

        self.vocab[word] += freq
        self.total_freq += freq

        if freq > self.max_freq:
            self.max_freq = freq

        if len(word) > self.max_word_len:
            self.max_word_len = len(word)

    def load_userdict(self, userdict):
        if not self.initialized:
            self.init()

        if isinstance(userdict, str):
            self.load_vocab(userdict)

        for item in userdict:
            if isinstance(item, list):
                if len(item) == 1:
                    word = item[0]
                    self.add_vocab(word)
                elif len(item) == 2:
                    word = item[0]
                    freq = item[1]
                    self.add_vocab(word, freq)
            elif isinstance(item, str):
                self.add_vocab(word=item)

    def seg(self, sentence, mode="default"):
        if not self.initialized:
            self.init()

        if mode == "probe":
            return list(self.seg_new_word(sentence))
        else:
            return list(self.seg_default(sentence))

    def cut_words(self, sentence):
        DAG = self.create_DAG(sentence)
        route = {}
        self.calc_route(sentence, DAG, route)

        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''


    def calc_route(self, sentence, DAG, route):
        vocab = self.vocab
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total_freq)

        for idx in range(N - 1, -1, -1):
            route[idx] = max(
                (log(vocab.get(sentence[idx:idx + 1]) or 1) - logtotal + route[x + 1][0], x) for x in DAG[idx])

    def create_DAG(self, sentence):
        vocab = self.vocab
        max_word_len = self.max_word_len
        DAG = {}
        N = len(sentence)
        for idx in range(N):
            cand_idx = [idx]
            for i in range(idx + 1, idx + min(max_word_len, N - idx), 1):
                cand = sentence[idx: i + 1]
                if cand in vocab:
                    cand_idx.append(i)
            DAG[idx] = cand_idx
        return DAG

    def seg_default(self, sentence):
        blocks = re_han.split(sentence)
        cut_block = self.cut_words
        cut_all = False
        for block in blocks:
            if not block:
                continue
            if re_han.match(block):
                for word in cut_block(block):
                    yield word


if __name__ == '__main__':
    s = Segment()
    s.load_userdict(['金凤区'])
    text = "银川市金凤区北京中路福宁城11-1-号"
    words = s.seg(text)
    print(words)
