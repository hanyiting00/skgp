import jieba


# 结巴分词
class JiebaSegment:

    def seg(self, sentence):
        return jieba.lcut(sentence)


if __name__ == '__main__':
    j = JiebaSegment()

    text = '春天的花开秋天的风以及冬天的落阳'
    result = j.seg(text)
    print(result)

