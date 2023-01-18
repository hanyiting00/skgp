import jieba


# 结巴分词
class JiebaSegment:

    def __init__(self, custom_path="dict/custom.txt"):
        jieba.load_userdict(custom_path)  # 装入词典

    # cut_all, False：默认精确模式；True：全模式
    def seg(self, sentence, cut_all=False):
        return jieba.lcut(sentence, cut_all=cut_all)

    def delword(self, word):
        jieba.del_word(word)

    def addword(self, word):
        jieba.add_word(word)

    def suggestfreq(self, word, tune=False):
        jieba.suggest_freq(word, tune)


if __name__ == '__main__':
    j = JiebaSegment()

    text = '如果放到post中将出错。'
    test_sent = (
        "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
        "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
        "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
    )
    result = j.seg(test_sent)
    print(result)
    result = j.seg(text, cut_all=False)
    print(result)

    j.delword("中将")
    j.addword("将出错")
    result = j.seg(text, cut_all=False)
    print(result)

    j.addword("post中")
    j.suggestfreq("post中", True)
    result = j.seg(text, cut_all=False)
    print(result)