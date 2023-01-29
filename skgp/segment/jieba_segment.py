import os
import jieba
import jieba.analyse
from jieba import enable_paddle, posseg
from jieba.posseg import dt
import paddle

paddle.enable_static()


# 结巴分词
class JiebaSegment:

    def add_curr_dir(self, name):
        return os.path.join(os.path.dirname(__file__), name)

    def load_userdict(self, custom_path):
        jieba.load_userdict(self.add_curr_dir(custom_path))  # 装入词典

    # cut_all, False：默认精确模式；True：全模式
    def seg(self, sentence, cut_all=False):
        return jieba.lcut(sentence, cut_all=cut_all)

    def delword(self, word):
        jieba.del_word(word)

    def addword(self, word):
        jieba.add_word(word)

    def suggestfreq(self, word, tune=False):
        jieba.suggest_freq(word, tune)

    # 提取topK词汇
    def extracttags(self, text, topK=20):
        return jieba.analyse.extract_tags(text, topK)

    # 获取提取的关键词的权重
    def textrank(self, text, topK=20, withWeight=True):
        keyword_dict = {}
        for x, w in jieba.analyse.textrank(str_text, topK, withWeight=True):
            keyword_dict[x] = w
        return keyword_dict

    # paddle模式,词性标注
    def pseg(self, text, use_paddle=True):
        return posseg.lcut(text, use_paddle)

    # 返回词性标注
    def pos(self, words):
        pos = []
        for word in words:
            pos.append(dt.word_tag_tab.get(word))
        return pos


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
    #
    # j.delword("中将")
    # j.addword("将出错")
    # result = j.seg(text, cut_all=False)
    # print(result)
    #
    # j.addword("post中")
    # j.suggestfreq("post中", True)
    # result = j.seg(text, cut_all=False)
    # print(result)

    str_text = "线程（英语：thread" \
               "）是操作系统能够进行运算调度的最小单位。它被包含在进程之中，是进程中的实际运作单位。一条线程指的是进程中一个单一顺序的控制流，一个进程中可以并发多个线程，每条线程并行执行不同的任务。在Unix " \
               "System V及SunOS中也被称为轻量进程（lightweight processes），但轻量进程更多指内核线程（kernel thread），而把用户线程（user thread）称为线程。 "
    # keywords_top1 = j.extracttags(str_text, 10)
    # print("/".join(keywords_top1))

    # print(j.textrank(str_text, topK=10, withWeight=True))

    # print(j.pseg(str_text))

    str_text = '苏州的天气不错'
    print(j.pseg(str_text, use_paddle=True))

