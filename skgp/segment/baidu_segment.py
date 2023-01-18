from LAC import LAC


# 百度分词
class BaiduSegment:

    def __init__(self, model_path=None, mode="lac", custom_path="../dict/custom.txt"):
        self.lac = LAC(mode=mode, model_path=model_path)
        self.lac.load_customization(custom_path)

    # mode:分词模式 seg、lac、rank
    def seg(self, sentence, mode="lac"):
        # 装载分词模型
        self.lac = LAC(mode=mode)
        return self.lac.run(sentence)


if __name__ == '__main__':
    s = BaiduSegment()

    # 单个样本输入，输入为Unicode编码的字符串
    text = '春天的花开秋天的风以及冬天的落阳'
    lac_result = s.seg(text, mode="seg")
    print(lac_result)
    lac_result = s.seg(text, mode="lac")
    print(lac_result)
    lac_result = s.seg(text, mode="rank")
    print(lac_result)

    # 批量样本输入, 输入为多个句子组成的list，平均速率会更快
    texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
    lac_result = s.seg(texts, mode="seg")
    print(lac_result)
