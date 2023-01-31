from skgp import analyze
from skgp.segment.jieba_segment import JiebaSegment

any = analyze.Analyze()

# 分词
seg = any.seg

# 词性标注
pos = any.pos

# 命名实体识别
ner = any.ner


load_userdict = any.load_userdict

# 关键字抽取
keywords = any.keywords

# 中文摘要
summarize = any.summarize

# 新词发现
findword = any.findword

# 知识图谱
knowledge = any.knowledge

# 情感分析
sentiment = any.sentiment
