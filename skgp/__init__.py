from skgp import analyze
from skgp.segment.jieba_segment import JiebaSegment

any = analyze.Analyze()

# 分词
seg = any.seg

# 词性标注
pos = any.pos

# 命名实体识别
ner = any.ner

# 关键字抽取
keywords = any.keywords

# 中文摘要
summarize = any.summarize
