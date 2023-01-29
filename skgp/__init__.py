from skgp import analyze
from skgp.segment.jieba_segment import JiebaSegment


any = analyze.Analyze()

# 分词
seg = any.seg

# 词性标注
pos = any.pos

