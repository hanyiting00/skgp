import skgp

text = '苏州的天气不错'
words = skgp.seg(text)  # 分词
print(words)

# text = '李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿'
# words = skgp.seg(text, cut_all=True)  # 分词
# print(words)

pos = skgp.pos(words)  # 词性标注
print(pos)

