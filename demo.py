import skgp

# text = '苏州的天气不错'
# words = skgp.seg(text)  # 分词
# print(words)

# text = '李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿'
# words = skgp.seg(text, cut_all=True)  # 分词
# print(words)

# pos = skgp.pos(words)  # 词性标注
# print(pos)
#
#
# ner = skgp.ner(words)  # 命名实体识别
# print(ner)

text = '''
该研究主持者之一、波士顿大学地球与环境科学系博士陈池（音）表示，“尽管中国和印度国土面积仅占全球陆地的9%，但两国为这一绿化过程贡献超过三分之一。考虑到人口过多的国家一般存在对土地过度利用的问题，这个发现令人吃惊。”
NASA埃姆斯研究中心的科学家拉玛·内曼尼（Rama Nemani）说，“这一长期数据能让我们深入分析地表绿化背后的影响因素。我们一开始以为，植被增加是由于更多二氧化碳排放，导致气候更加温暖、潮湿，适宜生长。”
“MODIS的数据让我们能在非常小的尺度上理解这一现象，我们发现人类活动也作出了贡献。”
NASA文章介绍，在中国为全球绿化进程做出的贡献中，有42%来源于植树造林工程，对于减少土壤侵蚀、空气污染与气候变化发挥了作用。
据观察者网过往报道，2017年我国全国共完成造林736.2万公顷、森林抚育830.2万公顷。其中，天然林资源保护工程完成造林26万公顷，退耕还林工程完成造林91.2万公顷。京津风沙源治理工程完成造林18.5万公顷。三北及长江流域等重点防护林体系工程完成造林99.1万公顷。完成国家储备林建设任务68万公顷。
'''
# keywords = skgp.keywords(text, 5)  # 关键词抽取
# print(keywords)

# summarize = skgp.summarize(text, 3)  # 文本摘要
# print(summarize)

# skgp.findword('input.txt', 'output.txt') # 根据大规模语料，利用信息熵做新词发现。

# 知识图谱关系抽取
text = '姚明1980年9月12日出生于上海市徐汇区，祖籍江苏省苏州市吴江区震泽镇，前中国职业篮球运动员，司职中锋，现任中职联公司董事长兼总经理。'
knowledge =skgp.knowledge(text)
print(knowledge)
