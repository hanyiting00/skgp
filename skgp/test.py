from itertools import count, product

# #encoding:UTF-8
# def yield_test(n):
#     for i in range(n):
#         yield call(i)
#         print("i=",i)
#     #做一些其它的事情
#     print("do something.")
#     print("end.")
#
# def call(i):
#     return i*2
#
# #使用for循环
# for i in yield_test(5):
#     print(i,",")


# a = [[0.0 for _ in range(10)] for _ in range(20)]
# print(a)

# a= [11,22,33,44,55]
# b = zip(a,count())
# print([s for s in b])


# for i,j,k in product(range(10), repeat=3):
#     print(i,j,k)

a = {(1, 2, 3, 4, 5), (2, 1, 4, 5, 3), (7, 6, 5, 4, 3)}
b = {(1, 2, 3, 4, 5)}
c = a-b
print(c)
