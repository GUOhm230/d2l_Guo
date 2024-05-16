"""
主要学习一下可迭代对象，迭代器和生成器
2023.11 Created by G.tj
"""
from collections.abc import Iterable, Iterator
"""
1. 迭代器：
迭代器类型
1）类中定义了__iter__和__next__两个方法
2）__iter__方法需要返回其对象本身，即self
3) __next__方法，返回下一个数据，如果没有数据量，则需要抛出一个stopIteration异常
"""

# 依据以上的定义创建迭代器类
class IT(object):
    def __init__(self):
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter == 3:
            raise StopIteration
        return self.counter

# 实例化一个迭代器对象: 这个迭代器可以通过next()方法来取值，每个next都取下一个数字。next有点像指针或者程序计数器，每个next之后，指针便指向下一位
obj1 = IT()
v1 = next(obj1) #next()和__next__()方法等价
v2 = obj1.__next__()
# v3 = next(obj1) # 执行到这个，就会抛出异常。但是如果用for的话，就会直接处理异常，而不会输出异常
print(v1, v2)
for i in obj1:
    print(i)
# 因此for循环的执行方式就是先执行__iter__()方法返回一个迭代器对象，然后每次都用next()取值，直到抛出StopIteration异常，则for循环截止。
# 故而，当该类有__next__()和__iter__()则表示定义了一个容器，该容器可以通过for循环进行取值。如果没有__iter__()则表示该容器不是一个可迭代对象，
# 此时，也不能使用for循环进行取值S
# a = [1, 2, 3]
# b = iter(a)
# print(next(b))

print("-----------------------2. 生成器-----------------------")
"""
2. 生成器
所谓生成器，就是函数中有yield，调用该函数之后，得到生成器对象。
生成器类中的__iter__方法返回的是一个生成器对象，而生成器对象的创建就要包含yield。
这个yield就是next执行，也就是每次for循环执行的一个开关，每次遍历都是从上一个yield开始执行.
实际，该生成器对象是内部根据生成器类generator创建的对象。生成器类generator也声明了__iter__和__next__方法
所以说生成器是一种特殊的迭代器
使用yield会使该函数成为一个生成器函数
"""
# 创建生成器对象：方法一：构建一个带有yield的函数即可。
def func():
    yield 1
    yield 2

obj2 = func()

v1 = next(obj2)
print(v1)
v2 = obj2.__next__()
# print(next(obj2)) # 结果同样是抛出异常StopIteration

print("----------------------3. 可迭代对象----------------------")
"""
如果一个类中有__iter__()方法且返回一个迭代器对象，则该类创建的对象为可迭代对象
所以实际上，可迭代对象也要利用迭代器，创建的对象返回的仍然是个迭代器对象。
"""
# 创建一个可迭代对象的类
class Foo(object):
    def __iter__(self):
        return IT()

obj3 = Foo()
for item in iter(obj3): # for循环的时候优先执行iter()方法，获取一个迭代器对象。因此直接for item in obj3同样有效
    print(item)
# v1 = next(iter(obj3))
# print(v1)

# 使用range为例
import torch
v1 = torch.arange(1000)
v2 = iter(v1)
if '__next__' in dir(v1):
    print("是迭代器")
else:
    print("不是迭代器")
# print(dir(v2))
# （1）.用迭代器
# 自己尝试用迭代器和可迭代对象自定义range.range是一个可迭代对象，输出从0到n-1的实数
# 先定义一个迭代器，迭代器要实现iter和next方法
class IterRange(object):
    def __init__(self, num):
        self.counter = -1
        self.num = num # 类和对象的小问题居然还有些# v1 = next(iter(obj3))
# print(v1)不清楚。传的形参还要作为实参

    def __iter__(self): # iter返回迭代器对象，迭代器类则返回自己
        return self

    def __next__(self):
        self.counter += 1
        if self.counter >= self.num: # 这里的形参如果不传过来
            raise StopIteration
        return self.counter

# 创建一个这样的可迭代对象
class MyRange(object):
    def __init__(self, num):
        self.num = num

    def __iter__(self):
        return IterRange(self.num)

# 使用生成器重写range
class GeneratorRange(object):
    def __init__(self, num):
        self.counter = -1
        self.num = num

    # def __next__(self): # 这里其实是错的，说明我对生成器的原理不是很清楚。实际是调用for循环时先调用iter
    def __iter__(self):
        # 这里不是很懂
        counter = 0
        while counter < self.num:
            yield counter # 返回yield之后的值。接下来的next就执行yield之后的语句，直到输出yield后的值
            counter += 1

# 创建一个生成器对象
obj4 = GeneratorRange(10) #是一个可迭代对象
v4 = dir(obj4)

v5 = iter(obj4)
print(next(v5))
v6 = next(v5)
if '__next__' in dir(v5):
    print("有")
# for item in obj4:
#     print(item)
# 为什么能起作用呢？
# 先调用iter
# 因此可以通过这些东西创建一个
print("------------------------4. 常见数据类型-------------------------------")
# 1.列表，元组，集合，
# 2. 判断是否为迭代器或者可迭代对象：instance(obj, iterable),instance(obj, iterator)
print(isinstance(obj4, Iterable))
print(isinstance(v5, Iterator))
print("------------------------5. yield关键字-------------------------------")
def yieldTest():
    for i in range(10):
        print("yield开始执行")
        yield i
        print("执行yield语句之后")

y = yieldTest()
print(next(iter(y)))
print("------------------------------")
print(next(iter(y)))
# for i in y:
#     print(i)