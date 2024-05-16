l = [2, 3, 4, 1]
ls = [7, 8, 9]
l.sort()



# 给定一个子列表，添加一个元素后，返回所有可取的金额
# coins = [1, 4, 10, 5, 7, 19]
coins = [1, 1, 1]
target = 20
def subCoinsGet(subCoins, addCoin):
    return subCoins + [addCoin + selm for selm in subCoins if addCoin + selm <= target]

coins.sort()  # 先给coins输入排序
subCoins = [0]  # 保存循环至当前元素时，可能组成的可取得金额
result = []  # 需要添加的面值
# 对coins中每个元素进行循环，并获取其中前面的子序列所有的可能取值
for i in range(len(coins)-1):
    subCoins = subCoinsGet(subCoins, coins[i])
    print(subCoins)
    start = subCoins[-1]
    for j in range(start, coins[i + 1]):
        print(subCoins[-1], coins[i + 1])
        if j not in subCoins:
            print(j)
            result.append(j)
            subCoins = subCoinsGet(subCoins, j)
            print("修复后的subCoins:", subCoins)
print(result)
