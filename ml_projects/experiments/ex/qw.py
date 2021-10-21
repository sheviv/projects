from itertools import groupby
n = 1534236469
# n = 123
# print(f"n:{abs(n) <= 0xffffffff}")


def reverse(x):
    negative = False
    if x < 0:
        x = x * -1
        negative = True
    else:
        x = x
    sum = 0
    dig = 1
    strX = str(x)
    lst = list(strX)
    for i in lst:
        sum += int(i) * dig
        dig *= 10
    if abs(sum) > ((1 << 31) - 1):
        return 0
    elif negative:
        return sum * -1
    else:
        return sum
zxc = reverse(x=n)
print(f"rev:{zxc}")
# flowerbed = [1,0,0,0,1]
# n = 1  # t
flowerbed = [0,0,1,0,0]
n = 2  # t
# [1,0,0,0,0,0,1]
# flowerbed = [1,0,0,0,1]
# n = 2  # f
# flowerbed = [1,0,0,0,1,0,0]
# n = 2  # f
# print(canPlaceFlowers(flowerbed=flowerbed, n=n))
