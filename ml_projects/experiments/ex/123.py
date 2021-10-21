import cv2
import numpy as np
from mss import mss
from typing import List
from collections import Counter
from itertools import groupby
import math
import re
import timeit
import copy


class Solution:
    def reverse(self, x: int) -> int:
        if abs(x) < 2 ** 31 and x != 2 ** 31 - 1:
            if x == 0:
                return 0
            if x > 0:
                return int("".join(list(str(x))[::-1]))
            if x < 0:
                return int("".join([str(i) for i in str(x) if i.isdigit()][::-1])) * -1
        else:
            return 0



# nums1 = 123  # 321
# nums1 = -123  # -321
# nums1 = "aa"  # t
# nums1 = 120  # 21
# nums1 = 0  # 0
nums1 = 1534236469  # 0

# start = timeit.default_timer()
cv = Solution().reverse(x=nums1)
print(cv)
# elapsed = timeit.default_timer() - start
# print(f"time: {elapsed}")
