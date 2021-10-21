from typing import List

from collections import defaultdict
from collections import deque

class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        asd = []
        c = 1
        f = nums[0]
        for i in nums[1:]:
            if i > f:
                c += 1
                f = i
            else:
                f = i
                asd.append(c)
                c = 1
        cv = sorted(asd)
        return sorted(asd)[-1]
        # for idx, i in enumerate(nums[1:]):
        #     if i > asd[-1] and :
        #         asd.append(i)
        # print(f"ff: {asd}")
        # print(f"ff: {len(asd)}")




# nums = [1,3,5,4,7]  # 3
# nums = [2,2,2,2,2]  # 1
nums = [1,3,5,4,7]  # 3
fo = Solution().findLengthOfLCIS(nums=nums)
print(f"f: {fo}")
