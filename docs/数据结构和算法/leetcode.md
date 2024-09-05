# LeetCode 题目

在 LeetCode 刷题过程中，掌握常见的算法题型和实现方法非常重要。以下是几道常见的 LeetCode 题目及其 Python 实现，涵盖数组、链表、动态规划等不同领域的算法。

### 1. 两数之和（Two Sum）

**题目**：给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那两个整数，并返回它们的数组下标。

**LeetCode 链接**：[两数之和](https://leetcode.com/problems/two-sum/)

#### Python 实现

```python
def two_sum(nums, target):
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i

# 使用示例
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # 输出: [0, 1]
```

#### 详解

- 使用哈希表记录数组元素及其索引，通过检查 `target - num` 是否在哈希表中，时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 2. 盛最多水的容器（Container With Most Water）

**题目**：给定一个数组 `height`，其中每个元素表示一个垂直线的高度。找出两条线所围成的容器可以盛最多的水。

**LeetCode 链接**：[盛最多水的容器](https://leetcode.com/problems/container-with-most-water/)

#### Python 实现

```python
def max_area(height):
    l, r = 0, len(height) - 1
    max_area = 0
    while l < r:
        max_area = max(max_area, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return max_area

# 使用示例
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print(max_area(height))  # 输出: 49
```

#### 详解

- 使用双指针法，逐步移动左右两端，计算并更新当前最大面积。时间复杂度为 O(n)。

---

### 3. 三数之和（3Sum）

**题目**：给定一个包含 `n` 个整数的数组，判断数组中是否存在三个元素，使得它们的和为零。找出所有满足条件且不重复的三元组。

**LeetCode 链接**：[三数之和](https://leetcode.com/problems/3sum/)

#### Python 实现

```python
def three_sum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1
                r -= 1
    return res

# 使用示例
nums = [-1, 0, 1, 2, -1, -4]
print(three_sum(nums))  # 输出: [[-1, -1, 2], [-1, 0, 1]]
```

#### 详解

- 先将数组排序，使用双指针技巧找出三元组。时间复杂度为 O(n²)。

---

### 4. 删除排序数组中的重复项（Remove Duplicates from Sorted Array）

**题目**：给定一个有序数组 `nums`，你需要原地删除其中的重复元素，使得每个元素只出现一次，返回删除后数组的新长度。

**LeetCode 链接**：[删除排序数组中的重复项](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

#### Python 实现

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1

# 使用示例
nums = [1, 1, 2]
print(remove_duplicates(nums))  # 输出: 2，nums 数组变为 [1, 2]
```

#### 详解

- 使用双指针法，一个指针 `i` 指向当前去重的最后一个元素，另一个指针 `j` 遍历整个数组。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 5. 二叉树的最大深度（Maximum Depth of Binary Tree）

**题目**：给定一个二叉树，找出其最大深度。

**LeetCode 链接**：[二叉树的最大深度](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    if not root:
        return 0
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    return max(left_depth, right_depth) + 1

# 使用示例
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20, TreeNode(15), TreeNode(7))
print(max_depth(root))  # 输出: 3
```

#### 详解

- 使用递归法计算每个节点的最大深度。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 `n` 为节点总数，`h` 为树的高度。

---

### 6. 斐波那契数列（Fibonacci Sequence）

**题目**：求解斐波那契数列的第 `n` 项。

**LeetCode 链接**：[斐波那契数列](https://leetcode.com/problems/fibonacci-number/)

#### Python 实现

```python
def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 使用示例
n = 10
print(fib(n))  # 输出: 55
```

#### 详解

- 通过迭代法优化递归方式，避免重复计算。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

以下是更多经典的 LeetCode 题目及其 Python 实现，涵盖更多常见算法和数据结构，包括动态规划、二叉树、回溯、贪心算法等。

### 7. 爬楼梯（Climbing Stairs）

**题目**：假设你正在爬楼梯。需要 `n` 阶才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶？

**LeetCode 链接**：[Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

#### Python 实现

```python
def climb_stairs(n):
    if n == 1:
        return 1
    a, b = 1, 2
    for i in range(3, n + 1):
        a, b = b, a + b
    return b

# 使用示例
n = 5
print(climb_stairs(n))  # 输出: 8
```

#### 详解

- 这是经典的动态规划问题，类似于斐波那契数列。每一步的选择可以通过前两步来决定。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 8. 组合总和（Combination Sum）

**题目**：给定一个无重复元素的数组 `candidates` 和一个目标数 `target`，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

**LeetCode 链接**：[Combination Sum](https://leetcode.com/problems/combination-sum/)

#### Python 实现

```python
def combination_sum(candidates, target):
    res = []
    def dfs(start, path, target):
        if target == 0:
            res.append(path)
            return
        for i in range(start, len(candidates)):
            if candidates[i] <= target:
                dfs(i, path + [candidates[i]], target - candidates[i])
    
    dfs(0, [], target)
    return res

# 使用示例
candidates = [2, 3, 6, 7]
target = 7
print(combination_sum(candidates, target))  # 输出: [[2, 2, 3], [7]]
```

#### 详解

- 使用深度优先搜索（DFS）来递归查找所有组合。每次从当前数字开始递归，避免重复选择。时间复杂度较高，取决于输入大小和解空间。

---

### 9. 最大子序和（Maximum Subarray）

**题目**：给定一个整数数组 `nums`，找到一个具有最大和的连续子数组，返回其最大和。

**LeetCode 链接**：[Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

#### Python 实现

```python
def max_sub_array(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

# 使用示例
nums = [-2,1,-3,4,-1,2,1,-5,4]
print(max_sub_array(nums))  # 输出: 6
```

#### 详解

- 使用动态规划的思想，每次选择当前元素或者与前面的子数组相加，来获得当前最大和。时间复杂度为 O(n)。

---

### 10. 不同路径（Unique Paths）

**题目**：一个机器人位于 m x n 网格的左上角。机器人每次只能向下或者向右移动一步，机器人试图达到网格的右下角。计算共有多少条不同的路径。

**LeetCode 链接**：[Unique Paths](https://leetcode.com/problems/unique-paths/)

#### Python 实现

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

# 使用示例
m, n = 3, 7
print(unique_paths(m, n))  # 输出: 28
```

#### 详解

- 这是一个典型的动态规划问题。使用二维数组 `dp` 来存储到达每个位置的路径数，结果为右下角的路径数。时间复杂度为 O(m * n)。

---

### 11. 买卖股票的最佳时机（Best Time to Buy and Sell Stock）

**题目**：给定一个数组 `prices`，其中 `prices[i]` 是一支股票第 `i` 天的价格。你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。

**LeetCode 链接**：[Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

#### Python 实现

```python
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

# 使用示例
prices = [7, 1, 5, 3, 6, 4]
print(max_profit(prices))  # 输出: 5
```

#### 详解

- 这是一个贪心算法问题。我们通过遍历找到最低的买入价格，并计算当前价格与最低价格之间的差值作为可能的最大利润。时间复杂度为 O(n)。

---

### 12. 岛屿数量（Number of Islands）

**题目**：给定一个由 '1'（陆地）和 '0'（水）组成的二维网格，计算岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由水平和垂直方向上相邻的陆地连接。

**LeetCode 链接**：[Number of Islands](https://leetcode.com/problems/number-of-islands/)

#### Python 实现

```python
def num_islands(grid):
    if not grid:
        return 0

    def dfs(i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        dfs(i-1, j)
        dfs(i+1, j)
        dfs(i, j-1)
        dfs(i, j+1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count

# 使用示例
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
print(num_islands(grid))  # 输出: 1
```

#### 详解

- 使用深度优先搜索（DFS）来遍历所有陆地，将相连的陆地标记为水。时间复杂度为 O(m * n)，其中 m 是行数，n 是列数。

---

### 13. 全排列（Permutations）

**题目**：给定一个没有重复数字的序列，返回所有可能的全排列。

**LeetCode 链接**：[Permutations](https://leetcode.com/problems/permutations/)

#### Python 实现

```python
def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i, num in enumerate(nums):
            if used[i]:
                continue
            path.append(num)
            used[i] = True
            backtrack(path, used)
            path.pop()
            used[i] = False

    backtrack([], [False] * len(nums))
    return res

# 使用示例
nums = [1, 2, 3]
print(permute(nums))  # 输出: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```

#### 详解

- 使用回溯法生成所有排列，通过递归和状态回溯来构建全排列。时间复杂度为 O(n!)。

---

继续深入一些经典 LeetCode 题目及其 Python 实现，涉及更多领域和复杂度的题目。这些题目包括回溯算法、贪心算法、图算法等。

### 14. 组合总和 II（Combination Sum II）

**题目**：给定一个包含 `n` 个整数的数组 `candidates` 和一个目标数 `target`，找出 `candidates` 中所有可以使数字和为 `target` 的组合。每个组合中的数字可以使用多次，但每个数字在组合中只能出现一次。

**LeetCode 链接**：[Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

#### Python 实现

```python
def combination_sum2(candidates, target):
    res = []
    candidates.sort()
    
    def backtrack(start, path, target):
        if target == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            if candidates[i] > target:
                break
            path.append(candidates[i])
            backtrack(i + 1, path, target - candidates[i])
            path.pop()
    
    backtrack(0, [], target)
    return res

# 使用示例
candidates = [10, 1, 2, 7, 6, 5]
target = 8
print(combination_sum2(candidates, target))  # 输出: [[1, 2, 5], [1, 7], [2, 6]]
```

#### 详解

- 使用回溯法来生成所有组合。排序数组可以帮助我们更快地剪枝。避免重复组合的方法是跳过相同的元素。时间复杂度为 O(2^n)。

---

### 15. 合并区间（Merge Intervals）

**题目**：给定一个区间的集合，合并所有重叠的区间。

**LeetCode 链接**：[Merge Intervals](https://leetcode.com/problems/merge-intervals/)

#### Python 实现

```python
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            last_merged[1] = max(last_merged[1], current[1])
        else:
            merged.append(current)
    
    return merged

# 使用示例
intervals = [[1,3],[2,6],[8,10],[15,18]]
print(merge(intervals))  # 输出: [[1, 6], [8, 10], [15, 18]]
```

#### 详解

- 首先将区间按起始位置排序，然后遍历区间，合并重叠部分。时间复杂度为 O(n log n)。

---

### 16. 单词拆分 II（Word Break II）

**题目**：给定一个字符串 `s` 和一个字典 `wordDict`，请你将 `s` 拆分成若干个词汇，并返回所有可能的拆分结果。

**LeetCode 链接**：[Word Break II](https://leetcode.com/problems/word-break-ii/)

#### Python 实现

```python
def word_break(s, word_dict):
    memo = {}
    
    def backtrack(start):
        if start in memo:
            return memo[start]
        if start == len(s):
            return [[]]
        
        res = []
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_dict:
                for sub_sentence in backtrack(end):
                    res.append([s[start:end]] + sub_sentence)
        
        memo[start] = res
        return res
    
    word_dict = set(word_dict)
    return [' '.join(words) for words in backtrack(0)]

# 使用示例
s = "catsanddog"
word_dict = ["cat", "cats", "and", "sand", "dog"]
print(word_break(s, word_dict))  # 输出: ["cat sand dog", "cats and dog"]
```

#### 详解

- 使用回溯法结合记忆化搜索来处理子问题。时间复杂度较高，但通过记忆化减少重复计算。时间复杂度为 O(2^n)。

---

### 17. 最长递增子序列（Longest Increasing Subsequence）

**题目**：给定一个无序整数数组，找到其中最长递增子序列的长度。

**LeetCode 链接**：[Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

#### Python 实现

```python
def length_of_lis(nums):
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# 使用示例
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(length_of_lis(nums))  # 输出: 4
```

#### 详解

- 使用动态规划来解决这个问题。时间复杂度为 O(n^2)。可以通过二分查找优化到 O(n log n)。

---

### 18. 图的最短路径（Shortest Path in a Graph）

**题目**：给定一个带权图，找出从起点到终点的最短路径。

**LeetCode 链接**：[Shortest Path in a Graph](https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/)

#### Python 实现（Dijkstra 算法）

```python
import heapq

def shortest_path(grid):
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def in_bounds(x, y):
        return 0 <= x < rows and 0 <= y < cols
    
    pq = [(0, 0, 0)]  # (cost, x, y)
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[0][0] = 0
    
    while pq:
        cost, x, y = heapq.heappop(pq)
        if x == rows - 1 and y == cols - 1:
            return cost
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny):
                new_cost = cost + grid[nx][ny]
                if new_cost < dist[nx][ny]:
                    dist[nx][ny] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))
    
    return -1

# 使用示例
grid = [[0, 1, 1], [1, 1, 1], [1, 0, 0]]
print(shortest_path(grid))  # 输出: 4
```

#### 详解

- 使用 Dijkstra 算法来解决最短路径问题。优先队列用于高效选择当前最小代价的节点。时间复杂度为 O((V + E) log V)，其中 V 为节点数，E 为边数。

---

### 19. 从前序与中序遍历序列构造二叉树（Construct Binary Tree from Preorder and Inorder Traversal）

**题目**：给定前序遍历和中序遍历的结果，构造出对应的二叉树。

**LeetCode 链接**：[Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root_val = preorder[0]
    root = TreeNode(root_val)
    root_index = inorder.index(root_val)
    
    root.left = build_tree(preorder[1:1 + root_index], inorder[:root_index])
    root.right = build_tree(preorder[1 + root_index:], inorder[root_index + 1:])
    
    return root

# 使用示例
preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]
tree = build_tree(preorder, inorder)
```

#### 详解

- 利用前序遍历的第一个元素作为根节点，然后根据中序遍历将左右子树划分。递归构建左右子树。时间复杂度为 O(n^2)，可以通过哈希表优化为 O(n)。

---

### 20. 汉明距离（Hamming Distance）

**题目**：计算两个整数之间的汉明距离。汉明距离是两个字符串对应位置上不同字符的数量。

**LeetCode 链接**：[Hamming Distance](https://leetcode.com/problems/hamming-distance/)

#### Python 实现

```python
def hamming_distance(x, y):
    return bin(x ^ y).count('1')

# 使用示例
x, y = 1, 4
print(hamming_distance(x, y))  # 输出: 2
```

#### 详解

- 通过异或运算找出两个数字的不同位数，然后计算结果中 1 的个数。时间复杂度为 O(1)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖更广泛的算法和数据结构。

### 21. 寻找两个正序数组的中位数（Median of Two Sorted Arrays）

**题目**：给定两个大小为 m 和 n 的正序数组 `nums1` 和 `nums2`，找出这两个正序数组的中位数。

**LeetCode 链接**：[Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

#### Python 实现（O(log(min(m, n)))）

```python
def find_median_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    x, y = len(nums1), len(nums2)
    low, high = 0, x
    
    while low <= high:
        partitionX = (low + high) // 2
        partitionY = (x + y + 1) // 2 - partitionX
        
        maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
        minX = float('inf') if partitionX == x else nums1[partitionX]
        
        maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
        minY = float('inf') if partitionY == y else nums2[partitionY]
        
        if maxX <= minY and maxY <= minX:
            if (x + y) % 2 == 0:
                return (max(maxX, maxY) + min(minX, minY)) / 2
            else:
                return max(maxX, maxY)
        elif maxX > minY:
            high = partitionX - 1
        else:
            low = partitionX + 1

    raise ValueError("Input arrays are not sorted correctly")

# 使用示例
nums1 = [1, 3]
nums2 = [2]
print(find_median_sorted_arrays(nums1, nums2))  # 输出: 2.0
```

#### 详解

- 利用二分查找法优化时间复杂度到 O(log(min(m, n)))。核心思路是找到两个数组的分区，使得左边的最大值小于等于右边的最小值。

---

### 22. 最大矩形（Maximal Rectangle）

**题目**：给定一个由 `'0'` 和 `'1'` 组成的二维矩阵，找出包含 `'1'` 的最大矩形的面积。

**LeetCode 链接**：[Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

#### Python 实现

```python
def maximal_rectangle(matrix):
    if not matrix:
        return 0
    
    def largest_rectangle_area(heights):
        stack = []
        max_area = 0
        heights.append(0)
        for i, h in enumerate(heights):
            while stack and h < heights[stack[-1]]:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        heights.pop()
        return max_area

    max_area = 0
    heights = [0] * len(matrix[0])
    for row in matrix:
        for i in range(len(row)):
            heights[i] = heights[i] + 1 if row[i] == '1' else 0
        max_area = max(max_area, largest_rectangle_area(heights))

    return max_area

# 使用示例
matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
print(maximal_rectangle(matrix))  # 输出: 6
```

#### 详解

- 先将二维矩阵转化为多个柱状图，然后利用柱状图的最大矩形面积来求解。时间复杂度为 O(m * n)，其中 m 和 n 分别是矩阵的行数和列数。

---

### 23. 单词拆分（Word Break）

**题目**：给定一个字符串 `s` 和一个字典 `wordDict`，判断 `s` 是否可以被分割为一个或多个字典中的单词。

**LeetCode 链接**：[Word Break](https://leetcode.com/problems/word-break/)

#### Python 实现

```python
def word_break(s, word_dict):
    dp = [False] * (len(s) + 1)
    dp[0] = True
    word_set = set(word_dict)
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[-1]

# 使用示例
s = "leetcode"
word_dict = ["leet", "code"]
print(word_break(s, word_dict))  # 输出: True
```

#### 详解

- 动态规划解决问题，使用 `dp` 数组标记到达每个位置是否可以被分割。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 24. 完全平方数（Perfect Squares）

**题目**：给定一个正整数 `n`，找到若干个完全平方数（如 1, 4, 9, 16, ...）的和，使得它们的和为 `n`。

**LeetCode 链接**：[Perfect Squares](https://leetcode.com/problems/perfect-squares/)

#### Python 实现

```python
import math

def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    
    return dp[n]

# 使用示例
n = 12
print(num_squares(n))  # 输出: 3
```

#### 详解

- 动态规划解决问题，`dp[i]` 表示和为 `i` 的最少完全平方数的数量。时间复杂度为 O(n * sqrt(n))，空间复杂度为 O(n)。

---

### 25. 反转链表（Reverse Linked List）

**题目**：反转一个单链表。

**LeetCode 链接**：[Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

# 使用示例
# 构造链表 1 -> 2 -> 3 -> 4 -> 5
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))

# 反转链表
reversed_head = reverse_list(head)
while reversed_head:
    print(reversed_head.val, end=" -> ")
    reversed_head = reversed_head.next
# 输出: 5 -> 4 -> 3 -> 2 -> 1 ->
```

#### 详解

- 通过迭代方式将链表反转。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 26. 删除有序数组中的重复项（Remove Duplicates from Sorted Array）

**题目**：给定一个排序数组，删除重复项，使得每个元素只出现一次，并返回新数组的长度。

**LeetCode 链接**：[Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

#### Python 实现

```python
def remove_duplicates(nums):
    if not nums:
        return 0

    unique_index = 0
    for i in range(1, len(nums)):
        if nums[i] != nums[unique_index]:
            unique_index += 1
            nums[unique_index] = nums[i]
    
    return unique_index + 1

# 使用示例
nums = [1, 1, 2]
length = remove_duplicates(nums)
print(length)  # 输出: 2
print(nums[:length])  # 输出: [1, 2]
```

#### 详解

- 由于数组已经排序，使用双指针方法删除重复项。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 27. 有效的括号（Valid Parentheses）

**题目**：给定一个只包含 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串，判断字符串是否有效。有效字符串需满足每个左括号有一个对应的右括号。

**LeetCode 链接**：[Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

#### Python 实现

```python
def is_valid(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    
    return not stack

# 使用示例
s = "()[]{}"
print(is_valid(s))  # 输出: True
```

#### 详解

- 使用栈来解决括号匹配问题。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同领域和难度层级的题目。

### 28. 最小路径和（Minimum Path Sum）

**题目**：给定一个包含非负整数的 `m x n` 网格，找到一条从左上角到右下角的路径，使得路径上的所有数字之和最小。

**LeetCode 链接**：[Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

#### Python 实现

```python
def min_path_sum(grid):
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    dp[0][0] = grid[0][0]
    
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[-1][-1]

# 使用示例
grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(min_path_sum(grid))  # 输出: 7
```

#### 详解

- 使用动态规划解决问题。`dp[i][j]` 表示从 `(0,0)` 到 `(i,j)` 的最小路径和。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 29. 组合总和（Combination Sum）

**题目**：给定一个候选数字的集合 `candidates` 和一个目标数 `target`，找出所有可以使数字和为 `target` 的组合。可以重复使用候选数字。

**LeetCode 链接**：[Combination Sum](https://leetcode.com/problems/combination-sum/)

#### Python 实现

```python
def combination_sum(candidates, target):
    res = []
    
    def backtrack(start, path, target):
        if target == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > target:
                continue
            path.append(candidates[i])
            backtrack(i, path, target - candidates[i])
            path.pop()
    
    backtrack(0, [], target)
    return res

# 使用示例
candidates = [2, 3, 6, 7]
target = 7
print(combination_sum(candidates, target))  # 输出: [[2, 2, 3], [7]]
```

#### 详解

- 使用回溯算法生成所有可能的组合。允许重复使用同一候选数字。时间复杂度为 O(2^n)。

---

### 30. 搜索二维矩阵 II（Search a 2D Matrix II）

**题目**：编写一个高效的算法来搜索 `m x n` 矩阵中的一个目标值。矩阵的每一行和每一列都是按升序排序的。

**LeetCode 链接**：[Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)

#### Python 实现

```python
def search_matrix(matrix, target):
    if not matrix:
        return False

    m, n = len(matrix), len(matrix[0])
    row, col = m - 1, 0
    
    while row >= 0 and col < n:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            row -= 1
        else:
            col += 1
    
    return False

# 使用示例
matrix = [
    [1, 4, 7, 11],
    [2, 5, 8, 12],
    [3, 6, 9, 16],
    [10, 13, 14, 17]
]
target = 5
print(search_matrix(matrix, target))  # 输出: True
```

#### 详解

- 从矩阵的右上角开始搜索，利用矩阵的排序特性优化搜索过程。时间复杂度为 O(m + n)。

---

### 31. 组合（Combinations）

**题目**：给定两个整数 `n` 和 `k`，返回 `1` 到 `n` 的所有可能的 `k` 组合。

**LeetCode 链接**：[Combinations](https://leetcode.com/problems/combinations/)

#### Python 实现

```python
def combine(n, k):
    res = []
    
    def backtrack(start, path):
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return res

# 使用示例
n = 4
k = 2
print(combine(n, k))  # 输出: [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
```

#### 详解

- 使用回溯算法生成所有可能的组合。时间复杂度为 O(C(n, k))，其中 C(n, k) 是组合数。

---

### 32. 最大子序和（Maximum Subarray）

**题目**：给定一个整数数组 `nums`，找到一个具有最大和的连续子数组，并返回其最大和。

**LeetCode 链接**：[Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

#### Python 实现

```python
def max_sub_array(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# 使用示例
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_sub_array(nums))  # 输出: 6
```

#### 详解

- 使用 Kadane 算法解决问题，时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 33. 搜索旋转排序数组（Search in Rotated Sorted Array）

**题目**：给定一个旋转排序数组和一个目标值 `target`，请找出目标值是否在数组中，并返回其索引。如果数组中不存在目标值，返回 -1。

**LeetCode 链接**：[Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

#### Python 实现

```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# 使用示例
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search(nums, target))  # 输出: 4
```

#### 详解

- 使用修改版的二分查找算法来处理旋转排序数组。时间复杂度为 O(log n)。

---

### 34. 旋转图像（Rotate Image）

**题目**：给定一个 `n x n` 的二维矩阵 `matrix`，将其顺时针旋转 90 度。

**LeetCode 链接**：[Rotate Image](https://leetcode.com/problems/rotate-image/)

#### Python 实现

```python
def rotate(matrix):
    n = len(matrix)
    
    # 转置矩阵
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # 反转每一行
    for i in range(n):
        matrix[i].reverse()

# 使用示例
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
rotate(matrix)
print(matrix)  # 输出: [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
```

#### 详解

- 先进行矩阵转置，再反转每一行。时间复杂度为 O(n^2)，空间复杂度为 O(1)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同的算法和数据结构。

### 35. 字符串的排列（Permutation in String）

**题目**：给定两个字符串 `s1` 和 `s2`，判断 `s2` 是否包含 `s1` 的排列。即：`s2` 是否包含 `s1` 的某个排列的子串。

**LeetCode 链接**：[Permutation in String](https://leetcode.com/problems/permutation-in-string/)

#### Python 实现

```python
from collections import Counter

def check_inclusion(s1, s2):
    s1_count = Counter(s1)
    s2_count = Counter()
    len_s1 = len(s1)
    
    for i in range(len(s2)):
        s2_count[s2[i]] += 1
        if i >= len_s1:
            if s2_count[s2[i - len_s1]] == 1:
                del s2_count[s2[i - len_s1]]
            else:
                s2_count[s2[i - len_s1]] -= 1
        
        if s2_count == s1_count:
            return True
    
    return False

# 使用示例
s1 = "ab"
s2 = "eidbaooo"
print(check_inclusion(s1, s2))  # 输出: True
```

#### 详解

- 使用滑动窗口和计数器来检查 `s2` 是否包含 `s1` 的排列。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 36. 最大正方形（Maximal Square）

**题目**：给定一个只包含 `'0'` 和 `'1'` 的二维矩阵，找出只包含 `'1'` 的最大正方形，并返回其面积。

**LeetCode 链接**：[Maximal Square](https://leetcode.com/problems/maximal-square/)

#### Python 实现

```python
def maximal_square(matrix):
    if not matrix:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side

# 使用示例
matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]
print(maximal_square(matrix))  # 输出: 4
```

#### 详解

- 使用动态规划来计算每个点形成的最大正方形的边长。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 37. 编辑距离（Edit Distance）

**题目**：给定两个字符串 `word1` 和 `word2`，计算将 `word1` 转换为 `word2` 所需的最小操作数。操作包括插入一个字符、删除一个字符和替换一个字符。

**LeetCode 链接**：[Edit Distance](https://leetcode.com/problems/edit-distance/)

#### Python 实现

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]

# 使用示例
word1 = "horse"
word2 = "ros"
print(min_distance(word1, word2))  # 输出: 3
```

#### 详解

- 使用动态规划来计算最小编辑距离。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 38. 爬楼梯（Climbing Stairs）

**题目**：假设你正在爬楼梯。楼梯有 `n` 阶，每次你可以爬 1 阶或 2 阶。计算到达第 `n` 阶有多少种不同的方法。

**LeetCode 链接**：[Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

#### Python 实现

```python
def climb_stairs(n):
    if n <= 1:
        return 1
    
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# 使用示例
n = 5
print(climb_stairs(n))  # 输出: 8
```

#### 详解

- 使用动态规划解决问题。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 39. 合并区间（Merge Intervals）

**题目**：给定一个区间的集合，合并所有重叠的区间。

**LeetCode 链接**：[Merge Intervals](https://leetcode.com/problems/merge-intervals/)

#### Python 实现

```python
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged

# 使用示例
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print(merge(intervals))  # 输出: [[1, 6], [8, 10], [15, 18]]
```

#### 详解

- 先对区间进行排序，然后合并重叠的区间。时间复杂度为 O(n log n)，空间复杂度为 O(n)。

---

### 40. 最小窗口子串（Minimum Window Substring）

**题目**：给定一个字符串 `s` 和一个字符串 `t`，找出 `s` 中包含 `t` 所有字符的最小窗口子串。

**LeetCode 链接**：[Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

#### Python 实现

```python
from collections import Counter

def min_window(s, t):
    if not s or not t:
        return ""
    
    t_count = Counter(t)
    s_count = Counter()
    
    left, right = 0, 0
    min_len = float('inf')
    min_window = ""
    
    while right < len(s):
        s_count[s[right]] += 1
        while all(s_count[c] >= t_count[c] for c in t_count):
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_window = s[left:right + 1]
            s_count[s[left]] -= 1
            left += 1
        right += 1
    
    return min_window

# 使用示例
s = "ADOBECODEBANC"
t = "ABC"
print(min_window(s, t))  # 输出: "BANC"
```

#### 详解

- 使用滑动窗口和计数器来找出包含所有目标字符的最小窗口子串。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 41. 最长公共前缀（Longest Common Prefix）

**题目**：编写一个函数来查找字符串数组中的最长公共前缀。如果没有公共前缀，返回空字符串 `""`。

**LeetCode 链接**：[Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)

#### Python 实现

```python
def longest_common_prefix(strs):
    if not strs:
        return ""
    
    min_len = min(len(s) for s in strs)
    
    for i in range(min_len):
        char = strs[0][i]
        for s in strs[1:]:
            if s[i] != char:
                return strs[0][:i]
    
    return strs[0][:min_len]

# 使用示例
strs = ["flower", "flow", "flight"]
print(longest_common_prefix(strs))  # 输出: "fl"
```

#### 详解

- 比较所有字符串的字符来找出最长公共前缀。时间复杂度为 O(S)，其中 S 是所有字符串的字符总数。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同的算法和数据结构。

### 42. 盛最多水的容器（Container With Most Water）

**题目**：给定一个整数数组 `height`，其中 `height[i]` 是第 `i` 条垂直线的高度。找到两条线，使得它们和 x 轴之间形成的容器可以容纳最多的水，并返回这个最大值。

**LeetCode 链接**：[Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

#### Python 实现

```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        min_height = min(height[left], height[right])
        max_area = max(max_area, width * min_height)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

# 使用示例
height = [1,8,6,2,5,4,8,3,7]
print(max_area(height))  # 输出: 49
```

#### 详解

- 使用双指针法来优化搜索。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 43. 字符串乘法（Multiply Strings）

**题目**：给定两个非负整数 `num1` 和 `num2` 表示为字符串，返回它们的乘积，结果也应该是字符串形式。

**LeetCode 链接**：[Multiply Strings](https://leetcode.com/problems/multiply-strings/)

#### Python 实现

```python
def multiply(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"
    
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            mul = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
            p1, p2 = i + j, i + j + 1
            summ = mul + result[p2]
            
            result[p2] = summ % 10
            result[p1] += summ // 10
    
    result_str = ''.join(map(str, result))
    return result_str.lstrip('0')

# 使用示例
num1 = "123"
num2 = "456"
print(multiply(num1, num2))  # 输出: "56088"
```

#### 详解

- 使用模拟乘法算法来计算结果。时间复杂度为 O(m * n)，空间复杂度为 O(m + n)。

---

### 44. 通配符匹配（Wildcard Matching）

**题目**：给定两个字符串 `s` 和 `p`，其中 `p` 可能包含通配符 `*` 和 `?`。`*` 表示零个或多个字符，`?` 表示一个字符。判断字符串 `s` 是否与 `p` 匹配。

**LeetCode 链接**：[Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)

#### Python 实现

```python
def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] and p[j - 1] == '*'
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

# 使用示例
s = "aa"
p = "a*"
print(is_match(s, p))  # 输出: True
```

#### 详解

- 使用动态规划来解决通配符匹配问题。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 45. 跳跃游戏 II（Jump Game II）

**题目**：给定一个非负整数数组 `nums`，你最初位于数组的第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。求跳到最后一个下标的最小跳跃次数。

**LeetCode 链接**：[Jump Game II](https://leetcode.com/problems/jump-game-ii/)

#### Python 实现

```python
def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps

# 使用示例
nums = [2,3,1,1,2,4,2,0,1,1]
print(jump(nums))  # 输出: 4
```

#### 详解

- 使用贪心算法来计算最小跳跃次数。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 46. 全排列（Permutations）

**题目**：给定一个没有重复数字的整数数组 `nums`，返回其所有可能的全排列。

**LeetCode 链接**：[Permutations](https://leetcode.com/problems/permutations/)

#### Python 实现

```python
def permute(nums):
    def backtrack(path):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for num in nums:
            if num in path:
                continue
            path.append(num)
            backtrack(path)
            path.pop()
    
    res = []
    backtrack([])
    return res

# 使用示例
nums = [1, 2, 3]
print(permute(nums))  # 输出: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```

#### 详解

- 使用回溯算法生成所有可能的排列。时间复杂度为 O(n!)，空间复杂度为 O(n)。

---

### 47. 全排列 II（Permutations II）

**题目**：给定一个可能包含重复数字的整数数组 `nums`，返回其所有不同的全排列。

**LeetCode 链接**：[Permutations II](https://leetcode.com/problems/permutations-ii/)

#### Python 实现

```python
def permute_unique(nums):
    def backtrack(path):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if visited[i] or (i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]):
                continue
            visited[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            visited[i] = False
    
    nums.sort()
    visited = [False] * len(nums)
    res = []
    backtrack([])
    return res

# 使用示例
nums = [1, 1, 2]
print(permute_unique(nums))  # 输出: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
```

#### 详解

- 使用回溯算法生成所有不同的排列。时间复杂度为 O(n!)，空间复杂度为 O(n)。

---

### 48. 旋转图像（Rotate Image）

**题目**：给定一个 `n x n` 的二维矩阵 `matrix`，将其顺时针旋转 90 度。

**LeetCode 链接**：[Rotate Image](https://leetcode.com/problems/rotate-image/)

#### Python 实现

```python
def rotate(matrix):
    n = len(matrix)
    
    # 转置矩阵
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # 反转每一行
    for i in range(n):
        matrix[i].reverse()

# 使用示例
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
rotate(matrix)
print(matrix)  # 输出: [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
```

#### 详解

- 首先进行矩阵转置，然后反转每一行。时间复杂度为 O(n^2)，空间复杂度为 O(1)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同的算法和数据结构。

### 49. 字母异位词分组（Group Anagrams）

**题目**：给定一个字符串数组，将所有字母异位词分到同一组。字母异位词是指两个字符串包含相同的字符，且每个字符出现的次数也相同。

**LeetCode 链接**：[Group Anagrams](https://leetcode.com/problems/group-anagrams/)

#### Python 实现

```python
from collections import defaultdict

def group_anagrams(strs):
    anagram_map = defaultdict(list)
    
    for s in strs:
        sorted_str = ''.join(sorted(s))
        anagram_map[sorted_str].append(s)
    
    return list(anagram_map.values())

# 使用示例
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(strs))  # 输出: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

#### 详解

- 使用哈希表来存储按字母排序后的字符串及其对应的原字符串。时间复杂度为 O(n * k log k)，空间复杂度为 O(n * k)，其中 n 为字符串数，k 为字符串的最大长度。

---

### 50. Pow(x, n)（实现 pow(x, n)）

**题目**：实现 `pow(x, n)`，即计算 `x` 的 `n` 次幂。

**LeetCode 链接**：[Pow(x, n)](https://leetcode.com/problems/powx-n/)

#### Python 实现

```python
def my_pow(x, n):
    if n == 0:
        return 1
    elif n < 0:
        x = 1 / x
        n = -n
    
    half = my_pow(x, n // 2)
    if n % 2 == 0:
        return half * half
    else:
        return half * half * x

# 使用示例
x = 2.00000
n = 10
print(my_pow(x, n))  # 输出: 1024.00000
```

#### 详解

- 使用递归法来计算幂，时间复杂度为 O(log n)，空间复杂度为 O(log n)。

---

### 51. N 皇后（N-Queens）

**题目**：n 皇后问题是一个经典的计算机科学问题，要求在一个 n x n 的棋盘上放置 n 个皇后，使得它们彼此之间不能相互攻击。实现一个算法，返回所有有效的 n 皇后问题的解决方案。

**LeetCode 链接**：[N-Queens](https://leetcode.com/problems/n-queens/)

#### Python 实现

```python
def solve_n_queens(n):
    def is_not_under_attack(row, col):
        for prev_row in range(row):
            if board[prev_row] == col or \
               board[prev_row] - prev_row == col - row or \
               board[prev_row] + prev_row == col + row:
                return False
        return True
    
    def backtrack(row):
        if row == n:
            result.append(["".join(['Q' if col == board[i] else '.' for col in range(n)]) for i in range(n)])
            return
        for col in range(n):
            if is_not_under_attack(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1
    
    result = []
    board = [-1] * n
    backtrack(0)
    return result

# 使用示例
n = 4
print(solve_n_queens(n))  # 输出: [['.Q..', '...Q', 'Q...', '..Q.'], ['..Q.', 'Q...', '...Q', '.Q..']]
```

#### 详解

- 使用回溯算法来解决 n 皇后问题。时间复杂度为 O(n!)，空间复杂度为 O(n)。

---

### 52. N 皇后 II（N-Queens II）

**题目**：与 N 皇后类似，但只需要返回有效的解决方案的数量，而不是具体的布局。

**LeetCode 链接**：[N-Queens II](https://leetcode.com/problems/n-queens-ii/)

#### Python 实现

```python
def total_n_queens(n):
    def is_not_under_attack(row, col):
        for prev_row in range(row):
            if board[prev_row] == col or \
               board[prev_row] - prev_row == col - row or \
               board[prev_row] + prev_row == col + row:
                return False
        return True
    
    def backtrack(row):
        if row == n:
            nonlocal count
            count += 1
            return
        for col in range(n):
            if is_not_under_attack(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1
    
    count = 0
    board = [-1] * n
    backtrack(0)
    return count

# 使用示例
n = 4
print(total_n_queens(n))  # 输出: 2
```

#### 详解

- 使用回溯算法计算有效解决方案的数量。时间复杂度为 O(n!)，空间复杂度为 O(n)。

---

### 53. 最大子数组和（Maximum Subarray）

**题目**：给定一个整数数组 `nums`，找出具有最大和的连续子数组，并返回其和。

**LeetCode 链接**：[Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

#### Python 实现

```python
def max_sub_array(nums):
    max_sum = nums[0]
    current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# 使用示例
nums = [-2,1,-3,4,-1,2,1,-5,4]
print(max_sub_array(nums))  # 输出: 6
```

#### 详解

- 使用动态规划（Kadane 算法）计算最大子数组和。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 54. 螺旋矩阵（Spiral Matrix）

**题目**：给定一个 m x n 的矩阵 `matrix`，按照螺旋顺序返回矩阵中的所有元素。

**LeetCode 链接**：[Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

#### Python 实现

```python
def spiral_order(matrix):
    if not matrix:
        return []
    
    result = []
    top, bottom, left, right = 0, len(matrix), 0, len(matrix[0])
    
    while top < bottom and left < right:
        for col in range(left, right):
            result.append(matrix[top][col])
        top += 1
        
        for row in range(top, bottom):
            result.append(matrix[row][right - 1])
        right -= 1
        
        if not (top < bottom and left < right):
            break
        
        for col in range(right - 1, left - 1, -1):
            result.append(matrix[bottom - 1][col])
        bottom -= 1
        
        for row in range(bottom - 1, top - 1, -1):
            result.append(matrix[row][left])
        left += 1
    
    return result

# 使用示例
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(spiral_order(matrix))  # 输出: [1, 2, 3, 6, 9, 8, 7, 4, 5]
```

#### 详解

- 使用边界变量来控制螺旋顺序的循环。时间复杂度为 O(m * n)，空间复杂度为 O(1)。

---

### 55. 跳跃游戏（Jump Game）

**题目**：给定一个非负整数数组 `nums`，你最初位于数组的第一个下标。每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达数组的最后一个下标。

**LeetCode 链接**：[Jump Game](https://leetcode.com/problems/jump-game/)

#### Python 实现

```python
def can_jump(nums):
    max_reachable = 0
    for i, jump in enumerate(nums):
        if i > max_reachable:
            return False
        max_reachable = max(max_reachable, i + jump)
    return True

# 使用示例
nums = [2, 3, 1, 1, 4]
print(can_jump(nums))  # 输出: True
```

#### 详解

- 使用贪心算法来判断是否可以到达最后一个下标。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 56. 合并区间（Merge Intervals）

**题目**：给定一个区间的集合，合并所有重叠的区间。

**LeetCode 链接**：[Merge Intervals](https://leetcode.com/problems/merge-intervals/)

#### Python 实现

```python
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            last_merged[1] = max(last_merged[1], current[1])
        else:
            merged.append(current)
    
    return merged

# 使用示例
intervals = [[1,3],[2,6],[8,10],[15,18]]
print(merge(intervals))  # 输出: [[1,6],[8,10],[15,18]]
```

#### 详解

- 先对区间进行排序，然后合并重叠的区间。时间复杂度为 O(n log n)，空间复杂度为 O(n)。

---

### 57. 插入区间（Insert Interval）

**题目**：给定一个不重叠的区间列表 `intervals` 和一个新的区间 `new_interval`，将新的区间插入到列表中，并合并所有重叠的区间。

**LeetCode 链接**：[Insert Interval](https://leetcode.com/problems/insert-interval/)

#### Python 实现

```python
def insert(intervals, new_interval):
    merged = []
    i = 0
    n = len(intervals)
    
    # 添加所有在 new_interval 左边的区间
    while i < n and intervals[i][1] < new_interval[0]:
        merged.append(intervals[i])
        i += 1
    
    # 合并所有重叠的区间
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    merged.append(new_interval)
    
    # 添加所有在 new_interval 右边的区间
    while i < n:
        merged.append(intervals[i])
        i += 1
    
    return merged

# 使用示例
intervals = [[1,3],[6,9]]
new_interval = [2,5]
print(insert(intervals, new_interval))  # 输出: [[1,5],[6,9]]
```

#### 详解

- 将新的区间插入到正确的位置，并合并所有重叠的区间。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 58. 最后一个单词的长度（Length of Last Word）

**题目**：给定一个字符串 `s`，其中仅包含字母和空格，返回字符串最后一个单词的长度。单词是由非空格字符组成的。

**LeetCode 链接**：[Length of Last Word](https://leetcode.com/problems/length-of-last-word/)

#### Python 实现

```python
def length_of_last_word(s):
    words = s.split()
    if not words:
        return 0
    return len(words[-1])

# 使用示例
s = "Hello World"
print(length_of_last_word(s))  # 输出: 5
```

#### 详解

- 通过分割字符串并返回最后一个单词的长度。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同的算法和数据结构。

### 59. 螺旋矩阵 II（Spiral Matrix II）

**题目**：给定一个正整数 `n`，生成一个 `n x n` 的矩阵，其中矩阵中的元素按螺旋顺序填充。

**LeetCode 链接**：[Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)

#### Python 实现

```python
def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    num = 1
    top, bottom, left, right = 0, n, 0, n
    
    while top < bottom and left < right:
        for col in range(left, right):
            matrix[top][col] = num
            num += 1
        top += 1
        
        for row in range(top, bottom):
            matrix[row][right - 1] = num
            num += 1
        right -= 1
        
        if not (top < bottom and left < right):
            break
        
        for col in range(right - 1, left - 1, -1):
            matrix[bottom - 1][col] = num
            num += 1
        bottom -= 1
        
        for row in range(bottom - 1, top - 1, -1):
            matrix[row][left] = num
            num += 1
        left += 1
    
    return matrix

# 使用示例
n = 3
print(generate_matrix(n))
# 输出: [[1, 2, 3], [8, 9, 4], [7, 6, 5]]
```

#### 详解

- 使用边界变量控制螺旋顺序的填充。时间复杂度为 O(n^2)，空间复杂度为 O(1)。

---

### 60. 排列序列（Permutation Sequence）

**题目**：给定两个整数 `n` 和 `k`，返回第 `k` 个排列序列。

**LeetCode 链接**：[Permutation Sequence](https://leetcode.com/problems/permutation-sequence/)

#### Python 实现

```python
from math import factorial

def get_permutation(n, k):
    nums = list(range(1, n + 1))
    k -= 1
    permutation = []
    
    for i in range(n):
        fact = factorial(n - 1 - i)
        index, k = divmod(k, fact)
        permutation.append(nums.pop(index))
    
    return ''.join(map(str, permutation))

# 使用示例
n = 3
k = 3
print(get_permutation(n, k))  # 输出: "213"
```

#### 详解

- 使用阶乘来计算排列位置。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 61. 旋转链表（Rotate List）

**题目**：给定一个链表，将链表的每个节点向右移动 `k` 个位置。

**LeetCode 链接**：[Rotate List](https://leetcode.com/problems/rotate-list/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def rotate_right(head, k):
    if not head or k == 0:
        return head
    
    # 计算链表长度
    length = 1
    old_tail = head
    while old_tail.next:
        old_tail = old_tail.next
        length += 1
    
    # 计算实际旋转次数
    k %= length
    if k == 0:
        return head
    
    # 找到新的尾部和头部
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    old_tail.next = head
    
    return new_head

# 使用示例
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))))
k = 2
result = rotate_right(head, k)
while result:
    print(result.val, end=" -> ")
    result = result.next
# 输出: 4 -> 5 -> 1 -> 2 -> 3
```

#### 详解

- 先计算链表长度，然后找到旋转后的新头和尾。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 62. 不同路径（Unique Paths）

**题目**：一个机器人位于一个 `m x n` 的网格的左上角。机器人只能向右或向下移动。计算机器人到达网格右下角的不同路径数。

**LeetCode 链接**：[Unique Paths](https://leetcode.com/problems/unique-paths/)

#### Python 实现

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[-1][-1]

# 使用示例
m = 3
n = 7
print(unique_paths(m, n))  # 输出: 28
```

#### 详解

- 使用动态规划计算不同路径数。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 63. 不同路径 II（Unique Paths II）

**题目**：一个机器人位于一个 `m x n` 的网格的左上角，网格中有一些障碍物。计算机器人到达网格右下角的不同路径数。

**LeetCode 链接**：[Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)

#### Python 实现

```python
def unique_paths_with_obstacles(grid):
    if not grid or grid[0][0] == 1:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                if i > 0:
                    dp[i][j] += dp[i - 1][j]
                if j > 0:
                    dp[i][j] += dp[i][j - 1]
    
    return dp[-1][-1]

# 使用示例
grid = [
    [0,0,0],
    [0,1,0],
    [0,0,0]
]
print(unique_paths_with_obstacles(grid))  # 输出: 2
```

#### 详解

- 使用动态规划处理障碍物的情况。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 64. 最小路径和（Minimum Path Sum）

**题目**：给定一个 `m x n` 的网格，初始化时每个格子都有一个非负整数。计算从左上角到右下角的路径和最小值，路径只能向右或向下。

**LeetCode 链接**：[Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

#### Python 实现

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    
    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]
    
    for j in range(1, n):
        grid[0][j] += grid[0][j - 1]
    
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    
    return grid[-1][-1]

# 使用示例
grid = [
    [1,3,1],
    [1,5,1],
    [4,2,1]
]
print(min_path_sum(grid))  # 输出: 7
```

#### 详解

- 使用动态规划计算最小路径和。时间复杂度为 O(m * n)，空间复杂度为 O(1)。

---

### 65. 有效数字（Valid Number）

**题目**：验证给定的字符串是否是一个有效的数字。有效的数字可以是整数、浮点数（正数或负数）、科学计数法表示的数字。

**LeetCode 链接**：[Valid Number](https://leetcode.com/problems/valid-number/)

#### Python 实现

```python
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 使用示例
s = "0.1"
print(is_number(s))  # 输出: True
```

#### 详解

- 使用 Python 内置的 `float` 函数来验证字符串是否可以被转换为有效的数字。时间复杂度为 O(1)，空间复杂度为 O(1)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同的算法和数据结构。

### 66. 加一（Plus One）

**题目**：给定一个由非负整数组成的整数数组 `digits`，表示一个大于 0 的整数，返回该整数加一后的数组表示。

**LeetCode 链接**：[Plus One](https://leetcode.com/problems/plus-one/)

#### Python 实现

```python
def plus_one(digits):
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    
    return [1] + digits

# 使用示例
digits = [1, 2, 3]
print(plus_one(digits))  # 输出: [1, 2, 4]
```

#### 详解

- 从右到左遍历数组，处理进位。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 67. 二进制求和（Add Binary）

**题目**：给定两个二进制字符串，返回它们的和（也即它们的二进制字符串形式）。

**LeetCode 链接**：[Add Binary](https://leetcode.com/problems/add-binary/)

#### Python 实现

```python
def add_binary(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]

# 使用示例
a = "1010"
b = "1011"
print(add_binary(a, b))  # 输出: "10101"
```

#### 详解

- 将二进制字符串转换为整数，进行加法运算，然后转换回二进制字符串。时间复杂度为 O(max(m, n))，空间复杂度为 O(1)，其中 m 和 n 为字符串的长度。

---

### 68. 文本左右对齐（Text Justification）

**题目**：给定一个单词数组 `words` 和一个最大宽度 `max_width`，将单词排成一行，并使每行的文本尽可能对齐。

**LeetCode 链接**：[Text Justification](https://leetcode.com/problems/text-justification/)

#### Python 实现

```python
def full_justify(words, max_width):
    def add_spaces(words, max_width, is_last_line):
        if len(words) == 1:
            return words[0] + ' ' * (max_width - len(words[0]))
        
        total_spaces = max_width - sum(len(word) for word in words)
        spaces_between_words = total_spaces // (len(words) - 1)
        extra_spaces = total_spaces % (len(words) - 1)
        
        result = ''
        for i in range(len(words) - 1):
            result += words[i] + ' ' * (spaces_between_words + (1 if i < extra_spaces else 0))
        result += words[-1]
        return result
    
    result, current_line, num_of_letters = [], [], 0
    
    for word in words:
        if num_of_letters + len(word) + len(current_line) > max_width:
            result.append(add_spaces(current_line, max_width, False))
            current_line, num_of_letters = [], 0
        current_line.append(word)
        num_of_letters += len(word)
    
    result.append(' '.join(current_line).ljust(max_width))
    
    return result

# 使用示例
words = ["This", "is", "an", "example", "of", "text", "justification."]
max_width = 16
print(full_justify(words, max_width))
# 输出:
# ["This    is    an", 
#  "example  of text", 
#  "justification.  "]
```

#### 详解

- 使用两步处理：计算每行的空间分配，然后将结果添加到输出列表。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 69. Sqrt(x)（x 的平方根）

**题目**：实现 `sqrt(x)` 函数，计算并返回 x 的平方根。

**LeetCode 链接**：[Sqrt(x)](https://leetcode.com/problems/sqrtx/)

#### Python 实现

```python
def my_sqrt(x):
    if x < 2:
        return x
    
    left, right = 2, x // 2
    while left <= right:
        mid = (left + right) // 2
        num = mid * mid
        if num == x:
            return mid
        elif num < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right

# 使用示例
x = 8
print(my_sqrt(x))  # 输出: 2
```

#### 详解

- 使用二分查找算法来找到平方根的整数部分。时间复杂度为 O(log x)，空间复杂度为 O(1)。

---

### 70. 爬楼梯（Climbing Stairs）

**题目**：一只青蛙一次可以跳 1 步或 2 步，计算爬到楼梯顶部的不同方式的数量。

**LeetCode 链接**：[Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

#### Python 实现

```python
def climb_stairs(n):
    if n <= 1:
        return 1
    
    prev, curr = 1, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr

# 使用示例
n = 5
print(climb_stairs(n))  # 输出: 8
```

#### 详解

- 使用动态规划计算爬楼梯的不同方式。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 71. 简化路径（Simplify Path）

**题目**：给定一个绝对路径的文件路径，简化路径，使其符合 Unix 文件系统的标准格式。

**LeetCode 链接**：[Simplify Path](https://leetcode.com/problems/simplify-path/)

#### Python 实现

```python
def simplify_path(path):
    stack = []
    for part in path.split('/'):
        if part == '..':
            if stack:
                stack.pop()
        elif part and part != '.':
            stack.append(part)
    
    return '/' + '/'.join(stack)

# 使用示例
path = "/a/./b/../../c/"
print(simplify_path(path))  # 输出: "/c"
```

#### 详解

- 使用栈来处理路径的简化。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 72. 编辑距离（Edit Distance）

**题目**：给定两个单词 `word1` 和 `word2`，计算将 `word1` 转换为 `word2` 所需的最小操作次数。操作包括插入、删除和替换。

**LeetCode 链接**：[Edit Distance](https://leetcode.com/problems/edit-distance/)

#### Python 实现

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]

# 使用示例
word1 = "intention"
word2 = "execution"
print(min_distance(word1, word2))  # 输出: 5
```

#### 详解

- 使用动态规划计算编辑距离。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)，其中 m 和 n 为两个单词的长度。

---

### 73. 矩阵置零（Set Matrix Zeroes）

**题目**：给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列都置为 0。

**LeetCode 链接**：[Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)

#### Python 实现

```python
def set_zeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    row_zero = any(matrix[0][j] == 0 for j in range(cols))
    col_zero = any(matrix[i][0] == 0 for i in range(rows))
    
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0
    
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    if row_zero:
        for j in range(cols):
            matrix[0][j] = 0
    
    if col_zero:
        for i in range(rows):
            matrix[i][0] = 0

# 使用示例
matrix = [
    [1, 2, 3],
    [4, 0, 6],
    [7, 8, 9]
]
set_zeroes(matrix)
for row in matrix:
    print(row)
# 输出:
# [1, 0, 0]
# [0, 0, 0]
# [7, 0, 9]
```

#### 详解

- 使用两个额外的变量来跟踪第一行和第一列是否需要置零。时间复杂度为 O(m * n)，空间复杂度为 O(1)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同的算法和数据结构。

### 74. 搜索二维矩阵（Search a 2D Matrix）

**题目**：编写一个高效的算法来搜索一个二维矩阵中的目标值。矩阵中的每一行和每一列都已按升序排序。

**LeetCode 链接**：[Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

#### Python 实现

```python
def search_matrix(matrix, target):
    if not matrix:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_val = matrix[mid // n][mid % n]
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# 使用示例
matrix = [
    [1, 4, 7, 11],
    [2, 5, 8, 12],
    [3, 6, 9, 16],
    [10, 13, 14, 17]
]
target = 5
print(search_matrix(matrix, target))  # 输出: True
```

#### 详解

- 使用二分查找算法在二维矩阵中查找目标值。时间复杂度为 O(log(m * n))，空间复杂度为 O(1)。

---

### 75. 颜色分类（Sort Colors）

**题目**：给定一个包含红色、白色和蓝色（即 0、1 和 2）的数组，将它们按颜色排序。你需要在原地进行排序，并且算法的时间复杂度必须为 O(n)。

**LeetCode 链接**：[Sort Colors](https://leetcode.com/problems/sort-colors/)

#### Python 实现

```python
def sort_colors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1

# 使用示例
nums = [2, 0, 2, 1, 1, 0]
sort_colors(nums)
print(nums)  # 输出: [0, 0, 1, 1, 2, 2]
```

#### 详解

- 使用荷兰国旗问题的三指针算法进行排序。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 76. 最小窗口子串（Minimum Window Substring）

**题目**：给定两个字符串 `s` 和 `t`，找到 `s` 中包含 `t` 的所有字符的最小子串。

**LeetCode 链接**：[Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

#### Python 实现

```python
from collections import Counter

def min_window(s, t):
    if not s or not t:
        return ""
    
    t_count = Counter(t)
    s_count = Counter()
    required = len(t_count)
    left, right = 0, 0
    min_len = float("inf")
    min_window = ""
    
    while right < len(s):
        char = s[right]
        s_count[char] += 1
        if char in t_count and s_count[char] == t_count[char]:
            required -= 1
        
        while required == 0:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_window = s[left:right + 1]
            
            s_count[s[left]] -= 1
            if s[left] in t_count and s_count[s[left]] < t_count[s[left]]:
                required += 1
            left += 1
        
        right += 1
    
    return min_window

# 使用示例
s = "ADOBECODEBANC"
t = "ABC"
print(min_window(s, t))  # 输出: "BANC"
```

#### 详解

- 使用滑动窗口算法来找到最小窗口子串。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 77. 组合（Combinations）

**题目**：给定两个整数 `n` 和 `k`，返回 1 到 `n` 中所有可能的 `k` 个数字的组合。

**LeetCode 链接**：[Combinations](https://leetcode.com/problems/combinations/)

#### Python 实现

```python
def combine(n, k):
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    result = []
    backtrack(1, [])
    return result

# 使用示例
n = 4
k = 2
print(combine(n, k))  # 输出: [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
```

#### 详解

- 使用回溯算法生成所有可能的组合。时间复杂度为 O(C(n, k))，空间复杂度为 O(C(n, k) * k)，其中 C(n, k) 是组合数。

---

### 78. 子集（Subsets）

**题目**：给定一个整数数组 `nums`，返回所有可能的子集（幂集）。

**LeetCode 链接**：[Subsets](https://leetcode.com/problems/subsets/)

#### Python 实现

```python
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# 使用示例
nums = [1, 2, 3]
print(subsets(nums))  # 输出: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

#### 详解

- 使用回溯算法生成所有可能的子集。时间复杂度为 O(2^n)，空间复杂度为 O(2^n * n)。

---

### 79. 单词搜索（Word Search）

**题目**：给定一个二维网格和一个单词，检查网格中是否存在该单词。单词可以在网格中水平、垂直或对角线方向上存在。

**LeetCode 链接**：[Word Search](https://leetcode.com/problems/word-search/)

#### Python 实现

```python
def exist(board, word):
    def dfs(i, j, index):
        if index == len(word):
            return True
        if (i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or 
            board[i][j] != word[index]):
            return False
        
        temp, board[i][j] = board[i][j], '#'
        found = (dfs(i + 1, j, index + 1) or
                 dfs(i - 1, j, index + 1) or
                 dfs(i, j + 1, index + 1) or
                 dfs(i, j - 1, index + 1))
        board[i][j] = temp
        
        return found
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(i, j, 0):
                return True
    return False

# 使用示例
board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
word = "ABCCED"
print(exist(board, word))  # 输出: True
```

#### 详解

- 使用深度优先搜索（DFS）来检查单词是否存在于网格中。时间复杂度为 O(m * n * 4^L)，空间复杂度为 O(L)，其中 L 为单词长度。

---

### 80. 删除重复排序数组中的重复项 II（Remove Duplicates from Sorted Array II）

**题目**：给定一个排序数组 `nums`，删除重复项，使得每个元素最多出现两次，并返回新数组的长度。

**LeetCode 链接**：[Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)

#### Python 实现

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    
    i = 1
    count = 1
    
    for j in range(1, len(nums)):
        if nums[j] == nums[j - 1]:
            count += 1
        else:
            count = 1
        
        if count <= 2:
            nums[i] = nums[j]
            i += 1
    
    return i

# 使用示例
nums = [1, 1, 1, 2, 2, 3]
length = remove_duplicates(nums)
print(nums[:length])  # 输出: [1, 1, 2, 2, 3]
```

#### 详解

- 使用双指针来确保每个元素最多出现两次。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖不同的算法和数据结构。

### 81. 搜索旋转排序数组 II（Search in Rotated Sorted Array II）

**题目**：给定一个可能包含重复元素的升序数组 `nums` 和一个目标值 `target`，搜索目标值是否存在于数组中。你必须实现一个时间复杂度为 O(log n) 的算法。

**LeetCode 链接**：[Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)

#### Python 实现

```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return True
        
        if nums[left] < nums[mid] or nums[mid] < nums[right]:  # 左侧有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        elif nums[mid] > nums[right] or nums[left] > nums[mid]:  # 右侧有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
        else:
            left += 1
    
    return False

# 使用示例
nums = [2, 5, 6, 0, 0, 1, 2]
target = 0
print(search(nums, target))  # 输出: True
```

#### 详解

- 使用二分查找算法，但需要处理重复元素的情况。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 82. 删除排序链表中的重复元素 II（Remove Duplicates from Sorted List II）

**题目**：给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中没有重复的数字。

**LeetCode 链接**：[Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_duplicates(head):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    while head:
        if head.next and head.val == head.next.val:
            while head.next and head.val == head.next.val:
                head = head.next
            prev.next = head.next
        else:
            prev = prev.next
        head = head.next
    
    return dummy.next

# 使用示例
# 链表: 1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
# 输出: 1 -> 5
```

#### 详解

- 使用哑节点和双指针来删除链表中的重复元素。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 83. 删除排序链表中的重复元素（Remove Duplicates from Sorted List）

**题目**：给定一个排序链表，删除所有重复的元素，使每个元素只出现一次。

**LeetCode 链接**：[Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_duplicates(head):
    current = head
    while current and current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next
    return head

# 使用示例
# 链表: 1 -> 1 -> 2 -> 3 -> 3
# 输出: 1 -> 2 -> 3
```

#### 详解

- 直接遍历链表并删除重复节点。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 84. 柱状图中最大的矩形（Largest Rectangle in Histogram）

**题目**：给定一个整数数组 `heights`，每个元素表示柱状图的高度，求柱状图中能够形成的最大矩形的面积。

**LeetCode 链接**：[Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

#### Python 实现

```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights.append(0)
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    return max_area

# 使用示例
heights = [2, 1, 5, 6, 2, 3]
print(largestRectangleArea(heights))  # 输出: 10
```

#### 详解

- 使用栈来跟踪柱子的索引，从而计算每个柱子作为最矮柱子的最大矩形面积。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 85. 最大矩形（Maximal Rectangle）

**题目**：给定一个由 `'0'` 和 `'1'` 组成的二维矩阵，找到其中由 `'1'` 组成的最大矩形的面积。

**LeetCode 链接**：[Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

#### Python 实现

```python
def maximalRectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    
    def maximal_histogram_area(heights):
        stack = []
        max_area = 0
        heights.append(0)
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        return max_area
    
    max_area = 0
    heights = [0] * len(matrix[0])
    
    for row in matrix:
        for j in range(len(row)):
            heights[j] = heights[j] + 1 if row[j] == '1' else 0
        max_area = max(max_area, maximal_histogram_area(heights))
    
    return max_area

# 使用示例
matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]
print(maximalRectangle(matrix))  # 输出: 6
```

#### 详解

- 先将二维矩阵转换为一维柱状图，然后使用最大矩形面积算法。时间复杂度为 O(m * n)，空间复杂度为 O(n)。

---

### 86. 分隔链表（Partition List）

**题目**：给定一个链表和一个值 `x`，将链表中所有小于 `x` 的节点移到链表的左侧，所有大于或等于 `x` 的节点移到链表的右侧。

**LeetCode 链接**：[Partition List](https://leetcode.com/problems/partition-list/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def partition(head, x):
    before_head = ListNode(0)
    after_head = ListNode(0)
    before, after = before_head, after_head
    
    while head:
        if head.val < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next
    
    after.next = None
    before.next = after_head.next
    
    return before_head.next

# 使用示例
# 链表: 1 -> 4 -> 3 -> 2 -> 5 -> 2
# x = 3
# 输出: 1 -> 2 -> 2 -> 4 -> 3 -> 5
```

#### 详解

- 使用两个链表分别存储小于 `x` 和大于等于 `x` 的节点，最后连接两个链表。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 87. 扰乱字符串（Scramble String）

**题目**：给定两个字符串 `s1` 和 `s2`，检查 `s2` 是否是 `s1` 的扰乱字符串。扰乱字符串是通过对 `s1` 的字符进行任意次数的交换得到的。

**LeetCode 链接**：[Scramble String](https://leetcode.com/problems/scramble-string/)

#### Python 实现

```python
def is_scramble(s1, s2):
    if len(s1) != len(s2):
        return False
    if s1 == s2:
        return True
    if sorted(s1) != sorted(s2):
        return False
    
    n = len(s1)
    dp = [[[False] * n for _ in range(n)] for _ in range(n + 1)]
    
    for i in range(n):
        for j in range(n):
            dp[1][i][j] = s1[i] == s2[j]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            for j in range(n - length + 1):
                for k in range(1, length):
                    if (dp[k][i][j] and dp[length - k][i + k][j + k]) or (dp[k][i][j + length - k] and dp[length - k][i + k][j]):
                        dp[length][i][j] = True
                        break
    
    return dp[n][0][0]

# 使用示例
s1 = "great"
s2 = "rgeat"
print(is_scramble(s1, s2))  # 输出: True
```

#### 详解

- 使用动态规划算法来判断是否可以通过交换得到扰乱字符串。时间复杂度为 O(n^4)，空间复杂度为 O(n^3)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖各种算法和数据结构的应用。

### 88. 合并两个有序数组（Merge Sorted Array）

**题目**：给定两个有序整数数组 `nums1` 和 `nums2`，将 `nums2` 合并到 `nums1` 中，使得 `nums1` 成为一个有序数组。初始化 `nums1` 和 `nums2` 的长度分别为 `m` 和 `n`，`nums1` 的大小为 `m + n`，其中前 `m` 个元素是 `nums1` 的有效部分，后 `n` 个元素是 `nums1` 的空闲部分。

**LeetCode 链接**：[Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

#### Python 实现

```python
def merge(nums1, m, nums2, n):
    while m > 0 and n > 0:
        if nums1[m - 1] > nums2[n - 1]:
            nums1[m + n - 1] = nums1[m - 1]
            m -= 1
        else:
            nums1[m + n - 1] = nums2[n - 1]
            n -= 1
    
    if n > 0:
        nums1[:n] = nums2[:n]

# 使用示例
nums1 = [1, 2, 3, 0, 0, 0]
m = 3
nums2 = [2, 5, 6]
n = 3
merge(nums1, m, nums2, n)
print(nums1)  # 输出: [1, 2, 2, 3, 5, 6]
```

#### 详解

- 从数组的尾部开始合并，避免覆盖 `nums1` 中的有效部分。时间复杂度为 O(m + n)，空间复杂度为 O(1)。

---

### 89. 格雷编码（Gray Code）

**题目**：格雷编码是二进制数字的一种编码方式，其中两个连续的数字之间仅有一位不同。给定一个整数 `n`，返回 `n` 位格雷编码的序列。

**LeetCode 链接**：[Gray Code](https://leetcode.com/problems/gray-code/)

#### Python 实现

```python
def grayCode(n):
    result = [0]
    for i in range(n):
        result += [x + (1 << i) for x in reversed(result)]
    return result

# 使用示例
n = 2
print(grayCode(n))  # 输出: [0, 1, 3, 2]
```

#### 详解

- 通过逐位添加生成格雷编码序列。时间复杂度为 O(2^n)，空间复杂度为 O(2^n)。

---

### 90. 子集 II（Subsets II）

**题目**：给定一个可能包含重复数字的整数数组 `nums`，返回所有唯一的子集（幂集）。

**LeetCode 链接**：[Subsets II](https://leetcode.com/problems/subsets-ii/)

#### Python 实现

```python
def subsetsWithDup(nums):
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    nums.sort()
    result = []
    backtrack(0, [])
    return result

# 使用示例
nums = [1, 2, 2]
print(subsetsWithDup(nums))  # 输出: [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
```

#### 详解

- 使用回溯算法生成唯一的子集，并避免重复。时间复杂度为 O(2^n)，空间复杂度为 O(2^n * n)。

---

### 91. 解码方法（Decode Ways）

**题目**：给定一个只包含数字的字符串 `s`，计算有多少种不同的方法可以解码这个字符串，其中 'A' 到 'Z' 映射到 '1' 到 '26'。

**LeetCode 链接**：[Decode Ways](https://leetcode.com/problems/decode-ways/)

#### Python 实现

```python
def numDecodings(s):
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        one_digit = int(s[i - 1])
        two_digits = int(s[i - 2:i])
        
        if one_digit >= 1:
            dp[i] += dp[i - 1]
        if 10 <= two_digits <= 26:
            dp[i] += dp[i - 2]
    
    return dp[n]

# 使用示例
s = "226"
print(numDecodings(s))  # 输出: 3
```

#### 详解

- 使用动态规划算法计算不同的解码方法。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 92. 反转链表 II（Reverse Linked List II）

**题目**：给定一个链表和两个整数 `m` 和 `n`，反转链表从位置 `m` 到位置 `n` 的部分，返回修改后的链表。

**LeetCode 链接**：[Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_between(head, m, n):
    if not head or m == n:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    pre = dummy
    
    for _ in range(m - 1):
        pre = pre.next
    
    start = pre.next
    then = start.next
    
    for _ in range(n - m):
        start.next = then.next
        then.next = pre.next
        pre.next = then
        then = start.next
    
    return dummy.next

# 使用示例
# 链表: 1 -> 2 -> 3 -> 4 -> 5
# m = 2, n = 4
# 输出: 1 -> 4 -> 3 -> 2 -> 5
```

#### 详解

- 通过遍历链表和重新连接节点来反转部分链表。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 93. 邻接矩阵中的有向图（Graph Valid Tree）

**题目**：给定一个包含 `n` 个节点和一个边列表 `edges` 的图，确定这个图是否是一个有效的树。有效的树必须是连通的且无环的。

**LeetCode 链接**：[Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

#### Python 实现

```python
from collections import defaultdict, deque

def validTree(n, edges):
    if len(edges) != n - 1:
        return False
    
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    
    def bfs(start):
        queue = deque([start])
        visited.add(start)
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    bfs(0)
    
    return len(visited) == n

# 使用示例
n = 5
edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
print(validTree(n, edges))  # 输出: True
```

#### 详解

- 使用 BFS 检查图的连通性，同时检查边的数量是否符合树的性质。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 94. 二叉树的中序遍历（Binary Tree Inorder Traversal）

**题目**：给定一个二叉树，返回它的中序遍历。

**LeetCode 链接**：[Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorderTraversal(root):
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    result = []
    inorder(root)
    return result

# 使用示例
# 二叉树:    1
#           / \
#          2   3
#         /
#        4
# 输出: [4, 2, 1, 3]
```

#### 详解

- 使用递归实现中序遍历。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖更多的算法和数据结构应用。

### 95. 不同的二叉搜索树 II（Unique Binary Search Trees II）

**题目**：给定一个整数 `n`，生成所有由 `1` 到 `n` 组成的不同的二叉搜索树。

**LeetCode 链接**：[Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def generateTrees(n):
    def generate(start, end):
        if start > end:
            return [None]
        result = []
        for i in range(start, end + 1):
            left_trees = generate(start, i - 1)
            right_trees = generate(i + 1, end)
            for left in left_trees:
                for right in right_trees:
                    root = TreeNode(i)
                    root.left = left
                    root.right = right
                    result.append(root)
        return result
    
    if n == 0:
        return []
    return generate(1, n)

# 使用示例
n = 3
trees = generateTrees(n)
print(len(trees))  # 输出: 5
```

#### 详解

- 使用递归生成所有可能的二叉搜索树。时间复杂度为 O(C(n))，其中 C(n) 是第 n 个卡特兰数，空间复杂度为 O(C(n))。

---

### 96. 不同的二叉搜索树（Unique Binary Search Trees）

**题目**：给定一个整数 `n`，计算可以生成多少个不同的二叉搜索树。

**LeetCode 链接**：[Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)

#### Python 实现

```python
def numTrees(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - j - 1]
    
    return dp[n]

# 使用示例
n = 3
print(numTrees(n))  # 输出: 5
```

#### 详解

- 使用动态规划算法计算二叉搜索树的数量。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 97. 交错字符串（Interleaving String）

**题目**：给定三个字符串 `s1`、`s2` 和 `s3`，判断 `s3` 是否是由 `s1` 和 `s2` 交错组成的。

**LeetCode 链接**：[Interleaving String](https://leetcode.com/problems/interleaving-string/)

#### Python 实现

```python
def isInterleave(s1, s2, s3):
    if len(s1) + len(s2) != len(s3):
        return False
    
    dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    dp[0][0] = True
    
    for i in range(1, len(s1) + 1):
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
    
    for j in range(1, len(s2) + 1):
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
    
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or \
                       (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
    
    return dp[len(s1)][len(s2)]

# 使用示例
s1 = "aabcc"
s2 = "dbbca"
s3 = "aadbbcbcac"
print(isInterleave(s1, s2, s3))  # 输出: True
```

#### 详解

- 使用动态规划算法来检查字符串是否可以交错组合。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 98. 验证二叉搜索树（Validate Binary Search Tree）

**题目**：给定一个二叉树，判断其是否为有效的二叉搜索树。

**LeetCode 链接**：[Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isValidBST(root):
    def is_valid(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return is_valid(node.left, low, node.val) and is_valid(node.right, node.val, high)
    
    return is_valid(root, float('-inf'), float('inf'))

# 使用示例
# 二叉树:    2
#           / \
#          1   3
# 输出: True
```

#### 详解

- 通过递归检查每个节点是否满足二叉搜索树的条件。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 99. 恢复二叉搜索树（Recover Binary Search Tree）

**题目**：给定一个二叉搜索树，其中的两个节点被错误地交换了，恢复树使其成为一个有效的二叉搜索树。

**LeetCode 链接**：[Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def recoverTree(root):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node] + inorder(node.right)
    
    nodes = inorder(root)
    x = y = pred = None
    
    for i in range(len(nodes)):
        if pred and nodes[i].val < pred.val:
            y = nodes[i]
            if not x:
                x = pred
            else:
                break
        pred = nodes[i]
    
    if x and y:
        x.val, y.val = y.val, x.val

# 使用示例
# 二叉树:    1
#           / \
#          3   2
# 输出: 二叉搜索树变为:    2
#                     / \
#                    1   3
```

#### 详解

- 通过中序遍历找到错误的节点并交换。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 100. 相同的树（Same Tree）

**题目**：给定两个二叉树，检查它们是否完全相同。

**LeetCode 链接**：[Same Tree](https://leetcode.com/problems/same-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val) and isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

# 使用示例
# 二叉树1:    1        二叉树2:    1
#            / \                / \
#           2   3              2   3
# 输出: True
```

#### 详解

- 递归检查两个树的节点值和结构。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 101. 对称二叉树（Symmetric Tree）

**题目**：给定一个二叉树，检查它是否是对称的（即，左右子树互为镜像）。

**LeetCode 链接**：[Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSymmetric(root):
    def is_mirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (t1.val == t2.val) and is_mirror(t1.right, t2.left) and is_mirror(t1.left, t2.right)
    
    return is_mirror(root, root)

# 使用示例
# 二叉树:    1
#           / \
#          2   2
#         / \ / \
#        3  4 4  3
# 输出: True
```

#### 详解

- 使用递归检查树的左右子树是否对称。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 102. 二叉树的层序遍历（Binary Tree Level Order Traversal）

**题目**：给定一个二叉树，返回其节点值的层序遍历（即逐层访问）。

**LeetCode 链接**：[Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

#### Python 实现

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def levelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    
    return result

# 使用示例
# 二叉树:    3
#           / \
#          9  20
#             /  \
#            15   7
# 输出: [[3], [9, 20], [15, 7]]
```

#### 详解

- 使用队列进行层序遍历。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涉及不同的数据结构和算法问题。

### 103. 二叉树的锯齿形层序遍历（Binary Tree Zigzag Level Order Traversal）

**题目**：给定一个二叉树，返回其节点值的锯齿形层序遍历（即逐层访问，但每层的顺序交替）。

**LeetCode 链接**：[Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

#### Python 实现

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def zigzagLevelOrder(root):
    if not root:
        return []
    
    result, level, zigzag = [], deque([root]), False
    
    while level:
        current_level, next_level = [], deque()
        while level:
            node = level.popleft()
            current_level.append(node.val)
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        result.append(current_level if not zigzag else current_level[::-1])
        level, zigzag = next_level, not zigzag
    
    return result

# 使用示例
# 二叉树:    3
#           / \
#          9  20
#             /  \
#            15   7
# 输出: [[3], [20, 9], [15, 7]]
```

#### 详解

- 使用队列和布尔变量控制层序遍历的顺序。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 104. 二叉树的最大深度（Maximum Depth of Binary Tree）

**题目**：给定一个二叉树，返回其最大深度。最大深度是指从根节点到最远叶子节点的最长路径上的节点数。

**LeetCode 链接**：[Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root):
    if not root:
        return 0
    return max(maxDepth(root.left), maxDepth(root.right)) + 1

# 使用示例
# 二叉树:    3
#           / \
#          9  20
#             /  \
#            15   7
# 输出: 3
```

#### 详解

- 通过递归计算左右子树的深度并取最大值。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 105. 从前序与中序遍历序列构造二叉树（Construct Binary Tree from Preorder and Inorder Traversal）

**题目**：给定前序遍历和中序遍历的结果，构造二叉树。

**LeetCode 链接**：[Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root_val = preorder[0]
    root = TreeNode(root_val)
    root_index = inorder.index(root_val)
    
    root.left = buildTree(preorder[1:1 + root_index], inorder[:root_index])
    root.right = buildTree(preorder[1 + root_index:], inorder[root_index + 1:])
    
    return root

# 使用示例
preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]
tree = buildTree(preorder, inorder)
```

#### 详解

- 通过前序遍历的第一个元素确定根节点，并利用中序遍历分割左右子树。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 106. 从中序与后序遍历序列构造二叉树（Construct Binary Tree from Inorder and Postorder Traversal）

**题目**：给定中序遍历和后序遍历的结果，构造二叉树。

**LeetCode 链接**：[Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None
    
    root_val = postorder[-1]
    root = TreeNode(root_val)
    root_index = inorder.index(root_val)
    
    root.left = buildTree(inorder[:root_index], postorder[:root_index])
    root.right = buildTree(inorder[root_index + 1:], postorder[root_index:-1])
    
    return root

# 使用示例
inorder = [9, 3, 15, 20, 7]
postorder = [9, 15, 7, 20, 3]
tree = buildTree(inorder, postorder)
```

#### 详解

- 通过后序遍历的最后一个元素确定根节点，并利用中序遍历分割左右子树。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 107. 二叉树的层序遍历 II（Binary Tree Level Order Traversal II）

**题目**：给定一个二叉树，返回其节点值的层序遍历的逆序（即自底向上的层序遍历）。

**LeetCode 链接**：[Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)

#### Python 实现

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def levelOrderBottom(root):
    if not root:
        return []
    
    result, level = [], deque([root])
    
    while level:
        current_level = []
        for _ in range(len(level)):
            node = level.popleft()
            current_level.append(node.val)
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        result.insert(0, current_level)
    
    return result

# 使用示例
# 二叉树:    3
#           / \
#          9  20
#             /  \
#            15   7
# 输出: [[15, 7], [9, 20], [3]]
```

#### 详解

- 使用队列进行层序遍历，并在结果中插入每层的节点值。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 108. 将有序数组转换为二叉搜索树（Convert Sorted Array to Binary Search Tree）

**题目**：将一个有序数组转换为一个高度平衡的二叉搜索树。

**LeetCode 链接**：[Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sortedArrayToBST(nums):
    def convert(left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = convert(left, mid - 1)
        root.right = convert(mid + 1, right)
        return root
    
    return convert(0, len(nums) - 1)

# 使用示例
nums = [-10, -3, 0, 5, 9]
tree = sortedArrayToBST(nums)
```

#### 详解

- 使用递归将有序数组转换为平衡的二叉搜索树。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 109. 有序链表转换为二叉搜索树（Convert Sorted List to Binary Search Tree）

**题目**：给定一个按升序排列的链表，将其转换为高度平衡的二叉搜索树。

**LeetCode 链接**：[Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sortedListToBST(head):
    def findMiddle(left, right):
        slow, fast = left, left
        while fast != right and fast.next != right:
            slow = slow.next
            fast = fast.next
            fast = fast.next
        return slow
    
    def convert(left, right):
        if left == right:
            return None
        mid = findMiddle(left, right)
        root = TreeNode(mid.val)
        root.left = convert(left, mid)
        root.right = convert(mid.next, right)
        return root
    
    return convert(head, None)

# 使用示例
# 链表: -10 -> -3 -> 0 -> 5 -> 9
# 输出: 二叉搜索树
```

#### 详解

- 通过递归构造二叉搜索树，使用快慢指针找到中间节点。时间复杂度为 O(n)，空间复杂度为 O(log n)（递归栈的深度）。

---

### 110. 平衡二叉树（Balanced Binary Tree）

**题目**：给定一个二叉树，判断其是否为平衡二叉树。平衡二叉树的定义是：每个节点的左右子树的高度差的绝对值不超过 1。

**LeetCode 链接**：[Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isBalanced(root):
    def check_height(node):
        if not node:
            return 0
        left_height = check_height(node.left)
        right_height = check_height(node.right)
        if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
            return -1
        return max(left_height, right_height) + 1
    
    return check_height(root) != -1

# 使用示例
# 二叉树:    3
#           / \
#          9  20
#             /  \
#            15   7
# 输出: True
```

#### 详解

- 使用递归检查每个节点的左右子树的高度差。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖更广泛的算法和数据结构应用。

### 111. 二叉树的最小深度（Minimum Depth of Binary Tree）

**题目**：给定一个二叉树，返回其最小深度。最小深度是指从根节点到最近的叶子节点的路径上的节点数。

**LeetCode 链接**：[Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def minDepth(root):
    if not root:
        return 0
    if not root.left and not root.right:
        return 1
    if not root.left:
        return minDepth(root.right) + 1
    if not root.right:
        return minDepth(root.left) + 1
    return min(minDepth(root.left), minDepth(root.right)) + 1

# 使用示例
# 二叉树:    3
#           / \
#          9  20
#             /  \
#            15   7
# 输出: 2
```

#### 详解

- 使用递归计算每个节点的最小深度。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 112. 路径总和（Path Sum）

**题目**：给定一个二叉树和一个目标和，判断树中是否存在从根节点到叶子节点的路径，使得路径上的所有节点值之和等于目标和。

**LeetCode 链接**：[Path Sum](https://leetcode.com/problems/path-sum/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def hasPathSum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    targetSum -= root.val
    return hasPathSum(root.left, targetSum) or hasPathSum(root.right, targetSum)

# 使用示例
# 二叉树:    5
#           / \
#          4   8
#         /   / \
#        11  13  4
#       /  \      \
#      7    2      1
# 输出: True (5 -> 4 -> 11 -> 2)
```

#### 详解

- 使用递归检查每个路径的节点值之和。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 113. 路径总和 II（Path Sum II）

**题目**：给定一个二叉树和一个目标和，返回所有从根节点到叶子节点的路径，使得路径上的所有节点值之和等于目标和。

**LeetCode 链接**：[Path Sum II](https://leetcode.com/problems/path-sum-ii/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def pathSum(root, targetSum):
    def dfs(node, current_path, target):
        if not node:
            return
        current_path.append(node.val)
        if not node.left and not node.right and sum(current_path) == target:
            result.append(list(current_path))
        else:
            dfs(node.left, current_path, target)
            dfs(node.right, current_path, target)
        current_path.pop()
    
    result = []
    dfs(root, [], targetSum)
    return result

# 使用示例
# 二叉树:    5
#           / \
#          4   8
#         /   / \
#        11  13  4
#       /  \      \
#      7    2      1
# 输出: [[5, 4, 11, 2], [5, 8, 4, 1]]
```

#### 详解

- 使用深度优先搜索（DFS）查找所有符合条件的路径。时间复杂度为 O(n)，空间复杂度为 O(h)。

---

### 114. 二叉树展开为链表（Flatten Binary Tree to Linked List）

**题目**：给定一个二叉树，将其展开为链表，使得链表的前序遍历与二叉树的前序遍历一致。

**LeetCode 链接**：[Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def flatten(root):
    def flatten_tree(node):
        if not node:
            return None
        if not node.left and not node.right:
            return node
        left_tail = flatten_tree(node.left)
        right_tail = flatten_tree(node.right)
        
        if node.left:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        
        return right_tail if right_tail else left_tail
    
    flatten_tree(root)

# 使用示例
# 二叉树:    1
#           / \
#          2   5
#         / \   \
#        3   4   6
# 输出: 1 -> 2 -> 3 -> 4 -> 5 -> 6
```

#### 详解

- 递归地展开二叉树为链表。时间复杂度为 O(n)，空间复杂度为 O(h)。

---

### 115. 不同的子序列（Distinct Subsequences）

**题目**：给定一个字符串 `s` 和一个字符串 `t`，计算 `t` 在 `s` 中出现的不同子序列的数量。

**LeetCode 链接**：[Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)

#### Python 实现

```python
def numDistinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        dp[0][j] = 0
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j]
    
    return dp[m][n]

# 使用示例
s = "rabbbit"
t = "rabbit"
print(numDistinct(s, t))  # 输出: 3
```

#### 详解

- 使用动态规划计算子序列的数量。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)。

---

### 116. 填充每个节点的下一个右侧节点指针（Populating Next Right Pointers in Each Node）

**题目**：给定一个完美二叉树，填充每个节点的下一个右侧节点指针，使其指向右侧的节点。如果没有右侧节点，则填充 `null`。

**LeetCode 链接**：[Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

def connect(root):
    if not root:
        return root
    
    def connect_two_nodes(node1, node2):
        if not node1 or not node2:
            return
        node1.next = node2
        connect_two_nodes(node1.left, node1.right)
        connect_two_nodes(node2.left, node2.right)
        connect_two_nodes(node1.right, node2.left)
    
    connect_two_nodes(root.left, root.right)
    return root

# 使用示例
# 二叉树:    1
#           / \
#          2   3
#         / \ / \
#        4  5 6  7
# 输出: 每个节点的 next 指针被正确填充
```

#### 详解

- 使用递归连接相邻的节点。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 118. 杨辉三角（Pascal's Triangle）

**题目**：给定一个非负整数 `numRows`，生成杨辉三角的前 `numRows` 行。

**LeetCode 链接**：[Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/)

#### Python 实现

```python
def generate(numRows):
    result = []
    for i in range(numRows):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = result[i - 1][j - 1] + result[i - 1][j]
        result.append(row)
    return result

# 使用示例
print(generate(5))  # 输出: [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
```

#### 详解

- 通过动态生成每一行的值。时间复杂度为 O(numRows^2)，空间复杂度为 O(numRows^2)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涉及更复杂的算法和数据结构问题。

### 119. 杨辉三角 II（Pascal's Triangle II）

**题目**：给定一个非负整数 `rowIndex`，返回杨辉三角中第 `rowIndex` 行的元素（索引从 0 开始）。

**LeetCode 链接**：[Pascal's Triangle II](https://leetcode.com/problems/pascals-triangle-ii/)

#### Python 实现

```python
def getRow(rowIndex):
    row = [1]
    for i in range(1, rowIndex + 1):
        row.append(row[-1] * (rowIndex - i + 1) // i)
    return row

# 使用示例
print(getRow(3))  # 输出: [1, 3, 3, 1]
```

#### 详解

- 利用杨辉三角的性质直接计算每一行的元素，避免了生成整个三角形。时间复杂度为 O(rowIndex)，空间复杂度为 O(rowIndex)。

---

### 121. 买卖股票的最佳时机（Best Time to Buy and Sell Stock）

**题目**：给定一个数组 `prices`，其中 `prices[i]` 是股票在第 `i` 天的价格。你可以在多次交易中只进行一次买卖，计算你能够获得的最大利润。

**LeetCode 链接**：[Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

#### Python 实现

```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

# 使用示例
print(maxProfit([7, 1, 5, 3, 6, 4]))  # 输出: 5
```

#### 详解

- 使用贪心算法，通过遍历计算最小价格和最大利润。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 122. 买卖股票的最佳时机 II（Best Time to Buy and Sell Stock II）

**题目**：给定一个数组 `prices`，其中 `prices[i]` 是股票在第 `i` 天的价格。你可以在多次交易中进行多次买卖，计算你能够获得的最大利润。

**LeetCode 链接**：[Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

#### Python 实现

```python
def maxProfit(prices):
    total_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            total_profit += prices[i] - prices[i - 1]
    return total_profit

# 使用示例
print(maxProfit([7, 1, 5, 3, 6, 4]))  # 输出: 7
```

#### 详解

- 贪心算法，通过找出每次价格上涨的区间来计算总利润。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 124. 二叉树中的最大路径和（Binary Tree Maximum Path Sum）

**题目**：给定一个非空二叉树，计算其最大路径和。路径的定义是：从任何节点出发，沿着父节点到子节点的路径可以重新开始。

**LeetCode 链接**：[Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxPathSum(root):
    def helper(node):
        nonlocal max_sum
        if not node:
            return 0
        left = max(helper(node.left), 0)
        right = max(helper(node.right), 0)
        max_sum = max(max_sum, node.val + left + right)
        return node.val + max(left, right)
    
    max_sum = float('-inf')
    helper(root)
    return max_sum

# 使用示例
# 二叉树:    1
#           / \
#          2   3
# 输出: 6 (2 -> 1 -> 3)
```

#### 详解

- 使用递归计算每个节点的最大路径和，并更新全局最大值。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 126. 单词接龙 II（Word Ladder II）

**题目**：给定两个单词 `beginWord` 和 `endWord` 以及一个字典 `wordList`，找到所有从 `beginWord` 到 `endWord` 的最短转换序列。每次转换只能改变一个字母，并且每个转换后的单词必须在字典中。

**LeetCode 链接**：[Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)

#### Python 实现

```python
from collections import defaultdict, deque

def findLadders(beginWord, endWord, wordList):
    def add_to_graph(word):
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i+1:]
            graph[pattern].append(word)

    def bfs():
        queue = deque([(beginWord, [beginWord])])
        visited = set([beginWord])
        while queue:
            word, path = queue.popleft()
            if word == endWord:
                results.append(path)
                continue
            for i in range(len(word)):
                pattern = word[:i] + "*" + word[i+1:]
                for neighbor in graph[pattern]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
    
    graph = defaultdict(list)
    wordList.add(beginWord)
    wordList.add(endWord)
    for word in wordList:
        add_to_graph(word)
    
    results = []
    bfs()
    return results

# 使用示例
beginWord = "hit"
endWord = "cog"
wordList = set(["hot","dot","dog","lot","log","cog"])
print(findLadders(beginWord, endWord, wordList))
```

#### 详解

- 使用广度优先搜索（BFS）和图的邻接表表示单词转换。时间复杂度为 O(N * M^2)，空间复杂度为 O(N * M)，其中 N 为字典大小，M 为单词长度。

---

### 130. 被围绕的区域（Surrounded Regions）

**题目**：给定一个二维矩阵，包含 `'X'` 和 `'O'`。如果一个区域（被 `'X'` 包围的区域）中的所有 `'O'` 都被 `'X'` 包围，则该区域的 `'O'` 将被替换为 `'X'`。

**LeetCode 链接**：[Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)

#### Python 实现

```python
def solve(board):
    if not board or not board[0]:
        return
    
    def dfs(x, y):
        if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]) or board[x][y] != 'O':
            return
        board[x][y] = '#'
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs(x + dx, y + dy)
    
    rows, cols = len(board), len(board[0])
    for i in range(rows):
        dfs(i, 0)
        dfs(i, cols - 1)
    for j in range(cols):
        dfs(0, j)
        dfs(rows - 1, j)
    
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'

# 使用示例
board = [
    ["X","X","X","X"],
    ["X","O","O","X"],
    ["X","X","O","X"],
    ["X","O","X","X"]
]
solve(board)
# 输出: [["X","X","X","X"], ["X","X","X","X"], ["X","X","X","X"], ["X","O","X","X"]]
```

#### 详解

- 使用深度优先搜索（DFS）标记未被包围的 `'O'`，然后替换被包围的 `'O'` 为 `'X'`。时间复杂度为 O(m * n)，空间复杂度为 O(m * n)，其中 m 和 n 分别为矩阵的行和列数。

---

### 137. 只出现一次的数字 II（Single Number II）

**题目**：给定一个整数数组 `nums`，每个元素出现三次，除了一个元素只出现一次。找出只出现一次的那个元素。

**LeetCode 链接**：[Single Number II](https://leetcode.com/problems/single-number-ii/)

#### Python 实现

```python
def singleNumber(nums):
    ones, twos = 0, 0
    for num in nums:
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones
    return ones

# 使用示例
print(singleNumber([2, 2, 3, 2]))  # 输出: 3
```

#### 详解

- 使用位运算来追踪出现次数。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

继续介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖更复杂的算法和数据结构应用。

### 138. 复制带随机指针的链表（Copy List with Random Pointer）

**题目**：给定一个链表中的每个节点包含一个随机指针，随机指针可以指向链表中的任何节点或 `null`。复制链表并返回新链表的头节点。

**LeetCode 链接**：[Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

#### Python 实现

```python
class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copyRandomList(head):
    if not head:
        return None
    
    # Step 1: Copy each node and insert it right next to the original node
    current = head
    while current:
        new_node = Node(current.val, current.next)
        current.next = new_node
        current = new_node.next
    
    # Step 2: Copy the random pointers
    current = head
    while current:
        new_node = current.next
        new_node.random = current.random.next if current.random else None
        current = new_node.next
    
    # Step 3: Separate the copied list from the original list
    dummy = Node(0)
    copy_current, current = dummy, head
    while current:
        copy_current.next = current.next
        copy_current = copy_current.next
        current.next = copy_current.next
        current = current.next
    
    return dummy.next

# 使用示例
# 链表结构:
# 1 -> 2 -> 3
# |    |    |
# v    v    v
# 2    1    3
```

#### 详解

- 通过三步操作来复制链表。时间复杂度为 O(n)，空间复杂度为 O(1)（不包括返回的复制链表的空间）。

---

### 139. 单词拆分（Word Break）

**题目**：给定一个非空字符串 `s` 和一个单词字典 `wordDict`，判断 `s` 是否可以被空格拆分成一个或多个字典中出现的单词。

**LeetCode 链接**：[Word Break](https://leetcode.com/problems/word-break/)

#### Python 实现

```python
def wordBreak(s, wordDict):
    wordSet = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wordSet:
                dp[i] = True
                break
    
    return dp[-1]

# 使用示例
s = "leetcode"
wordDict = ["leet", "code"]
print(wordBreak(s, wordDict))  # 输出: True
```

#### 详解

- 使用动态规划检查是否可以用字典中的单词分解字符串。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 141. 环形链表（Linked List Cycle）

**题目**：给定一个链表，判断链表中是否有环。 

**LeetCode 链接**：[Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def hasCycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# 使用示例
# 链表: 3 -> 2 -> 0 -> -4 (环形: -4 -> 2)
```

#### 详解

- 使用快慢指针（Floyd 判圈算法）检测链表是否有环。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 142. 环形链表 II（Linked List Cycle II）

**题目**：给定一个链表，返回链表中环的入口节点。如果链表中没有环，返回 `null`。

**LeetCode 链接**：[Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def detectCycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            entry = head
            while entry != slow:
                entry = entry.next
                slow = slow.next
            return entry
    return None

# 使用示例
# 链表: 3 -> 2 -> 0 -> -4 (环形: -4 -> 2)
```

#### 详解

- 通过检测环的入口节点。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 144. 二叉树的前序遍历（Binary Tree Preorder Traversal）

**题目**：给定一个二叉树，返回其前序遍历的节点值。

**LeetCode 链接**：[Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorderTraversal(root):
    result = []
    def traverse(node):
        if not node:
            return
        result.append(node.val)
        traverse(node.left)
        traverse(node.right)
    
    traverse(root)
    return result

# 使用示例
# 二叉树:    1
#           / \
#          2   3
# 输出: [1, 2, 3]
```

#### 详解

- 使用递归实现前序遍历。时间复杂度为 O(n)，空间复杂度为 O(h)，其中 h 为树的高度。

---

### 146. LRU 缓存（LRU Cache）

**题目**：设计并实现一个 LRU（Least Recently Used）缓存机制。实现 `LRUCache` 类，支持 `get` 和 `put` 操作。

**LeetCode 链接**：[LRU Cache](https://leetcode.com/problems/lru-cache/)

#### Python 实现

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

# 使用示例
lru_cache = LRUCache(2)
lru_cache.put(1, 1)
lru_cache.put(2, 2)
print(lru_cache.get(1))  # 输出: 1
lru_cache.put(3, 3)     # 驱逐键 2
print(lru_cache.get(2))  # 输出: -1 (未找到)
lru_cache.put(4, 4)     # 驱逐键 1
print(lru_cache.get(1))  # 输出: -1 (未找到)
print(lru_cache.get(3))  # 输出: 3
print(lru_cache.get(4))  # 输出: 4
```

#### 详解

- 使用 `OrderedDict` 实现 LRU 缓存。时间复杂度为 O(1) 的 `get` 和 `put` 操作。空间复杂度为 O(capacity)。

---

### 151. 翻转字符串里的单词（Reverse Words in a String）

**题目**：给定一个字符串 `s`，返回字符串中的单词以相反的顺序排列。单词之间用单个空格分隔。

**LeetCode 链接**：[Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)

#### Python 实现

```python
def reverseWords(s):
    words = s.split()
    return ' '.join(reversed(words))

# 使用示例
print(reverseWords("the sky is blue"))  # 输出: "blue is sky the"
```

#### 详解

- 使用 Python 内置的 `split` 和 `join` 函数来处理字符串。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

继续为你介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖进阶算法和数据结构的应用。

### 152. 乘积最大子数组（Maximum Product Subarray）

**题目**：给你一个整数数组 `nums`，请找出数组中乘积最大的连续子数组（至少包含一个数字），并返回该子数组所对应的乘积。

**LeetCode 链接**：[Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

#### Python 实现

```python
def maxProduct(nums):
    if not nums:
        return 0

    max_product = min_product = result = nums[0]
    
    for num in nums[1:]:
        if num < 0:
            max_product, min_product = min_product, max_product
        max_product = max(num, max_product * num)
        min_product = min(num, min_product * num)
        result = max(result, max_product)
    
    return result

# 使用示例
nums = [2, 3, -2, 4]
print(maxProduct(nums))  # 输出: 6
```

#### 详解

- 动态规划，维护两个变量 `max_product` 和 `min_product` 分别记录当前的最大值和最小值，遇到负数时交换这两个值。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 153. 寻找旋转排序数组中的最小值（Find Minimum in Rotated Sorted Array）

**题目**：假设按照升序排序的数组在预先未知的某个点上进行了旋转，请找出旋转后数组中的最小元素。

**LeetCode 链接**：[Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

#### Python 实现

```python
def findMin(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]

# 使用示例
nums = [4, 5, 6, 7, 0, 1, 2]
print(findMin(nums))  # 输出: 0
```

#### 详解

- 使用二分查找法，在旋转数组中查找最小值。时间复杂度为 O(log n)，空间复杂度为 O(1)。

---

### 198. 打家劫舍（House Robber）

**题目**：给定一个数组 `nums` 代表每个房屋中的金额，不能抢相邻的两户，返回能够抢劫到的最高金额。

**LeetCode 链接**：[House Robber](https://leetcode.com/problems/house-robber/)

#### Python 实现

```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = 0, nums[0]
    
    for i in range(1, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    
    return prev1

# 使用示例
nums = [2, 7, 9, 3, 1]
print(rob(nums))  # 输出: 12
```

#### 详解

- 动态规划，`prev1` 表示到达当前房屋时能抢到的最大金额，`prev2` 表示到达前一个房屋时的最大金额。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 200. 岛屿数量（Number of Islands）

**题目**：给定一个二维网格，'1' 表示陆地，'0' 表示水域，计算网格中有多少个岛屿。岛屿被水域分隔，且只能垂直或水平连接。

**LeetCode 链接**：[Number of Islands](https://leetcode.com/problems/number-of-islands/)

#### Python 实现

```python
def numIslands(grid):
    if not grid:
        return 0
    
    def dfs(i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    
    return count

# 使用示例
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
print(numIslands(grid))  # 输出: 1
```

#### 详解

- 使用深度优先搜索（DFS）遍历每个岛屿，并将其所有陆地标记为已访问。时间复杂度为 O(m*n)，空间复杂度为 O(m*n)。

---

### 207. 课程表（Course Schedule）

**题目**：给你 `numCourses` 门课程以及课程间的先修关系 `prerequisites`，判断是否可以完成所有课程。

**LeetCode 链接**：[Course Schedule](https://leetcode.com/problems/course-schedule/)

#### Python 实现

```python
from collections import defaultdict, deque

def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    indegree = [0] * numCourses
    
    for dest, src in prerequisites:
        graph[src].append(dest)
        indegree[dest] += 1
    
    queue = deque([i for i in range(numCourses) if indegree[i] == 0])
    count = 0
    
    while queue:
        course = queue.popleft()
        count += 1
        for neighbor in graph[course]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == numCourses

# 使用示例
numCourses = 2
prerequisites = [[1, 0]]
print(canFinish(numCourses, prerequisites))  # 输出: True
```

#### 详解

- 使用拓扑排序（BFS）来检测是否存在环，判断能否完成所有课程。时间复杂度为 O(V + E)，空间复杂度为 O(V + E)。

---

### 208. 实现 Trie（前缀树）（Implement Trie (Prefix Tree)）

**题目**：实现一个 Trie（前缀树），包含插入、搜索、和判断是否有前缀等操作。

**LeetCode 链接**：[Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

#### Python 实现

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# 使用示例
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))  # 输出: True
print(trie.startsWith("app"))  # 输出: True
print(trie.search("app"))  # 输出: False
```

#### 详解

- 前缀树的实现，支持高效的单词插入和前缀查找。插入、搜索和前缀判断的时间复杂度均为 O(m)，m 是单词或前缀的长度。

---

### 215. 数组中的第 K 个最大元素（Kth Largest Element in an Array）

**题目**：给定一个未排序的数组，找到其中第 `k` 个最大的元素。

**LeetCode 链接**：[Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

#### Python 实现

```python
import heapq

def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]

# 使用示例
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(findKthLargest(nums, k))  # 输出: 5
```

#### 详解

- 使用堆（heapq 库）找到第 k 大的元素。时间复杂度为 O(n log k)，空间复杂度为 O(k)。

---

继续为你介绍更多经典的 LeetCode 题目及其 Python 实现，涵盖广泛的算法与数据结构，帮助你提升编程技能。

### 221. 最大正方形（Maximal Square）

**题目**：在一个由 0 和 1 组成的二维矩阵中，找到只包含 1 的最大正方形，并返回其面积。

**LeetCode 链接**：[Maximal Square](https://leetcode.com/problems/maximal-square/)

#### Python 实现

```python
def maximalSquare(matrix):
    if not matrix:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    max_side = 0
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if matrix[i - 1][j - 1] == '1':
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                max_side = max(max_side, dp[i][j])

    return max_side * max_side

# 使用示例
matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]
print(maximalSquare(matrix))  # 输出: 4
```

#### 详解

- 使用动态规划构造辅助表 `dp`，每个 `dp[i][j]` 记录以 `(i, j)` 为右下角的最大正方形的边长。时间复杂度为 O(m*n)，空间复杂度为 O(m*n)。

---

### 234. 回文链表（Palindrome Linked List）

**题目**：给定一个单链表，判断其是否为回文链表。

**LeetCode 链接**：[Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

#### Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def isPalindrome(head):
    fast = slow = head
    prev = None

    # 找到中间节点
    while fast and fast.next:
        fast = fast.next.next
        next_slow = slow.next
        slow.next = prev
        prev = slow
        slow = next_slow

    if fast:
        slow = slow.next

    # 比较前后两部分
    while prev and slow:
        if prev.val != slow.val:
            return False
        prev = prev.next
        slow = slow.next

    return True

# 使用示例
# 链表：1 -> 2 -> 2 -> 1
head = ListNode(1, ListNode(2, ListNode(2, ListNode(1))))
print(isPalindrome(head))  # 输出: True
```

#### 详解

- 使用快慢指针找到链表中点，翻转前半部分链表，然后逐一比较前后两部分。时间复杂度为 O(n)，空间复杂度为 O(1)。

---

### 236. 二叉树的最近公共祖先（Lowest Common Ancestor of a Binary Tree）

**题目**：给定一个二叉树，找到两个指定节点的最近公共祖先。

**LeetCode 链接**：[Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

#### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root
    return left if left else right

# 使用示例
# 二叉树:       3
#             /   \
#            5     1
#           / \   / \
#          6   2 0   8
#             / \
#            7   4
root = TreeNode(3)
root.left = TreeNode(5)
root.right = TreeNode(1)
p = root.left  # 节点 5
q = root.right  # 节点 1
print(lowestCommonAncestor(root, p, q))  # 输出: 3
```

#### 详解

- 使用递归方法，判断左右子树是否包含目标节点。如果左右子树都有节点，说明当前节点就是最近公共祖先。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 238. 除自身以外数组的乘积（Product of Array Except Self）

**题目**：给定一个数组 `nums`，返回一个数组，数组的每个元素为除自身以外其他元素的乘积。

**LeetCode 链接**：[Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

#### Python 实现

```python
def productExceptSelf(nums):
    n = len(nums)
    output = [1] * n
    left = right = 1

    for i in range(n):
        output[i] *= left
        left *= nums[i]

    for i in range(n - 1, -1, -1):
        output[i] *= right
        right *= nums[i]

    return output

# 使用示例
nums = [1, 2, 3, 4]
print(productExceptSelf(nums))  # 输出: [24, 12, 8, 6]
```

#### 详解

- 使用两个遍历，分别计算每个位置左边元素的乘积和右边元素的乘积。时间复杂度为 O(n)，空间复杂度为 O(1)（除结果数组外）。

---

### 279. 完全平方数（Perfect Squares）

**题目**：给定一个正整数 `n`，找到若干个完全平方数（如 1, 4, 9, 16 ...）使得它们的和等于 `n`，你需要让完全平方数的数量最少。

**LeetCode 链接**：[Perfect Squares](https://leetcode.com/problems/perfect-squares/)

#### Python 实现

```python
import math
from collections import deque

def numSquares(n):
    squares = [i**2 for i in range(1, int(math.sqrt(n)) + 1)]
    queue = deque([n])
    level = 0

    while queue:
        level += 1
        for _ in range(len(queue)):
            remainder = queue.popleft()
            for square in squares:
                if remainder == square:
                    return level
                if remainder < square:
                    break
                queue.append(remainder - square)

# 使用示例
n = 12
print(numSquares(n))  # 输出: 3
```

#### 详解

- 使用 BFS 逐层递减 `n`，每次减去一个完全平方数，直到 `n` 等于某个完全平方数。时间复杂度为 O(n√n)，空间复杂度为 O(n)。

---

### 300. 最长上升子序列（Longest Increasing Subsequence）

**题目**：给定一个无序的整数数组，找到其中最长的严格递增子序列的长度。

**LeetCode 链接**：[Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

#### Python 实现

```python
def lengthOfLIS(nums):
    if not nums:
        return 0

    dp = [1] * len(nums)

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# 使用示例
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(lengthOfLIS(nums))  # 输出: 4
```

#### 详解

- 动态规划，`dp[i]` 表示以 `nums[i]` 结尾的最长上升子序列长度。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 322. 零钱兑换（Coin Change）

**题目**：给定不同面额的硬币 `coins` 和一个总金额 `amount`，计算可以凑成该金额的最少硬币数。如果无法凑成，返回 -1。

**LeetCode 链接**：[Coin Change](https://leetcode.com/problems/coin-change/)

#### Python 实现

```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# 使用示例
coins = [1, 2, 5]
amount = 11
print(coinChange(coins, amount))  # 输出: 3
```

#### 详解

- 动态规划，`dp[i]` 表示凑齐金额 `i` 所需的最少硬币数。时间复杂度为 O(n * amount)，空间复杂度为 O(amount)。

---

继续为你提供更多经典的 LeetCode 题目及其 Python 实现，进一步深入各种算法与数据结构的应用。

### 406. 根据身高重建队列（Queue Reconstruction by Height）

**题目**：假设有一群人站成一个队列，每个人由一个整数对 (h, k) 表示，其中 h 是这个人的身高，k 是这个人前面身高大于等于 h 的人数。请根据每个人的身高和 k 值，重新构造队列。

**LeetCode 链接**：[Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)

#### Python 实现

```python
def reconstructQueue(people):
    people.sort(key=lambda x: (-x[0], x[1]))
    queue = []
    for person in people:
        queue.insert(person[1], person)
    return queue

# 使用示例
people = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
print(reconstructQueue(people))  # 输出: [[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
```

#### 详解

- 先按身高从高到低排序，若身高相同，则按 k 值升序排序。然后按照 k 值将每个人插入到对应位置。时间复杂度为 O(n^2)，空间复杂度为 O(n)。

---

### 416. 分割等和子集（Partition Equal Subset Sum）

**题目**：给定一个只包含正整数的非空数组，是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**LeetCode 链接**：[Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

#### Python 实现

```python
def canPartition(nums):
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]

    return dp[target]

# 使用示例
nums = [1, 5, 11, 5]
print(canPartition(nums))  # 输出: True
```

#### 详解

- 转化为背包问题，目标是找到一个子集，其和等于总和的一半。时间复杂度为 O(n*sum/2)，空间复杂度为 O(sum/2)。

---

### 494. 目标和（Target Sum）

**题目**：给定一个整数数组 `nums` 和一个目标值 `S`，你可以在数组中每个元素前添加 `+` 或 `-`，从而形成一个表达式。找到所有不同的表达式等于 `S` 的方法数。

**LeetCode 链接**：[Target Sum](https://leetcode.com/problems/target-sum/)

#### Python 实现

```python
def findTargetSumWays(nums, S):
    total_sum = sum(nums)
    if total_sum < S or (total_sum + S) % 2 != 0:
        return 0

    target = (total_sum + S) // 2
    dp = [0] * (target + 1)
    dp[0] = 1

    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] += dp[i - num]

    return dp[target]

# 使用示例
nums = [1, 1, 1, 1, 1]
S = 3
print(findTargetSumWays(nums, S))  # 输出: 5
```

#### 详解

- 这是一个转化为子集和问题的动态规划题目。目标是找到一个子集，其和等于 `(total_sum + S) / 2`。时间复杂度为 O(n*target)，空间复杂度为 O(target)。

---

### 560. 和为K的子数组（Subarray Sum Equals K）

**题目**：给定一个整数数组和一个整数 `k`，你需要找到该数组中和为 `k` 的连续子数组的个数。

**LeetCode 链接**：[Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

#### Python 实现

```python
def subarraySum(nums, k):
    count = 0
    current_sum = 0
    sum_dict = {0: 1}

    for num in nums:
        current_sum += num
        if current_sum - k in sum_dict:
            count += sum_dict[current_sum - k]
        sum_dict[current_sum] = sum_dict.get(current_sum, 0) + 1

    return count

# 使用示例
nums = [1, 1, 1]
k = 2
print(subarraySum(nums, k))  # 输出: 2
```

#### 详解

- 使用哈希表存储前缀和及其出现次数，通过前缀和的差值快速找到满足条件的子数组。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 581. 最短无序连续子数组（Shortest Unsorted Continuous Subarray）

**题目**：给定一个整数数组，找到一个连续子数组，使得如果对这个子数组进行升序排序，那么整个数组也是升序的。返回你需要排序的最短子数组的长度。

**LeetCode 链接**：[Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)

#### Python 实现

```python
def findUnsortedSubarray(nums):
    sorted_nums = sorted(nums)
    start, end = len(nums), 0

    for i in range(len(nums)):
        if nums[i] != sorted_nums[i]:
            start = min(start, i)
            end = max(end, i)

    return end - start + 1 if end >= start else 0

# 使用示例
nums = [2, 6, 4, 8, 10, 9, 15]
print(findUnsortedSubarray(nums))  # 输出: 5
```

#### 详解

- 对数组排序，然后比较原数组和排序后的数组，找到第一个和最后一个不同的位置。时间复杂度为 O(n log n)，空间复杂度为 O(n)。

---

### 647. 回文子串（Palindromic Substrings）

**题目**：给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

**LeetCode 链接**：[Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

#### Python 实现

```python
def countSubstrings(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    count = 0

    for i in range(n):
        dp[i][i] = True
        count += 1

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                if j - i == 1 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    count += 1

    return count

# 使用示例
s = "aaa"
print(countSubstrings(s))  # 输出: 6
```

#### 详解

- 使用动态规划，`dp[i][j]` 表示 `s[i:j+1]` 是否为回文子串。时间复杂度为 O(n^2)，空间复杂度为 O(n^2)。

---

### 695. 岛屿的最大面积（Max Area of Island）

**题目**：给定一个包含了一些 0 和 1 的非空二维数组，1 代表陆地，0 代表水域，计算并返回该二维数组中最大的岛屿面积。

**LeetCode 链接**：[Max Area of Island](https://leetcode.com/problems/max-area-of-island/)

#### Python 实现

```python
def maxAreaOfIsland(grid):
    def dfs(i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 0
        grid[i][j] = 0
        return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)

    max_area = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                max_area = max(max_area, dfs(i, j))

    return max_area

# 使用示例
grid = [
    [0,0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,1,0,1,0,0],
    [0,1,0,0,1,1,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0,0]
]
print(maxAreaOfIsland(grid))  # 输出: 6
```

#### 详解

- 使用 DFS 遍历岛屿，每次找到一个岛屿时，计算其面积并更新最大面积。时间复杂度为 O(n*m)，空间复杂度为 O(n*m)。

---

继续为你提供更多 LeetCode 经典题目及其 Python 实现，帮助你深入理解算法和数据结构。

### 739. 每日温度（Daily Temperatures）

**题目**：给定一个列表 `T`，表示每天的温度，返回一个列表 `answer`，其中 `answer[i]` 是你需要等待的天数，直到温度超过 `T[i]` 的天数。如果之后没有更高的温度，`answer[i]` = 0。

**LeetCode 链接**：[Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

#### Python 实现

```python
def dailyTemperatures(T):
    answer = [0] * len(T)
    stack = []

    for i in range(len(T)):
        while stack and T[i] > T[stack[-1]]:
            prev_day = stack.pop()
            answer[prev_day] = i - prev_day
        stack.append(i)

    return answer

# 使用示例
T = [73, 74, 75, 71, 69, 72, 76, 73]
print(dailyTemperatures(T))  # 输出: [1, 1, 4, 2, 1, 1, 0, 0]
```

#### 详解

- 使用单调栈，栈中存储的是温度数组的索引。当遇到比栈顶元素更高的温度时，弹出栈顶并计算等待天数。时间复杂度为 O(n)，空间复杂度为 O(n)。

---

### 581. 最短无序连续子数组（Shortest Unsorted Continuous Subarray）

**题目**：给定一个整数数组，找到一个连续子数组，使得如果对这个子数组进行升序排序，那么整个数组也是升序的。返回你需要排序的最短子数组的长度。

**LeetCode 链接**：[Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)

#### Python 实现

```python
def findUnsortedSubarray(nums):
    sorted_nums = sorted(nums)
    start, end = len(nums), 0

    for i in range(len(nums)):
        if nums[i] != sorted_nums[i]:
            start = min(start, i)
            end = max(end, i)

    return end - start + 1 if end >= start else 0

# 使用示例
nums = [2, 6, 4, 8, 10, 9, 15]
print(findUnsortedSubarray(nums))  # 输出: 5
```

#### 详解

- 对数组进行排序，然后比较原数组和排序后的数组，找到第一个和最后一个不相同的位置。时间复杂度为 O(n log n)，空间复杂度为 O(n)。

---

### 76. 最小覆盖子串（Minimum Window Substring）

**题目**：给定两个字符串 `s` 和 `t`，在 `s` 中找到最小的包含 `t` 所有字符的子串，并返回这个子串。如果不存在包含 `t` 所有字符的子串，则返回空字符串。

**LeetCode 链接**：[Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

#### Python 实现

```python
from collections import Counter

def minWindow(s, t):
    if not t or not s:
        return ""
    
    dict_t = Counter(t)
    required = len(dict_t)
    
    l, r = 0, 0
    formed = 0
    window_counts = {}
    
    ans = float("inf"), None, None
    
    while r < len(s):
        char = s[r]
        window_counts[char] = window_counts.get(char, 0) + 1        
        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        while l <= r and formed == required:
            char = s[l]
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)
            
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            l += 1    
        r += 1
    
    return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]

# 使用示例
s = "ADOBECODEBANC"
t = "ABC"
print(minWindow(s, t))  # 输出: "BANC"
```

#### 详解

- 使用双指针滑动窗口的方法，窗口从左到右扩展，当窗口包含所有需要的字符时，开始收缩窗口以找到最小覆盖。时间复杂度为 O(n)，空间复杂度为 O(t + s)。

---

### 312. 戳气球（Burst Balloons）

**题目**：给定 n 个气球，每个气球上都标有一个数字，气球破裂时你可以获得相邻两个未破裂气球的乘积。通过在恰当的顺序戳破气球，最大化你获得的分数。

**LeetCode 链接**：[Burst Balloons](https://leetcode.com/problems/burst-balloons/)

#### Python 实现

```python
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            for i in range(left + 1, right):
                dp[left][right] = max(dp[left][right], 
                                      nums[left] * nums[i] * nums[right] + dp[left][i] + dp[i][right])

    return dp[0][n - 1]

# 使用示例
nums = [3, 1, 5, 8]
print(maxCoins(nums))  # 输出: 167
```

#### 详解

- 动态规划，`dp[i][j]` 表示在 `nums[i:j+1]` 中能够得到的最大分数。时间复杂度为 O(n^3)，空间复杂度为 O(n^2)。

---

### 128. 最长连续序列（Longest Consecutive Sequence）

**题目**：给定一个未排序的整数数组，找出最长连续序列的长度。要求算法的时间复杂度为 O(n)。

**LeetCode 链接**：[Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

#### Python 实现

```python
def longestConsecutive(nums):
    num_set = set(nums)
    longest_streak = 0

    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num +=