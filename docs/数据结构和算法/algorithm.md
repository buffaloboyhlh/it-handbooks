# 算法

我们继续深入讨论常见算法的 Python 实现及详解。这些算法涵盖了排序、搜索、图算法、动态规划等多个领域。

### 1. 排序算法

排序算法是算法学习中的基础内容，常见的排序算法有冒泡排序、选择排序、插入排序、归并排序和快速排序等。

#### 1.1 冒泡排序（Bubble Sort）

冒泡排序是最基础的排序算法之一，通过多次比较相邻的元素并交换它们的位置，将较大的元素逐步"冒泡"到数组的最后。

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 使用示例
arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr))  # 输出: [11, 12, 22, 25, 34, 64, 90]
```

#### 1.2 选择排序（Selection Sort）

选择排序每次从未排序部分中选择最小的元素，并将其放在已排序部分的末尾。

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 使用示例
arr = [64, 25, 12, 22, 11]
print(selection_sort(arr))  # 输出: [11, 12, 22, 25, 64]
```

#### 1.3 插入排序（Insertion Sort）

插入排序通过逐一取出元素，并将其插入到已排序部分的适当位置。

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 使用示例
arr = [12, 11, 13, 5, 6]
print(insertion_sort(arr))  # 输出: [5, 6, 11, 12, 13]
```

#### 1.4 归并排序（Merge Sort）

归并排序是一种递归的排序算法，它将数组分成两个子数组，分别排序后再合并。时间复杂度为 O(n log n)。

```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr

# 使用示例
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # 输出: [3, 9, 10, 27, 38, 43, 82]
```

#### 1.5 快速排序（Quick Sort）

快速排序是分治算法的典型应用，它选择一个基准点，将数组划分为两部分，分别对这两部分进行排序。平均时间复杂度为 O(n log n)。

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 使用示例
arr = [10, 7, 8, 9, 1, 5]
print(quick_sort(arr))  # 输出: [1, 5, 7, 8, 9, 10]
```

### 2. 搜索算法

搜索算法包括线性搜索和二分搜索等。二分搜索适用于已排序的数组，时间复杂度为 O(log n)。

#### 2.1 线性搜索（Linear Search）

线性搜索逐一遍历数组，直到找到目标元素。

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 使用示例
arr = [2, 3, 4, 10, 40]
target = 10
print(linear_search(arr, target))  # 输出: 3
```

#### 2.2 二分搜索（Binary Search）

二分搜索通过不断将数组对半分来查找目标元素，要求数组必须有序。

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

# 使用示例
arr = [2, 3, 4, 10, 40]
target = 10
print(binary_search(arr, target))  # 输出: 3
```

### 3. 图算法

图算法包括广度优先搜索（BFS）、深度优先搜索（DFS）和最短路径算法等。

#### 3.1 广度优先搜索（BFS）

广度优先搜索通过层层展开搜索每个节点，适合用于查找最短路径等问题。

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# 使用示例
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
bfs(graph, 2)  # 输出: 2 0 3 1
```

#### 3.2 深度优先搜索（DFS）

深度优先搜索通过沿着一条路径深入搜索，适合用于遍历图中的所有节点。

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# 使用示例
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
dfs(graph, 2)  # 输出: 2 0 1 3
```

#### 3.3 Dijkstra 算法

Dijkstra 算法用于查找从源点到其他所有点的最短路径，适合加权图。

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# 使用示例
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}
print(dijkstra(graph, 'A'))  # 输出: {'A': 0, 'B': 1, 'C': 3, 'D': 4}
```

### 4. 动态规划（Dynamic Programming）

动态规划用于解决具有重叠子问题和最优子结构的问题，如斐波那契数列、最长公共子序列、背包问题等。

#### 4.1 斐波那契数列（Fibonacci）

```python
def fibonacci(n):
    dp = [0, 1]
    for i in range(2, n+1):
        dp.append(dp[i-1] + dp[i-2])
    return dp[n]

# 使用示例
print(fibonacci(10))  # 输出: 55
```

#### 4.2 背包问题（Knapsack）

```python
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0 for _ in range(W+1)] for _ in range(n+1)]
    
    for i in range(1, n+1):
        for w in range(1, W+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
                
    return dp[n][W]

# 使用示例
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
W = 7
print(knapsack(weights, values, W))  # 输出: 9
```

继续深入探讨更多经典算法的 Python 实现与详解，包括贪心算法、回溯算法、分治算法以及更多动态规划相关的问题。

### 5. 贪心算法（Greedy Algorithm）

贪心算法通过每一步都选择当前最优解，从而期望最终得到全局最优解。贪心算法的经典应用包括活动选择问题、最小生成树等。

#### 5.1 活动选择问题（Activity Selection Problem）

在一组活动中，每个活动有开始时间和结束时间。选择最多数量的活动，使它们之间不重叠。

```python
def activity_selection(activities):
    activities.sort(key=lambda x: x[1])  # 按活动结束时间排序
    selected_activities = [activities[0]]  # 选择第一个活动

    for i in range(1, len(activities)):
        if activities[i][0] >= selected_activities[-1][1]:  # 如果当前活动的开始时间 >= 上一个活动的结束时间
            selected_activities.append(activities[i])

    return selected_activities

# 使用示例
activities = [(1, 2), (3, 4), (0, 6), (5, 7), (8, 9), (5, 9)]
print(activity_selection(activities))  # 输出: [(1, 2), (3, 4), (5, 7), (8, 9)]
```

#### 5.2 零钱兑换问题（Coin Change Problem）

给定不同面值的硬币，求最少使用多少硬币来凑成一个特定金额。

```python
def coin_change(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        count += amount // coin
        amount %= coin
    if amount == 0:
        return count
    else:
        return -1  # 表示不能凑出准确金额

# 使用示例
coins = [1, 2, 5]
amount = 11
print(coin_change(coins, amount))  # 输出: 3 (5 + 5 + 1)
```

### 6. 回溯算法（Backtracking）

回溯算法用于解决组合优化问题，它尝试所有可能的解，当发现某一解不能达到目标时就回溯。

#### 6.1 N 皇后问题（N-Queens Problem）

在一个 N × N 的棋盘上放置 N 个皇后，使得它们不能互相攻击。

```python
def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(col):
            if board[row][i] == 1:
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        for i, j in zip(range(row, n), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        return True

    def solve(board, col):
        if col >= n:
            result.append(["".join("Q" if cell == 1 else "." for cell in row) for row in board])
            return True
        res = False
        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 1
                res = solve(board, col + 1) or res
                board[i][col] = 0
        return res

    result = []
    board = [[0 for _ in range(n)] for _ in range(n)]
    solve(board, 0)
    return result

# 使用示例
print(solve_n_queens(4))
# 输出:
# [
#  [".Q..", "...Q", "Q...", "..Q."],
#  ["..Q.", "Q...", "...Q", ".Q.."]
# ]
```

#### 6.2 子集求解（Subset Sum Problem）

在一个给定的集合中，判断是否存在子集，使得子集的元素和为目标值。

```python
def subset_sum(nums, target):
    def backtrack(start, current_sum):
        if current_sum == target:
            return True
        if current_sum > target:
            return False
        for i in range(start, len(nums)):
            if backtrack(i + 1, current_sum + nums[i]):
                return True
        return False

    return backtrack(0, 0)

# 使用示例
nums = [3, 34, 4, 12, 5, 2]
target = 9
print(subset_sum(nums, target))  # 输出: True (子集: [4, 5])
```

### 7. 动态规划（Dynamic Programming）

动态规划通过保存子问题的解来避免重复计算。经典问题有背包问题、最长公共子序列问题等。

#### 7.1 最长公共子序列（Longest Common Subsequence, LCS）

给定两个字符串，求它们的最长公共子序列长度。

```python
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# 使用示例
X = "AGGTAB"
Y = "GXTXAYB"
print(lcs(X, Y))  # 输出: 4 ("GTAB" 为最长公共子序列)
```

#### 7.2 编辑距离（Edit Distance）

编辑距离是指将一个字符串变为另一个字符串所需的最少编辑操作（插入、删除、替换）的次数。

```python
def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]

# 使用示例
str1 = "sunday"
str2 = "saturday"
print(edit_distance(str1, str2))  # 输出: 3
```

#### 7.3 最长递增子序列（Longest Increasing Subsequence, LIS）

找到一个数组中最长的递增子序列的长度。

```python
def lis(arr):
    n = len(arr)
    dp = [1] * n

    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# 使用示例
arr = [10, 22, 9, 33, 21, 50, 41, 60, 80]
print(lis(arr))  # 输出: 6 (最长递增子序列: [10, 22, 33, 50, 60, 80])
```

### 8. 分治算法（Divide and Conquer）

分治算法将问题分解为子问题，递归解决子问题，最终合并子问题的解。

#### 8.1 最大子数组和（Maximum Subarray Problem）

找到数组中具有最大和的连续子数组。

```python
def max_subarray(arr, low, high):
    if low == high:
        return arr[low]

    mid = (low + high) // 2

    left_sum = max_subarray(arr, low, mid)
    right_sum = max_subarray(arr, mid + 1, high)
    cross_sum = max_crossing_subarray(arr, low, mid, high)

    return max(left_sum, right_sum, cross_sum)

def max_crossing_subarray(arr, low, mid, high):
    left_sum = float('-inf')
    sum = 0
    for i in range(mid, low - 1, -1):
        sum += arr[i]
        if sum > left_sum:
            left_sum = sum

    right_sum = float('-inf')
    sum = 0
    for i in range(mid + 1, high + 1):
        sum += arr[i]
        if sum > right_sum:
            right_sum = sum

    return left_sum + right_sum

# 使用示例
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_subarray(arr, 0, len(arr) - 1))  # 输出: 6 (子数组: [4, -1, 2, 1])
```
