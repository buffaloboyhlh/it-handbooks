# 数据结构

### Python 数据结构详解

数据结构是计算机存储、组织数据的方式。在 Python 中，有许多内置的数据结构可以帮助我们高效地处理和操作数据。本文将详细讲解 Python 中常用的数据结构，包括列表、元组、字典、集合，以及一些常见的算法和高级数据结构。

### 1. 列表（List）

列表是 Python 中最常用的数据结构之一，它是一个可变的、有序的集合，能够存储任意类型的数据。

#### 1.1 列表的创建

```python
# 创建一个空列表
my_list = []

# 创建包含多个元素的列表
my_list = [1, 2, 3, "Python", True]
```

#### 1.2 列表的基本操作

- **添加元素**：使用 `append()`、`insert()` 或 `extend()` 方法。
  
```python
my_list.append(4)  # 在末尾添加元素
my_list.insert(2, "New")  # 在索引 2 处插入元素
```

- **删除元素**：使用 `remove()` 或 `pop()` 方法。
  
```python
my_list.remove(2)  # 移除第一个匹配的元素
my_list.pop(0)  # 移除并返回索引为 0 的元素
```

- **列表切片**：列表可以通过切片操作来访问或修改子列表。
  
```python
sub_list = my_list[1:3]  # 获取索引 1 到 2 的子列表
```

- **其他操作**：

```python
len(my_list)  # 获取列表长度
my_list.sort()  # 排序
my_list.reverse()  # 反转列表
```

#### 1.3 列表推导式

列表推导式是生成列表的一种简洁方式。

```python
squares = [x**2 for x in range(10)]
print(squares)  # 输出：[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### 2. 元组（Tuple）

元组和列表类似，但元组是**不可变**的。一旦创建就不能修改，这使得元组适合作为常量数据的容器。

#### 2.1 元组的创建

```python
my_tuple = (1, 2, 3, "Python", False)
```

#### 2.2 元组的操作

元组的操作与列表相似，但由于它是不可变的，不能进行增删改操作。你可以使用索引和切片访问元组元素。

```python
print(my_tuple[1])  # 输出：2
sub_tuple = my_tuple[1:3]  # 获取子元组
```

#### 2.3 元组解包

元组支持解包，可以将元组的值赋给多个变量。

```python
a, b, c = (1, 2, 3)
print(a, b, c)  # 输出：1 2 3
```

### 3. 字典（Dictionary）

字典是一种**无序的键值对集合**，可以通过键（Key）快速查找对应的值（Value）。键必须是不可变类型，如字符串、数字或元组。

#### 3.1 字典的创建

```python
my_dict = {"name": "Alice", "age": 25, "city": "New York"}
```

#### 3.2 字典的基本操作

- **访问和修改**：

```python
print(my_dict["name"])  # 输出：Alice
my_dict["age"] = 30  # 修改字典中的值
```

- **添加和删除**：

```python
my_dict["country"] = "USA"  # 添加新的键值对
my_dict.pop("city")  # 删除键为 "city" 的键值对
```

- **字典的方法**：

```python
keys = my_dict.keys()  # 获取所有键
values = my_dict.values()  # 获取所有值
items = my_dict.items()  # 获取所有键值对
```

### 4. 集合（Set）

集合是**无序**的、**不重复**的元素集合，适用于去重和集合运算。

#### 4.1 集合的创建

```python
my_set = {1, 2, 3, 4}
empty_set = set()  # 创建一个空集合
```

#### 4.2 集合的基本操作

- **添加和删除元素**：

```python
my_set.add(5)  # 添加元素
my_set.remove(3)  # 删除元素，若不存在则抛出错误
```

- **集合运算**：并集、交集、差集。

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

union_set = set1 | set2  # 并集
intersection_set = set1 & set2  # 交集
difference_set = set1 - set2  # 差集
```

### 5. 队列和堆栈

#### 5.1 队列（Queue）

队列是一种先进先出（FIFO）的数据结构。在 Python 中可以使用 `collections.deque` 来实现队列。

```python
from collections import deque

queue = deque([1, 2, 3])
queue.append(4)  # 入队
queue.popleft()  # 出队
```

#### 5.2 堆栈（Stack）

堆栈是一种后进先出（LIFO）的数据结构，可以通过列表模拟堆栈操作。

```python
stack = [1, 2, 3]
stack.append(4)  # 压栈
stack.pop()  # 出栈
```

### 6. 高级数据结构

#### 6.1 堆（Heap）

堆是一种特殊的树形结构，常用于实现优先队列。Python 的 `heapq` 库提供了堆的实现。

```python
import heapq

heap = [1, 3, 5, 7, 9]
heapq.heapify(heap)  # 将列表转换为堆
heapq.heappush(heap, 4)  # 将元素加入堆中
smallest = heapq.heappop(heap)  # 弹出最小的元素
```

#### 6.2 有序字典（OrderedDict）

`OrderedDict` 是字典的子类，保留了键值对的插入顺序。

```python
from collections import OrderedDict

ordered_dict = OrderedDict()
ordered_dict["apple"] = 1
ordered_dict["banana"] = 2
```

#### 6.3 默认字典（defaultdict）

`defaultdict` 是字典的子类，在访问不存在的键时可以提供默认值。

```python
from collections import defaultdict

default_dict = defaultdict(int)
default_dict["count"] += 1
```

### 7. 数据结构与算法常用的 Python 库

- **`bisect`**：用于有序列表的二分查找。
  
```python
import bisect

lst = [1, 2, 3, 5]
bisect.insort(lst, 4)  # 将 4 插入有序列表中
```

- **`collections.Counter`**：计数器，用于统计元素出现的频率。

```python
from collections import Counter

counter = Counter(["apple", "banana", "apple", "orange"])
print(counter)  # 输出：Counter({'apple': 2, 'banana': 1, 'orange': 1})
```

### 8. 总结

Python 提供了丰富的内置数据结构，如列表、元组、字典、集合等，同时还提供了许多高效的算法和数据结构库，如 `heapq`、`collections` 等。这些数据结构和库为开发者提供了处理不同场景的强大工具。理解和熟练掌握这些数据结构将极大提高代码的效率和性能。


# 数据结构基于Python实现

在 Python 中，可以使用类和内置的数据结构来实现一些常见的数据结构，例如栈（Stack）、队列（Queue）、链表（Linked List）、树（Tree）和图（Graph）。以下是使用 Python 实现这些数据结构的代码示例。

### 1. 栈（Stack）

栈是一种**后进先出（LIFO）**的数据结构。可以使用 Python 的列表来实现栈。

```python
class Stack:
    def __init__(self):
        self.stack = []

    def is_empty(self):
        return len(self.stack) == 0

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return "Stack is empty"

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            return "Stack is empty"

    def size(self):
        return len(self.stack)

# 使用示例
s = Stack()
s.push(1)
s.push(2)
print(s.pop())  # 输出：2
print(s.peek())  # 输出：1
```

### 2. 队列（Queue）

队列是一种**先进先出（FIFO）**的数据结构。可以使用 `collections.deque` 来实现队列。

```python
from collections import deque

class Queue:
    def __init__(self):
        self.queue = deque()

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.popleft()
        else:
            return "Queue is empty"

    def size(self):
        return len(self.queue)

# 使用示例
q = Queue()
q.enqueue(1)
q.enqueue(2)
print(q.dequeue())  # 输出：1
print(q.size())  # 输出：1
```

### 3. 链表（Linked List）

链表是一种线性数据结构，其中元素存储在节点中，每个节点包含数据和指向下一个节点的指针。

#### 3.1 单向链表（Singly Linked List）

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def append(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        print(elements)

# 使用示例
sll = SinglyLinkedList()
sll.append(1)
sll.append(2)
sll.append(3)
sll.display()  # 输出：[1, 2, 3]
```

#### 3.2 双向链表（Doubly Linked List）

双向链表中，每个节点有两个指针，一个指向前一个节点，一个指向后一个节点。

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def append(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
            new_node.prev = current

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        print(elements)

# 使用示例
dll = DoublyLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)
dll.display()  # 输出：[1, 2, 3]
```

### 4. 树（Tree）

树是一种层级结构，每个节点有零个或多个子节点。常见的树有二叉树、二叉搜索树等。

#### 4.1 二叉树（Binary Tree）

```python
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        if self.root is None:
            self.root = TreeNode(data)
        else:
            self._insert(self.root, data)

    def _insert(self, node, data):
        if data < node.data:
            if node.left is None:
                node.left = TreeNode(data)
            else:
                self._insert(node.left, data)
        else:
            if node.right is None:
                node.right = TreeNode(data)
            else:
                self._insert(node.right, data)

    def inorder_traversal(self, node):
        if node:
            self.inorder_traversal(node.left)
            print(node.data, end=" ")
            self.inorder_traversal(node.right)

# 使用示例
bt = BinaryTree()
bt.insert(10)
bt.insert(5)
bt.insert(15)
bt.inorder_traversal(bt.root)  # 输出：5 10 15
```

#### 4.2 二叉搜索树（Binary Search Tree）

二叉搜索树是一种特殊的二叉树，其中左子树的所有节点都小于根节点，右子树的所有节点都大于根节点。

```python
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        if self.root is None:
            self.root = TreeNode(data)
        else:
            self._insert(self.root, data)

    def _insert(self, node, data):
        if data < node.data:
            if node.left is None:
                node.left = TreeNode(data)
            else:
                self._insert(node.left, data)
        else:
            if node.right is None:
                node.right = TreeNode(data)
            else:
                self._insert(node.right, data)

    def inorder_traversal(self, node):
        if node:
            self.inorder_traversal(node.left)
            print(node.data, end=" ")
            self.inorder_traversal(node.right)

# 使用示例
bst = BinarySearchTree()
bst.insert(10)
bst.insert(5)
bst.insert(15)
bst.inorder_traversal(bst.root)  # 输出：5 10 15
```

### 5. 图（Graph）

图是一种由节点（顶点）和边组成的结构。可以用邻接表或邻接矩阵来表示图。

#### 5.1 邻接表表示图

```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def bfs(self, start):
        visited = set()
        queue = [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                print(vertex, end=" ")
                visited.add(vertex)
                queue.extend([node for node in self.graph[vertex] if node not in visited])

# 使用示例
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

g.bfs(2)  # 输出：2 0 3 1
```

### 6. 堆（Heap）

堆是一种特殊的树形结构，可以用来实现优先队列。Python 提供了 `heapq` 库来支持堆的操作。

```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, element):
        heapq.heappush(self.heap, element)

    def extract_min(self):
        return heapq.heappop(self.heap)

# 使用示例
heap = MinHeap()
heap.insert(3)
heap.insert(2)
heap.insert(15)
print(heap.extract_min())  # 输出：2
```

### 7. 优先队列（Priority Queue）

优先队列是一种特殊的队列，每个元素都有一个优先级，优先级高的元素优先出队。可以使用 Python 的 `heapq` 模块来实现优先队列。

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, item, priority):
        heapq.heappush(self.queue, (priority, item))

    def dequeue(self):
        if not self.is_empty():
            return heapq.heappop(self.queue)[1]
        else:
            return "Priority queue is empty"

# 使用示例
pq = PriorityQueue()
pq.enqueue("task1", 1)
pq.enqueue("task2", 2)
pq.enqueue("task3", 0)

print(pq.dequeue())  # 输出：task3
print(pq.dequeue())  # 输出：task1
```

### 8. 哈希表（Hash Table）

哈希表是一种根据键值对进行数据存储的数据结构，可以实现快速查找、插入和删除。Python 内置的 `dict` 就是哈希表的实现。

#### 8.1 自定义哈希表实现

```python
class HashTable:
    def __init__(self):
        self.size = 10
        self.table = [[] for _ in range(self.size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        hash_key = self._hash(key)
        for pair in self.table[hash_key]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[hash_key].append([key, value])

    def get(self, key):
        hash_key = self._hash(key)
        for pair in self.table[hash_key]:
            if pair[0] == key:
                return pair[1]
        return None

    def delete(self, key):
        hash_key = self._hash(key)
        for i, pair in enumerate(self.table[hash_key]):
            if pair[0] == key:
                del self.table[hash_key][i]
                return

# 使用示例
ht = HashTable()
ht.insert("name", "Alice")
ht.insert("age", 25)

print(ht.get("name"))  # 输出：Alice
print(ht.get("age"))  # 输出：25

ht.delete("age")
print(ht.get("age"))  # 输出：None
```

### 9. 图的深度优先搜索（DFS）

图的深度优先搜索是一种用于遍历或搜索图的算法。可以使用递归来实现。

```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def dfs(self, v, visited=None):
        if visited is None:
            visited = set()
        visited.add(v)
        print(v, end=" ")
        for neighbor in self.graph[v]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

# 使用示例
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

g.dfs(2)  # 输出：2 0 1 3
```

### 10. 图的广度优先搜索（BFS）

图的广度优先搜索是一种层级遍历图的算法，通常使用队列来实现。

```python
from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                print(vertex, end=" ")
                visited.add(vertex)
                queue.extend([node for node in self.graph[vertex] if node not in visited])

# 使用示例
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

g.bfs(2)  # 输出：2 0 3 1
```

### 11. Trie（字典树）

Trie 是一种树形数据结构，常用于存储字符串，如实现自动补全功能。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_end_of_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return current.is_end_of_word

    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return True

# 使用示例
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))  # 输出：True
print(trie.search("app"))  # 输出：False
print(trie.starts_with("app"))  # 输出：True
trie.insert("app")
print(trie.search("app"))  # 输出：True
```

### 12. 并查集（Union-Find）

并查集是一种常用的数据结构，用于处理不相交集合的合并与查询操作。并查集可以高效地解决连通性问题。

```python
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

# 使用示例
uf = UnionFind(10)
uf.union(1, 2)
uf.union(2, 3)
print(uf.find(1))  # 输出：1
print(uf.find(3))  # 输出：1
```

继续深入 Python 实现数据结构的讨论，我们可以进一步探讨一些高级数据结构和算法，提升理解的深度和应用范围。

### 13. AVL 树（平衡二叉搜索树）

AVL 树是一种自平衡的二叉搜索树。每个节点的两个子树的高度差最大为 1，因此可以保证搜索、插入、删除的时间复杂度为 O(log n)。

```python
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def get_height(self, node):
        if not node:
            return 0
        return node.height

    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def right_rotate(self, y):
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1

        return x

    def left_rotate(self, x):
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1

        return y

    def insert(self, root, key):
        if not root:
            return TreeNode(key)
        elif key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))

        balance = self.get_balance(root)

        # LL Case
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)

        # RR Case
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)

        # LR Case
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)

        # RL Case
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def pre_order(self, root):
        if root:
            print(f"{root.key}", end=" ")
            self.pre_order(root.left)
            self.pre_order(root.right)

# 使用示例
avl = AVLTree()
root = None
root = avl.insert(root, 10)
root = avl.insert(root, 20)
root = avl.insert(root, 30)
root = avl.insert(root, 40)
root = avl.insert(root, 50)
root = avl.insert(root, 25)

avl.pre_order(root)  # 输出：30 20 10 25 40 50
```

### 14. 红黑树

红黑树是一种平衡二叉搜索树，每个节点存储颜色（红色或黑色），并通过旋转和颜色交换来保持平衡。红黑树广泛用于实现关联数组和优先队列。

```python
class RedBlackNode:
    def __init__(self, data, color="R"):
        self.data = data
        self.color = color  # "R" for red, "B" for black
        self.left = None
        self.right = None
        self.parent = None

class RedBlackTree:
    def __init__(self):
        self.TNULL = RedBlackNode(0, "B")  # Sentinel node
        self.root = self.TNULL

    def pre_order_helper(self, node):
        if node != self.TNULL:
            print(node.data, end=" ")
            self.pre_order_helper(node.left)
            self.pre_order_helper(node.right)

    def insert(self, key):
        new_node = RedBlackNode(key)
        new_node.parent = None
        new_node.data = key
        new_node.left = self.TNULL
        new_node.right = self.TNULL
        new_node.color = "R"  # Node must be red initially

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            if new_node.data < x.data:
                x = x.left
            else:
                x = x.right

        new_node.parent = y
        if y is None:
            self.root = new_node
        elif new_node.data < y.data:
            y.left = new_node
        else:
            y.right = new_node

        if new_node.parent is None:
            new_node.color = "B"
            return

        if new_node.parent.parent is None:
            return

        self.fix_insert(new_node)

    def fix_insert(self, k):
        while k.parent.color == "R":
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == "R":
                    u.color = "B"
                    k.parent.color = "B"
                    k.parent.parent.color = "R"
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = "B"
                    k.parent.parent.color = "R"
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right

                if u.color == "R":
                    u.color = "B"
                    k.parent.color = "B"
                    k.parent.parent.color = "R"
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = "B"
                    k.parent.parent.color = "R"
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = "B"

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def pre_order(self):
        self.pre_order_helper(self.root)

# 使用示例
rbt = RedBlackTree()
rbt.insert(7)
rbt.insert(3)
rbt.insert(18)
rbt.insert(10)
rbt.insert(22)
rbt.insert(8)
rbt.insert(11)

rbt.pre_order()  # 输出：7 3 10 8 11 18 22
```

### 15. 跳表（Skip List）

跳表是一种支持快速查找、插入和删除的概率性数据结构。它是链表的升级版，通过多个层次来加速操作。

```python
import random

class Node:
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level):
        self.max_level = max_level
        self.header = Node(-1, max_level)
        self.level = 0

    def random_level(self):
        level = 0
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level

    def insert(self, value):
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        level = self.random_level()
        if level > self.level:
            for i in range(self.level + 1, level + 1):
                update[i] = self.header
            self.level = level

        new_node = Node(value, level)
        for i in range(level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def search(self, value):
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]

        current = current.forward[0]
        if current and current.value == value:
            return True
        return False

    def display_list(self):
        print("Skip List Levels:")
        for i in range(self.level + 1):
            current = self.header.forward[i]
            print(f"Level {i}: ", end=" ")
            while current:
                print(current.value, end=" ")
                current = current.forward[i]
            print("")

# 使用示例
skip_list = SkipList(3)
skip_list.insert(3)
skip_list.insert(6)
skip_list.insert(7)
skip_list.insert(9)
skip_list.insert(12)
skip_list.insert(19)
skip_list.insert(17)

skip_list.display_list()
# 输出：
# Level 0: 3 6 7 9 12 17 19 
# Level 1: 3 6 9 12 17 
# Level 2: 6 12 
```

我们可以继续讨论一些高级的数据结构和算法，进一步扩展 Python 实现的范围和复杂度。这部分内容将介绍更多在实际应用中非常有用的数据结构。

### 16. 哈希表（Hash Table）

哈希表是一种键值对数据结构，通过一个哈希函数来计算数据的存储位置。它能够在平均 O(1) 的时间内完成插入、查找和删除操作。

#### 哈希表的 Python 实现：

```python
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * self.size

    def hash_function(self, key):
        return key % self.size

    def insert(self, key, value):
        hash_index = self.hash_function(key)
        if self.table[hash_index] is None:
            self.table[hash_index] = [(key, value)]
        else:
            self.table[hash_index].append((key, value))

    def search(self, key):
        hash_index = self.hash_function(key)
        if self.table[hash_index] is not None:
            for kvp in self.table[hash_index]:
                if kvp[0] == key:
                    return kvp[1]
        return None

    def delete(self, key):
        hash_index = self.hash_function(key)
        if self.table[hash_index] is not None:
            for i, kvp in enumerate(self.table[hash_index]):
                if kvp[0] == key:
                    del self.table[hash_index][i]
                    return True
        return False

# 使用示例
ht = HashTable(10)
ht.insert(1, 'apple')
ht.insert(11, 'banana')
ht.insert(21, 'orange')

print(ht.search(11))  # 输出: 'banana'
print(ht.delete(21))  # 输出: True
print(ht.search(21))  # 输出: None
```

### 17. 堆（Heap）

堆是一种特殊的二叉树数据结构，主要用于实现优先队列。堆分为最大堆和最小堆，在最大堆中，父节点总是大于或等于子节点；在最小堆中，父节点总是小于或等于子节点。

#### 最小堆的 Python 实现：

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, index):
        return (index - 1) // 2

    def insert(self, key):
        self.heap.append(key)
        current = len(self.heap) - 1
        while current > 0 and self.heap[current] < self.heap[self.parent(current)]:
            self.heap[current], self.heap[self.parent(current)] = self.heap[self.parent(current)], self.heap[current]
            current = self.parent(current)

    def min_heapify(self, index):
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self.min_heapify(smallest)

    def extract_min(self):
        if len(self.heap) == 0:
            return None
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.min_heapify(0)
        return root

# 使用示例
min_heap = MinHeap()
min_heap.insert(3)
min_heap.insert(2)
min_heap.insert(15)
min_heap.insert(5)
min_heap.insert(4)
min_heap.insert(45)

print(min_heap.extract_min())  # 输出: 2
print(min_heap.extract_min())  # 输出: 3
```

### 18. 图（Graph）

图是由节点（顶点）和边组成的数据结构，可以表示为有向图或无向图。图可以用于建模网络、路径查找、关系等问题。

#### 使用邻接列表实现无向图：

```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append(v)
        self.graph[v].append(u)

    def bfs(self, start):
        visited = set()
        queue = [start]
        visited.add(start)

        while queue:
            vertex = queue.pop(0)
            print(vertex, end=" ")

            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        print(start, end=" ")

        for neighbor in self.graph[start]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

# 使用示例
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)

print("BFS:")
g.bfs(0)  # 输出: 0 1 2 3 4

print("\nDFS:")
g.dfs(0)  # 输出: 0 1 2 3 4
```

### 19. 拓扑排序（Topological Sort）

拓扑排序是用于有向无环图（DAG）的排序算法，它会生成一个线性顺序，使得对于每对边 `u -> v`，顶点 `u` 都排在顶点 `v` 之前。

#### 拓扑排序的 Python 实现（基于 DFS）：

```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def topological_sort_util(self, v, visited, stack):
        visited.add(v)

        if v in self.graph:
            for neighbor in self.graph[v]:
                if neighbor not in visited:
                    self.topological_sort_util(neighbor, visited, stack)

        stack.append(v)

    def topological_sort(self):
        visited = set()
        stack = []

        for vertex in list(self.graph):
            if vertex not in visited:
                self.topological_sort_util(vertex, visited, stack)

        print(stack[::-1])  # 输出逆序栈

# 使用示例
g = Graph()
g.add_edge(5, 2)
g.add_edge(5, 0)
g.add_edge(4, 0)
g.add_edge(4, 1)
g.add_edge(2, 3)
g.add_edge(3, 1)

g.topological_sort()  # 输出: [5, 4, 2, 3, 1, 0]
```

### 20. Trie 树（前缀树）

Trie 树是一种用于快速搜索前缀的树状数据结构，常用于字典树实现，适合处理字符串查询问题。

#### Trie 树的 Python 实现：

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

    def starts_with(self, prefix):
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
print(trie.search("app"))    # 输出: False
print(trie.starts_with("app"))  # 输出: True
trie.insert("app")
print(trie.search("app"))    # 输出: True
```

