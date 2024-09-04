# numpy 笔记

NumPy 是 Python 中用于科学计算的核心库，它提供了支持多维数组对象、各种派生对象（如掩码数组和矩阵）、以及大量的数学操作和统计函数。NumPy 是很多高级数据处理库的基础，例如 Pandas、SciPy 和 Scikit-learn。以下是一个详细的 NumPy 教程，帮助你了解如何使用这个库进行数据处理和计算。

## 创建数组

### 从列表创建数组
```
import numpy as np

# 从列表创建一维数组
a = np.array([1, 2, 3, 4, 5])
print(a)

# 从列表创建二维数组
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
```

### 使用 arange 和 linspace 创建数组
```
# 使用 arange 创建数组
a = np.arange(0, 10, 2)  # 从0到10，步长为2
print(a)

# 使用 linspace 创建数组
b = np.linspace(0, 1, 5)  # 在0和1之间创建5个均匀分布的值
print(b)
```

### 创建特殊数组
```
# 创建全零数组
a = np.zeros((2, 3))
print(a)

# 创建全一数组
b = np.ones((3, 4))
print(b)

# 创建单位矩阵
c = np.eye(4)
print(c)

# 创建随机数组
d = np.random.random((2, 2))
print(d)
```

###  数组属性
```
a = np.array([[1, 2, 3], [4, 5, 6]])

# 数组形状
print("Shape:", a.shape)

# 数组维度
print("Dimensions:", a.ndim)

# 数组大小 (元素的总个数)
print("Size:", a.size) # 输出6

# 数组元素数据类型
print("Data type:", a.dtype)
```

## 数组操作
### 数组切片和索引

#### 一维数组的切片
对于一维数组，切片的基本语法是 start:stop:step，其中：

	•	start：切片的起始索引（包含）。
	•	stop：切片的结束索引（不包含）。
	•	step：切片的步长（默认为 1）。


```
import numpy as np

# 创建一维数组
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 基本切片
print(a[2:7])        # 输出: [2 3 4 5 6]
print(a[:5])         # 输出: [0 1 2 3 4]
print(a[5:])         # 输出: [5 6 7 8 9]
print(a[::2])        # 输出: [0 2 4 6 8]
print(a[1:8:2])      # 输出: [1 3 5 7]
```

#### 二维数组的切片
对于二维数组，切片操作可以在每个维度上独立进行。切片的基本语法是 array[row_start:row_stop:row_step, col_start:col_stop:col_step]

```
import numpy as np

# 创建二维数组
a = np.array([[0, 1, 2, 3, 4],
              [5, 6, 7, 8, 9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24]])

# 提取子数组
print(a[1:4, 2:5])  # 输出: [[ 7  8  9]
                    #       [12 13 14]
                    #       [17 18 19]]

# 提取第二列
print(a[:, 1])      # 输出: [ 1  6 11 16 21]

# 提取第三行
print(a[2, :])      # 输出: [10 11 12 13 14]

# 提取第三行的第2到第4个元素
print(a[2, 1:4])    # 输出: [11 12 13]

# 步长切片
print(a[::2, ::2])  # 输出: [[ 0  2  4]
                    #       [10 12 14]
                    #       [20 22 24]]
```

#### 高维数组的切片
对于更高维度的数组，切片操作可以推广到每个维度。

```
import numpy as np

# 创建三维数组
a = np.array([[[0, 1, 2], [3, 4, 5]],
              [[6, 7, 8], [9, 10, 11]],
              [[12, 13, 14], [15, 16, 17]]])

# 提取子数组
print(a[1:, :, :])   # 提取从第二个“平面”开始的所有数据

# 提取特定平面的特定部分
print(a[:, 1, :])    # 提取每个平面的第二行

# 提取特定平面中的某一块
print(a[1, :, 1:3])  # 输出: [[ 7  8]
                     #       [10 11]]
```

####  结合布尔索引的切片
```
import numpy as np

# 创建一维数组
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 条件筛选
print(a[a > 5])   # 输出: [6 7 8 9]

# 条件筛选并切片
print(a[(a > 3) & (a < 8)])  # 输出: [4 5 6 7]
```

### 形状变换

```
nums = np.array([[1, 2, 3], [4, 5, 6]])
# 重塑数组
b = nums.reshape(3, 2)
print(b)
# 展平数组
c = b.flatten()
print(c)
# 重塑数组
c.resize((2,3))
print(c)
```

## 数组运算
NumPy 允许对数组进行各种算术运算、逻辑运算、统计运算等。
### 基本算术运算
```
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# 数组加法
print(a + b)

# 数组减法
print(a - b)

# 数组乘法
print(a * b)

# 数组除法
print(a / b)
```

### 广播机制
当两个数组的形状不同时，NumPy 会自动适应（广播）较小的数组以匹配较大的数组，从而进行运算。

NumPy 的广播机制（Broadcasting）是一种强大的功能，允许不同形状的数组在算术运算时进行自动扩展，以便它们能够兼容。广播机制的目的是在数组操作中避免创建不必要的多维数组，从而提高计算效率。

#### 广播机制的基本原则
	1.	如果两个数组的维度数不同，形状较小的数组将在前面补充一维，直到两个数组的维度数相同。

	2.	沿每个维度，数组的大小要么是相同的，要么其中一个数组的大小为 1，要么会发生广播。

	3.	在任何维度上，如果两个数组的大小都不相等且都不为 1，则不能进行广播，操作会抛出错误。

#### 广播机制示例
##### 示例 1：标量与数组之间的广播
```
import numpy as np

# 创建一个一维数组
a = np.array([1, 2, 3])

# 标量与数组相加
b = a + 5
print(b)  # 输出: [6 7 8]
```

##### 示例 2：二维数组与一维数组之间的广播
```
import numpy as np

# 创建一个二维数组
a = np.array([[1, 2, 3], [4, 5, 6]])

# 创建一个一维数组
b = np.array([1, 2, 3])

# 二维数组与一维数组相加
c = a + b
print(c)
# 输出:
# [[2 4 6]
#  [5 7 9]]
```
在这个示例中，数组 b 被广播以匹配数组 a 的形状，然后逐元素相加。

##### 示例 3：不同形状的二维数组之间的广播
```
import numpy as np

# 创建一个二维数组
a = np.array([[1, 2, 3], [4, 5, 6]])

# 创建一个形状为 (2, 1) 的二维数组
b = np.array([[10], [20]])

# 广播并相加
c = a + b
print(c)
# 输出:
# [[11 12 13]
#  [24 25 26]]
```
在这个示例中，数组 b 被广播以匹配数组 a 的形状，其中 b 的形状从 (2, 1) 被广播到 (2, 3)。

### 广播的具体工作原理

假设你有两个数组 A 和 B，它们的形状分别是 (m, n, p) 和 (n, p)。要使这两个数组进行广播，它们的形状需要按以下步骤进行调整：

	•	首先，数组 B 会在其最前面补充一维，变为 (1, n, p)。

	•	然后，B 会沿第一个维度复制，变为 (m, n, p)，这样它们就可以进行逐元素的运算了。


###  广播的应用

广播机制非常有用，尤其是在处理多维数据时。它可以让你以简洁的代码进行复杂的操作，而无需手动扩展数组维度。广播通常用于：

	•	标量与数组之间的操作：如将数组中的所有元素加上一个常数。

	•	低维数组与高维数组之间的操作：如将一个一维数组加到二维数组的每一行或列上。

	•	数据归一化：如在矩阵的每一行或列中减去均值。

## 统计函数
```
a = np.array([[1, 2, 3], [4, 5, 6]])

# 最小值
print("Min:", a.min())

# 最大值
print("Max:", a.max())

# 和
print("Sum:", a.sum())

# 平均值
print("Mean:", a.mean())

# 标准差
print("Standard Deviation:", a.std())
```

## 数组的高级操作
### 数组拼接和分割

```
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# 数组拼接
c = np.vstack((a, b))  # 垂直堆叠
print(c)

d = np.hstack((a, b.T))  # 水平堆叠
print(d)

# 数组分割
e = np.hsplit(a, 2)  # 水平分割
print(e)
```

### 条件筛选
```
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 筛选大于5的元素
print(a[a > 5])

# 筛选偶数元素
print(a[a % 2 == 0])
```

## 线性代数操作
NumPy 提供了多种线性代数操作，如矩阵乘法、逆矩阵、特征值和特征向量等。

### 矩阵乘法或点积

```
import numpy as np

# 创建两个二维数组
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = np.dot(A, B)
print(C)
# 输出:
# [[19 22]
#  [43 50]]
```

### np.matmul() 或 @ 运算符
矩阵乘法。与 np.dot() 类似，但更适用于矩阵乘法。

```
import numpy as np

# 创建两个二维数组
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = np.matmul(A, B)
print(C)

# 或使用 @ 运算符
D = A @ B
print(D)
```

### np.transpose() 或 array.T
矩阵转置。

```
import numpy as np

# 创建一个二维数组
A = np.array([[1, 2, 3], [4, 5, 6]])

# 矩阵转置
B = np.transpose(A)
print(B)
# 输出:
# [[1 4]
#  [2 5]
#  [3 6]]

# 或者使用属性 T
C = A.T
print(C)
```

###  np.linalg.inv()
计算矩阵的逆矩阵。前提是矩阵必须是方阵且可逆。

```
import numpy as np

# 创建一个二维数组（方阵）
A = np.array([[1, 2], [3, 4]])

# 计算逆矩阵
A_inv = np.linalg.inv(A)
print(A_inv)
# 输出:
# [[-2.   1. ]
#  [ 1.5 -0.5]]
```

### np.linalg.det()
计算方阵的行列式。

```
import numpy as np

# 创建一个二维数组（方阵）
A = np.array([[1, 2], [3, 4]])

# 计算行列式
det_A = np.linalg.det(A)
print(det_A)  # 输出: -2.0000000000000004
```

### np.linalg.eig()
计算方阵的特征值和特征向量。

```
import numpy as np

# 创建一个二维数组（方阵）
A = np.array([[1, 2], [3, 4]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)
# 输出: 特征值: [-0.37228132  5.37228132]
print("特征向量:\n", eigenvectors)
# 输出:
# 特征向量:
# [[-0.82456484 -0.41597356]
#  [ 0.56576746 -0.90937671]]
```

### np.linalg.solve()
解线性方程组 Ax = b，其中 A 是系数矩阵，b 是常数向量。

```
import numpy as np

# 创建系数矩阵 A 和常数向量 b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# 解线性方程组 Ax = b
x = np.linalg.solve(A, b)
print(x)  # 输出: [2. 3.]
```

### np.linalg.norm()
计算矩阵或向量的范数。

```
import numpy as np

# 创建一个向量
v = np.array([3, 4])

# 计算向量的二范数（欧几里得范数）
norm_v = np.linalg.norm(v)
print(norm_v)  # 输出: 5.0

# 创建一个二维数组（矩阵）
A = np.array([[1, 2], [3, 4]])

# 计算矩阵的二范数
norm_A = np.linalg.norm(A)
print(norm_A)  # 输出: 5.477225575051661
```

### np.linalg.qr()
对矩阵进行 QR 分解。

```
import numpy as np

# 创建一个二维数组（矩阵）
A = np.array([[1, 2], [3, 4], [5, 6]])

# 进行 QR 分解
Q, R = np.linalg.qr(A)
print("Q:\n", Q)
print("R:\n", R)
# 输出:
# Q:
# [[-0.16903085  0.89708523]
#  [-0.50709255  0.27602622]
#  [-0.84515425 -0.34503278]]
# R:
# [[-5.91607978 -7.43735744]
#  [ 0.          0.82807867]]
```
###  np.linalg.svd()
对矩阵进行奇异值分解（SVD）。

```
import numpy as np

# 创建一个二维数组（矩阵）
A = np.array([[1, 2, 3], [4, 5, 6]])

# 进行奇异值分解
U, S, V = np.linalg.svd(A)
print("U:\n", U)
print("S:\n", S)
print("V:\n", V)
# 输出:
# U:
# [[-0.3863177  -0.92236578]
#  [-0.92236578  0.3863177 ]]
# S:
# [9.508032   0.77286964]
# V:
# [[-0.42866713 -0.56630692 -0.7039467 ]
#  [ 0.80596391  0.11238241 -0.5811991 ]
#  [ 0.40824829 -0.81649658  0.40824829]]
```

## 随机数生成
```
# 生成0到1之间的随机数
print(np.random.rand(5))

# 生成指定范围内的随机整数
print(np.random.randint(0, 10, size=5))

# 生成标准正态分布的随机数
print(np.random.randn(5))

# 洗牌数组
a = np.array([1, 2, 3, 4, 5])
np.random.shuffle(a)
print(a)
```

### 读取和保存数据
NumPy 可以方便地读取和保存数据。

```
a = np.array([[1, 2, 3], [4, 5, 6]])

# 保存数组到文件
np.save('my_array.npy', a)

# 从文件读取数组
b = np.load('my_array.npy')
print(b)

# 保存为文本文件
np.savetxt('my_array.txt', a)

# 从文本文件读取数组
c = np.loadtxt('my_array.txt')
print(c)
```


