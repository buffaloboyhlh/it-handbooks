# NumPy 教程

本文详细讲解 `NumPy` 的各项功能，包括从基础到高级的操作，帮助你全面掌握这个科学计算库。每个部分都会深入解释其工作原理和具体使用方法。

---

#### 目录

1. 什么是 NumPy
2. 安装 NumPy
3. NumPy 基础操作
    - 创建数组
    - 数组的数据类型
    - 数组的基本属性
4. 数组运算
    - 数组的广播机制
    - 数学运算
    - 统计运算
5. 数组索引与切片
    - 基本索引
    - 布尔索引
    - 花式索引
6. 数组变形和拼接
    - 数组变形
    - 数组拼接与分割
7. 高级操作
    - 数组的排序
    - 条件逻辑操作
    - 线性代数
    - 随机数生成
8. NumPy 性能优化
    - 使用内建函数优化
    - 向量化操作
9. NumPy 实战案例

---

### 1. 什么是 NumPy

`NumPy`（Numerical Python）是 Python 语言的一个开源库，主要用于科学计算。它提供了高效操作大规模多维数组和矩阵的功能，还包括大量数学函数库，是数据分析、机器学习等领域的基础工具。

NumPy 的核心是 `ndarray`，也称为多维数组。`ndarray` 能高效存储和操作同类数据，非常适合用于数值计算。

---

### 2. 安装 NumPy

安装 NumPy 只需一行命令：

```bash
pip install numpy
```

如果安装成功，你可以通过以下代码来验证：

```python
import numpy as np
print(np.__version__)
```

---

### 3. NumPy 基础操作

#### 创建数组

`array()` 是创建 NumPy 数组的最基本函数，可以将 Python 列表、元组等转换为数组。

```python
import numpy as np

# 创建一维数组
arr1d = np.array([1, 2, 3, 4])

# 创建二维数组
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# 创建三维数组
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

此外，NumPy 提供了许多创建数组的函数：

- `zeros(shape)`：创建全为 0 的数组。
- `ones(shape)`：创建全为 1 的数组。
- `empty(shape)`：创建未初始化的数组。
- `arange(start, stop, step)`：创建等间隔的数组。
- `linspace(start, stop, num)`：生成从 `start` 到 `stop` 的 `num` 个等间隔数值。

```python
arr_zeros = np.zeros((3, 3))  # 创建 3x3 的全 0 数组
arr_ones = np.ones((2, 2))    # 创建 2x2 的全 1 数组
arr_arange = np.arange(0, 10, 2)  # 从 0 开始，间隔 2，直到 10（不包括 10）
arr_linspace = np.linspace(0, 1, 5)  # 生成从 0 到 1 的 5 个等间隔数
```

#### 数组的数据类型

NumPy 数组的数据类型可以通过 `dtype` 参数指定。例如，`dtype='int32'` 创建一个 32 位整数类型的数组。你还可以使用 `astype()` 方法转换数组的数据类型。

```python
arr = np.array([1, 2, 3], dtype='float64')  # 指定数据类型为 float64
print(arr.dtype)  # 输出 float64

# 转换数据类型
arr_int = arr.astype('int32')
print(arr_int)
```

#### 数组的基本属性

每个 `ndarray` 对象都有一些基本属性：

- **`ndim`**：数组的维度数（几维数组）。
- **`shape`**：数组的形状（每个维度的大小）。
- **`size`**：数组中的元素总数。
- **`dtype`**：数组中元素的数据类型。
- **`itemsize`**：每个元素所占字节数。

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.ndim)    # 输出 2，表示二维数组
print(arr.shape)   # 输出 (2, 3)，表示两行三列
print(arr.size)    # 输出 6，表示数组中有 6 个元素
print(arr.dtype)   # 输出 int64，表示数据类型为 64 位整数
print(arr.itemsize)  # 输出 8，每个元素占 8 个字节
```

---

### 4. 数组运算

#### 数组的广播机制

广播（Broadcasting）允许不同形状的数组在进行运算时自动扩展为相同形状，使得数组运算更加高效。广播的原则是沿着对齐的维度进行扩展，较小的数组会被自动重复。

```python
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([1, 2, 3])

# arr2 会被广播为 [[1, 2, 3], [1, 2, 3]]
result = arr1 + arr2
print(result)
```

#### 数学运算

NumPy 提供了丰富的数学运算功能，包括加、减、乘、除、幂运算、对数等。运算可以逐元素应用于数组。

```python
arr = np.array([1, 2, 3, 4])
print(arr + 2)     # 每个元素加 2 -> [3 4 5 6]
print(arr * 3)     # 每个元素乘 3 -> [3 6 9 12]
print(arr ** 2)    # 每个元素平方 -> [1 4 9 16]
```

#### 统计运算

- **`sum()`**：计算数组元素的和。
- **`mean()`**：计算平均值。
- **`max()` 和 `min()`**：返回数组中的最大值和最小值。
- **`std()`**：计算标准差。
- **`argmax()` 和 `argmin()`**：返回最大值和最小值的索引。

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(arr))          # 求和 -> 21
print(np.mean(arr))         # 平均值 -> 3.5
print(np.max(arr, axis=0))  # 按列求最大值 -> [4 5 6]
print(np.min(arr, axis=1))  # 按行求最小值 -> [1 4]
```

---

### 5. 数组索引与切片

#### 基本索引

NumPy 支持与 Python 列表类似的索引方式。可以通过索引访问和修改数组元素。

```python
arr = np.array([1, 2, 3, 4, 5])

# 访问第一个和最后一个元素
print(arr[0])    # 1
print(arr[-1])   # 5

# 修改第二个元素
arr[1] = 10
print(arr)       # [ 1 10  3  4  5 ]
```

对于多维数组，可以通过多个索引访问不同维度的元素。

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

print(arr2d[0, 1])  # 访问第一行第二列的元素 -> 2
print(arr2d[1, -1]) # 访问最后一行最后一列的元素 -> 6
```

#### 切片

NumPy 数组的切片操作允许获取数组的一部分。切片的语法为 `arr[start:stop:step]`。

```python
arr = np.array([1, 2, 3, 4, 5])

# 获取从索引 1 到 3 的元素
print(arr[1:4])  # [2 3 4]

# 每隔一个元素获取
print(arr[::2])  # [1 3 5]
```

多维数组的切片可以对每个维度分别操作。

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# 获取第二列的所有行
print(arr2d[:, 1])  # [2 5]

# 获取第一行的前两列
print(arr2d[0, :2])  # [1 2]
```

#### 布尔索引

布尔索

引用于通过条件筛选数组元素。

```python
arr = np.array([1, 2, 3, 4, 5])

# 返回大于 3 的元素
print(arr[arr > 3])  # [4 5]
```

#### 花式索引

花式索引是通过一组索引值获取数组中的元素。

```python
arr = np.array([10, 20, 30, 40, 50])

# 获取第 0, 2, 4 个元素
print(arr[[0, 2, 4]])  # [10 30 50]
```

---

### 6. 数组变形和拼接

#### 数组变形

`reshape()` 函数可以改变数组的形状，而不改变数组的数据。在变形时，变换后的数组大小必须与原数组的大小相同。

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 将一维数组变形为 2x3 的二维数组
reshaped = arr.reshape((2, 3))
print(reshaped)
```

输出：

```
[[1 2 3]
 [4 5 6]]
```

如果你不确定某一维的大小，可以用 `-1` 让 NumPy 自动计算。

```python
reshaped = arr.reshape((-1, 2))
print(reshaped)  # NumPy 会自动计算行数
```

`ravel()` 和 `flatten()` 用于将多维数组展平成一维数组：

- `ravel()`：返回的是原数组的视图，修改它会影响原数组。
- `flatten()`：返回的是数组的拷贝，修改它不会影响原数组。

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# ravel 展平成一维数组
arr1d_ravel = arr2d.ravel()
print(arr1d_ravel)  # [1 2 3 4 5 6]

# flatten 展平成一维数组
arr1d_flatten = arr2d.flatten()
print(arr1d_flatten)  # [1 2 3 4 5 6]
```

#### 数组拼接与分割

- **拼接**：可以使用 `concatenate()` 函数沿指定轴将多个数组拼接在一起。
- **堆叠**：`vstack()` 和 `hstack()` 分别用于垂直和水平堆叠数组。
- **分割**：`split()` 可以将数组分割成多个子数组。

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])

# 沿行拼接数组
concatenated = np.concatenate((arr1, arr2), axis=0)
print(concatenated)
```

垂直、水平拼接：

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 垂直堆叠
vstacked = np.vstack((arr1, arr2))
print(vstacked)

# 水平堆叠
hstacked = np.hstack((arr1, arr2))
print(hstacked)
```

数组分割：

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 将数组水平分割为两部分
split_arr = np.hsplit(arr, 3)
print(split_arr)
```

---

### 7. 高级操作

#### 数组排序

`sort()` 函数可以对数组进行排序。可以指定沿哪个轴进行排序：

```python
arr = np.array([[3, 2, 1], [6, 5, 4]])

# 沿着列排序
arr.sort(axis=0)
print(arr)

# 沿着行排序
arr.sort(axis=1)
print(arr)
```

`argsort()` 返回排序后的元素在原数组中的索引。

#### 条件逻辑操作

NumPy 提供了 `where()` 函数，可以根据条件筛选元素。它返回满足条件的元素的索引位置或根据条件选择的数组。

```python
arr = np.array([1, 2, 3, 4, 5])

# 返回满足条件的索引
index = np.where(arr > 3)
print(index)

# 使用条件返回新数组
result = np.where(arr > 3, arr, 0)
print(result)
```

#### 线性代数

NumPy 提供了用于线性代数运算的 `linalg` 模块：

- **`dot()`**：计算矩阵乘法。
- **`inv()`**：计算矩阵的逆。
- **`det()`**：计算矩阵的行列式。

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 矩阵乘法
dot_product = np.dot(arr1, arr2)
print(dot_product)

# 矩阵的逆
inv_arr = np.linalg.inv(arr1)
print(inv_arr)

# 矩阵的行列式
det_arr = np.linalg.det(arr1)
print(det_arr)
```

#### 随机数生成

NumPy 的 `random` 模块提供了生成随机数的工具：

- **`rand()`**：生成 0 到 1 之间的均匀分布随机数。
- **`randn()`**：生成均值为 0，标准差为 1 的正态分布随机数。
- **`randint()`**：生成指定范围内的随机整数。
- **`choice()`**：从数组中随机选择元素。

```python
# 生成 2x3 的随机数
rand_arr = np.random.rand(2, 3)
print(rand_arr)

# 生成 5 个正态分布随机数
randn_arr = np.random.randn(5)
print(randn_arr)

# 生成 10 到 20 之间的随机整数
randint_arr = np.random.randint(10, 20, size=5)
print(randint_arr)
```

---

### 8. NumPy 性能优化

#### 使用内建函数优化

尽量使用 NumPy 的内建函数而不是 Python 的循环，内建函数是在 C 语言实现的，性能更高。

```python
arr = np.arange(1e7)

# NumPy 内建求和函数
%time np.sum(arr)

# Python 的循环求和
%time sum(arr)
```

#### 向量化操作

向量化操作是指用数组而不是循环进行运算。这样可以大幅提升性能。

```python
# 向量化运算
arr = np.array([1, 2, 3, 4])
print(arr * 2)  # 每个元素乘以 2
```

---

### 9. NumPy 实战案例

#### 案例1：数据归一化

将数据缩放到指定范围，如 0 到 1：

```python
data = np.random.randint(0, 100, size=(3, 3))

# 归一化公式： (x - min) / (max - min)
normalized_data = (data - data.min()) / (data.max() - data.min())
print(normalized_data)
```

#### 案例2：多项式拟合

使用 NumPy 进行多项式拟合：

```python
x = np.linspace(-5, 5, 100)
y = 3 * x**2 + 2 * x + 1 + np.random.randn(100) * 5

# 拟合二次多项式
p = np.polyfit(x, y, 2)

# 计算拟合曲线的 y 值
y_fit = np.polyval(p, x)
```

---

通过学习这些内容，您将能够灵活使用 NumPy 进行各种数据操作和科学计算。