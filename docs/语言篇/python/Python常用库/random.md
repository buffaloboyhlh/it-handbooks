# random 模块 

`random` 模块是 Python 标准库中的一个模块，提供了生成随机数、随机选择、打乱序列等功能。以下是 `random` 模块的详细介绍。

### 1. 基本随机数生成函数

#### 1.1 `random.random()`
- **功能**：生成一个介于 `[0.0, 1.0)` 之间的随机浮点数。
- **示例**：
  ```python
  import random
  print(random.random())  # 输出类似于 0.37444
  ```

#### 1.2 `random.randint(a, b)`
- **功能**：生成一个介于 `[a, b]` 之间的随机整数，包含 `a` 和 `b`。
- **示例**：
  ```python
  print(random.randint(1, 10))  # 输出介于 1 和 10 之间的整数
  ```

#### 1.3 `random.uniform(a, b)`
- **功能**：生成一个介于 `[a, b)` 之间的随机浮点数。
- **示例**：
  ```python
  print(random.uniform(1, 10))  # 输出介于 1.0 和 10.0 之间的浮点数
  ```

### 2. 随机选择和打乱序列

#### 2.1 `random.choice(seq)`
- **功能**：从序列 `seq` 中随机选择一个元素。
- **示例**：
  ```python
  print(random.choice(['apple', 'banana', 'cherry']))  # 随机选择一个水果
  ```

#### 2.2 `random.choices(seq, weights=None, k=1)`
- **功能**：从序列 `seq` 中随机选择 `k` 个元素，返回一个列表。可以通过 `weights` 参数设置权重。
- **示例**：
  ```python
  print(random.choices(['apple', 'banana', 'cherry'], k=2))  # 随机选择两个水果
  ```

#### 2.3 `random.sample(population, k)`
- **功能**：从集合或序列 `population` 中随机选择 `k` 个不重复的元素，返回一个列表。
- **示例**：
  ```python
  print(random.sample([1, 2, 3, 4, 5], 3))  # 随机选择 3 个不同的数字
  ```

#### 2.4 `random.shuffle(x)`
- **功能**：就地打乱序列 `x`。
- **示例**：
  ```python
  nums = [1, 2, 3, 4, 5]
  random.shuffle(nums)
  print(nums)  # 输出打乱后的序列
  ```

### 3. 随机分布函数

#### 3.1 `random.gauss(mu, sigma)`
- **功能**：生成一个符合正态分布（高斯分布）的随机浮点数，`mu` 为均值，`sigma` 为标准差。
- **示例**：
  ```python
  print(random.gauss(0, 1))  # 输出符合正态分布的随机数
  ```

#### 3.2 `random.expovariate(lambd)`
- **功能**：生成一个符合指数分布的随机浮点数，`lambd` 是分布的参数（`1/平均值`）。
- **示例**：
  ```python
  print(random.expovariate(1.5))  # 输出符合指数分布的随机数
  ```

#### 3.3 `random.betavariate(alpha, beta)`
- **功能**：生成一个符合 Beta 分布的随机浮点数，`alpha` 和 `beta` 是形状参数。
- **示例**：
  ```python
  print(random.betavariate(0.5, 0.5))  # 输出符合 Beta 分布的随机数
  ```

### 4. 随机种子

#### 4.1 `random.seed(a=None, version=2)`
- **功能**：初始化随机数生成器，可以用相同的种子产生相同的随机数序列。
- **示例**：
  ```python
  random.seed(10)
  print(random.random())  # 输出将是固定的
  ```

### 5. 示例应用

#### 5.1 掷骰子
```python
def roll_dice():
    return random.randint(1, 6)

print(roll_dice())  # 模拟掷骰子
```

#### 5.2 模拟抽奖
```python
participants = ['Alice', 'Bob', 'Charlie', 'David']
winner = random.choice(participants)
print(f"The winner is {winner}")  # 随机选择获奖者
```

### 6. 注意事项

- `random` 模块生成的随机数是伪随机数，依赖于初始的种子值，因此并不适用于需要高安全性的场景，如加密。
- 对于安全性要求高的随机数生成，可以使用 `secrets` 模块。

`random` 模块提供了丰富的随机数生成和操作功能，能够满足大多数场景下的需求。在使用时，合理选择函数和参数可以实现高效且易读的代码。