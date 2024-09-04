# math 模块

### Python `math` 模块详解

`math` 模块提供了丰富的数学函数，用于执行各种数学运算，如三角函数、对数、幂运算等。该模块主要处理浮点数，但也提供了一些整数运算的支持。

#### 1. 导入模块

```python
import math
```

#### 2. 常用数学常量

- **数学常数 π（圆周率）**

```python
pi = math.pi
print(pi)  # 输出 3.141592653589793
```

- **数学常数 e（自然对数的底数）**

```python
e = math.e
print(e)  # 输出 2.718281828459045
```

#### 3. 基本数学函数

- **求绝对值**

```python
result = math.fabs(-5.3)
print(result)  # 输出 5.3
```

- **向上取整**

```python
result = math.ceil(4.1)
print(result)  # 输出 5
```

- **向下取整**

```python
result = math.floor(4.9)
print(result)  # 输出 4
```

- **四舍五入**

```python
result = round(4.5)
print(result)  # 输出 4 或 5 取决于 Python 版本和实现
```

- **幂运算**

```python
result = math.pow(2, 3)  # 计算 2^3
print(result)  # 输出 8.0
```

- **平方根**

```python
result = math.sqrt(16)
print(result)  # 输出 4.0
```

- **求和**

```python
result = math.fsum([0.1, 0.2, 0.3])
print(result)  # 输出 0.6000000000000001，比内建的 sum 准确性更高
```

#### 4. 三角函数

- **正弦**

```python
result = math.sin(math.pi / 2)
print(result)  # 输出 1.0
```

- **余弦**

```python
result = math.cos(math.pi)
print(result)  # 输出 -1.0
```

- **正切**

```python
result = math.tan(math.pi / 4)
print(result)  # 输出 1.0
```

- **反三角函数**

```python
result = math.asin(1)
print(result)  # 输出 π/2，约为 1.5707963267948966
```

- **角度转换**

```python
radians = math.radians(180)
degrees = math.degrees(math.pi)
print(radians)  # 输出 3.141592653589793
print(degrees)  # 输出 180.0
```

#### 5. 对数和指数函数

- **自然对数**

```python
result = math.log(math.e)
print(result)  # 输出 1.0
```

- **指定底的对数**

```python
result = math.log(100, 10)
print(result)  # 输出 2.0
```

- **10 为底的对数**

```python
result = math.log10(1000)
print(result)  # 输出 3.0
```

- **指数函数**

```python
result = math.exp(2)
print(result)  # 输出 e^2，约为 7.3890560989306495
```

#### 6. 高级数学函数

- **阶乘**

```python
result = math.factorial(5)
print(result)  # 输出 120
```

- **最大公约数**

```python
result = math.gcd(48, 64)
print(result)  # 输出 16
```

- **最小公倍数**

Python 3.9+ 中引入了 `math.lcm()` 函数：

```python
result = math.lcm(12, 18)
print(result)  # 输出 36
```

- **浮点数的分解**

```python
mantissa, exponent = math.frexp(8)
print(mantissa, exponent)  # 输出 (0.5, 4)
```

- **浮点数与整数和小数部分的分离**

```python
frac, whole = math.modf(3.14)
print(frac, whole)  # 输出 (0.14000000000000012, 3.0)
```

#### 7. 特殊函数

- **误差函数**

```python
result = math.erf(1)
print(result)  # 输出约为 0.8427007929497149
```

- **伽玛函数**

```python
result = math.gamma(5)
print(result)  # 输出 24.0
```

### 总结

`math` 模块是 Python 中处理数学运算的核心模块之一。它提供了常用的数学运算函数、三角函数、对数函数、幂运算以及特殊函数等。熟练掌握 `math` 模块能够帮助你在程序中处理复杂的数学问题。