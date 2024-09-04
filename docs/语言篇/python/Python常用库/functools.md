# functools 模块

`functools` 模块是 Python 标准库的一部分，提供了一些用于函数操作的高阶函数和装饰器。这个模块主要用于提高函数的复用性和简化代码。

### 1. `functools` 常用功能

#### 1.1 `functools.reduce()`
- **功能**：对一个序列的元素进行累计操作。`reduce()` 函数会对序列中的元素依次执行传入的函数，并将结果继续与下一个元素进行累积计算，直到整个序列计算完毕。
- **参数**：
  - `function`：用于计算的函数，必须接收两个参数。
  - `iterable`：要进行累计操作的序列。
  - `initializer`：可选，初始值。
- **示例**：
  ```python
  from functools import reduce
  
  nums = [1, 2, 3, 4]
  result = reduce(lambda x, y: x * y, nums)
  print(result)  # 输出: 24
  ```

#### 1.2 `functools.partial()`
- **功能**：创建一个新的函数，将部分参数固定，然后返回这个新的函数。常用于函数参数的预设或简化函数调用。
- **参数**：
  - `func`：要部分应用的函数。
  - `*args`：固定的参数。
  - `**kwargs`：固定的关键字参数。
- **示例**：
  ```python
  from functools import partial
  
  def multiply(x, y):
      return x * y
  
  double = partial(multiply, 2)
  print(double(5))  # 输出: 10
  ```

#### 1.3 `functools.lru_cache()`
- **功能**：为函数结果提供缓存功能，用于优化函数的重复调用。`LRU` 表示“最近最少使用”缓存策略。
- **参数**：
  - `maxsize`：缓存的最大数量。默认值为 `128`，`None` 表示不限制缓存大小。
  - `typed`：布尔值，`True` 表示将不同类型的参数分别缓存，默认值为 `False`。
- **示例**：
  ```python
  from functools import lru_cache
  
  @lru_cache(maxsize=100)
  def fibonacci(n):
      if n < 2:
          return n
      return fibonacci(n-1) + fibonacci(n-2)
  
  print(fibonacci(10))  # 输出: 55
  ```

#### 1.4 `functools.wraps()`
- **功能**：用于装饰器内，将被装饰函数的元数据（如函数名、文档字符串）复制到装饰器函数中，保持函数的元数据不变。
- **参数**：
  - `wrapped`：被装饰的函数。
- **示例**：
  ```python
  from functools import wraps
  
  def my_decorator(f):
      @wraps(f)
      def wrapper(*args, **kwargs):
          print("Before function call")
          result = f(*args, **kwargs)
          print("After function call")
          return result
      return wrapper
  
  @my_decorator
  def say_hello():
      """This is a greeting function."""
      print("Hello!")
  
  say_hello()
  print(say_hello.__name__)  # 输出: say_hello
  print(say_hello.__doc__)   # 输出: This is a greeting function.
  ```

#### 1.5 `functools.total_ordering`
- **功能**：为类提供全套比较操作，只需要定义 `__lt__()` (小于) 和 `__eq__()` (等于) 方法，其他比较操作（`__le__()`、`__gt__()`、`__ge__()` 和 `__ne__()`）会自动生成。
- **示例**：
  ```python
  from functools import total_ordering
  
  @total_ordering
  class Student:
      def __init__(self, name, grade):
          self.name = name
          self.grade = grade
  
      def __eq__(self, other):
          return self.grade == other.grade
  
      def __lt__(self, other):
          return self.grade < other.grade
  
  alice = Student("Alice", 90)
  bob = Student("Bob", 85)
  
  print(alice > bob)  # 输出: True
  print(alice <= bob) # 输出: False
  ```

#### 1.6 `functools.singledispatch()`
- **功能**：实现基于参数类型的单分派泛型函数。通过注册不同的类型处理函数，实现对不同类型参数的不同处理。
- **示例**：
  ```python
  from functools import singledispatch
  
  @singledispatch
  def process(value):
      print("Processing as a generic type:", value)
  
  @process.register(int)
  def _(value):
      print("Processing as an integer:", value)
  
  @process.register(str)
  def _(value):
      print("Processing as a string:", value)
  
  process(10)   # 输出: Processing as an integer: 10
  process("hi") # 输出: Processing as a string: hi
  process([1, 2, 3]) # 输出: Processing as a generic type: [1, 2, 3]
  ```

### 2. 高级用法

#### 2.1 使用 `partial` 优化函数参数
`partial` 可以在编写回调函数时非常有用，可以提前绑定部分参数，从而减少复杂度。例如，在事件处理函数中：
```python
from functools import partial

def callback(x, y):
    print(f"Clicked on button {x} at position {y}")

button_callback = partial(callback, "Button1")
button_callback((100, 200))  # 输出: Clicked on button Button1 at position (100, 200)
```

#### 2.2 使用 `lru_cache` 提高性能
对于递归函数，`lru_cache` 可以显著提高性能，尤其是在计算斐波那契数列或求解动态规划问题时，缓存功能可避免重复计算。

### 3. 注意事项

- **`partial` 的使用**：当使用 `partial` 时，如果传递的参数数量不足，会返回一个新的函数，而不是立即调用。要确保传递足够的参数以正确调用函数。
- **`lru_cache` 的缓存管理**：要注意缓存的大小（`maxsize` 参数），避免占用过多内存。可以定期清理缓存或根据应用需求调整缓存策略。

`functools` 模块为 Python 提供了许多强大的工具，使得函数的操作更加灵活和高效。无论是优化性能、简化代码，还是实现复杂的功能需求，`functools` 都是不可或缺的工具。