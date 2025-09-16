# Python `typing` 库使用教程

`typing` 是 Python 3.5+ 引入的标准库，用于提供类型提示（Type Hints）支持。它允许开发者明确指定变量、函数参数和返回值的类型，提高代码可读性、可维护性，并支持静态类型检查。

---

## 1. 基本类型注解

### 1.1 基础类型
```python
from typing import List, Dict, Set, Tuple, Optional

# 基本类型注解
name: str = "Alice"
age: int = 25
height: float = 1.75
is_student: bool = True

# 容器类型注解
names: List[str] = ["Alice", "Bob"]
scores: Dict[str, int] = {"Alice": 90, "Bob": 85}
unique_numbers: Set[int] = {1, 2, 3}
coordinates: Tuple[float, float] = (1.0, 2.0)
```

### 1.2 函数注解
```python
def greet(name: str) -> str:
    return f"Hello, {name}"

def calculate_average(scores: List[float]) -> float:
    return sum(scores) / len(scores)
```

---

## 2. 复合类型

### 2.1 联合类型 (Union)
表示值可以是多种类型之一：
```python
from typing import Union

def process_value(value: Union[int, str]) -> None:
    if isinstance(value, int):
        print(f"Integer: {value}")
    else:
        print(f"String: {value}")
```

### 2.2 可选类型 (Optional)
表示值可以是某种类型或 `None`：
```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    # 可能返回用户名，也可能返回 None
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)
```

### 2.3 任意类型 (Any)
表示任意类型（禁用类型检查）：
```python
from typing import Any

def process_data(data: Any) -> Any:
    # 可以接受和返回任何类型
    return data
```

---

## 3. 泛型和类型变量

### 3.1 类型变量 (TypeVar)
创建可重用的类型变量：
```python
from typing import TypeVar, List

T = TypeVar('T')  # 任意类型
Number = TypeVar('Number', int, float)  # 限制为数字类型

def first_element(items: List[T]) -> T:
    return items[0]

def add(a: Number, b: Number) -> Number:
    return a + b
```

### 3.2 泛型类
```python
from typing import TypeVar, Generic, List

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop()
```

---

## 4. 特殊类型

### 4.1 字面量类型 (Literal)
表示值必须是特定的字面量：
```python
from typing import Literal

def set_direction(direction: Literal["north", "south", "east", "west"]) -> None:
    print(f"Moving {direction}")

set_direction("north")  # 正确
set_direction("up")     # 类型检查器会报错
```

### 4.2 回调函数类型 (Callable)
注解函数参数：
```python
from typing import Callable

def process_numbers(numbers: List[int], processor: Callable[[int], float]) -> List[float]:
    return [processor(n) for n in numbers]

# 使用
result = process_numbers([1, 2, 3], lambda x: x * 1.5)
```

### 4.3 迭代器和生成器
```python
from typing import Iterator, Generator

def count_up_to(n: int) -> Iterator[int]:
    i = 1
    while i <= n:
        yield i
        i += 1

def number_generator() -> Generator[int, None, str]:
    for i in range(5):
        yield i
    return "Done"
```

---

## 5. 高级特性

### 5.1 类型别名
创建复杂的类型别名：
```python
from typing import Dict, List, Tuple

# 类型别名
UserId = int
UserName = str
UserDict = Dict[UserId, UserName]
Point = Tuple[float, float]

def process_users(users: UserDict) -> List[UserId]:
    return list(users.keys())
```

### 5.2 重载 (Overload)
处理函数多种参数模式：
```python
from typing import overload, Union

@overload
def process(x: int) -> str: ...
@overload
def process(x: str) -> int: ...

def process(x: Union[int, str]) -> Union[str, int]:
    if isinstance(x, int):
        return str(x)
    else:
        return int(x)
```

### 5.3 自引用类型
注解引用自身类的类型：
```python
from typing import List, Optional

class TreeNode:
    def __init__(self, value: int, children: Optional[List['TreeNode']] = None):
        self.value = value
        self.children = children or []
```

---

## 6. 与 Pydantic 结合使用

```python
from typing import List, Optional
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    name: str
    email: Optional[str] = None
    addresses: List[Address] = []
```

---

## 7. 静态类型检查

使用 `mypy` 进行静态类型检查：

1. 安装 mypy：
```bash
pip install mypy
```

2. 检查代码：
```bash
mypy your_script.py
```

3. 常见配置（在 `mypy.ini` 或 `pyproject.toml` 中）：
```ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

---

## 8. 最佳实践

1. **渐进式类型化**：从关键代码开始，逐步添加类型注解
2. **合理使用 Any**：尽量避免使用 `Any`，只在必要时使用
3. **保持一致**：团队应遵循相同的类型注解规范
4. **利用工具**：使用 IDE 和 mypy 等工具辅助类型检查
5. **文档化复杂类型**：为复杂的类型别名添加注释

---

## 9. 总结

`typing` 库为 Python 提供了强大的类型提示功能，主要优势包括：
- 提高代码可读性和可维护性
- 在开发阶段捕获类型错误
- 改善 IDE 的智能提示和自动补全
- 便于重构和代码维护

通过合理使用 `typing` 库，你可以编写更加健壮和可靠的 Python 代码。

官方文档：[https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)