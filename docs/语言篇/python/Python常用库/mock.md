# mock 模块

### Python `mock` 模块详解

`mock` 模块（从 Python 3.3 开始集成为 `unittest.mock`）用于在测试中替换和模拟对象的行为。它可以帮助模拟复杂的依赖关系，使单元测试更加独立和可控。

#### 1. 基本概念

`mock` 模块的核心类是 `Mock`，它允许你创建“伪造”的对象，这些对象可以被动态地赋予任何属性和行为。

```python
from unittest.mock import Mock

# 创建一个 Mock 对象
mock_obj = Mock()
```

#### 2. 模拟方法和属性

- **模拟方法**:

  ```python
  mock_obj.some_method.return_value = 42
  result = mock_obj.some_method()
  assert result == 42
  ```

- **模拟属性**:

  ```python
  mock_obj.some_attr = 'mocked attribute'
  assert mock_obj.some_attr == 'mocked attribute'
  ```

#### 3. 检查方法调用

- **检查方法是否被调用**:

  ```python
  mock_obj.some_method()
  mock_obj.some_method.assert_called()
  ```

- **检查方法调用次数**:

  ```python
  mock_obj.some_method()
  mock_obj.some_method()
  mock_obj.some_method.assert_called_once()  # 检查是否只调用了一次
  ```

- **检查方法调用的参数**:

  ```python
  mock_obj.some_method('hello', 'world')
  mock_obj.some_method.assert_called_with('hello', 'world')
  ```

- **检查方法的调用顺序**:

  ```python
  from unittest.mock import call

  mock_obj.some_method('first')
  mock_obj.some_method('second')
  
  mock_obj.some_method.assert_has_calls([call('first'), call('second')])
  ```

#### 4. 使用 `patch` 装饰器/上下文管理器

`patch` 是 `mock` 模块中最常用的功能之一，它可以临时替换对象或模块的属性。

- **使用 `patch` 装饰器**:

  ```python
  from unittest.mock import patch

  @patch('module.ClassName')
  def test_something(mock_class):
      instance = mock_class.return_value
      instance.method.return_value = 'mocked'
      
      result = instance.method()
      assert result == 'mocked'
  ```

- **使用 `patch` 上下文管理器**:

  ```python
  with patch('module.ClassName') as mock_class:
      instance = mock_class.return_value
      instance.method.return_value = 'mocked'
      
      result = instance.method()
      assert result == 'mocked'
  ```

#### 5. 使用 `patch.object` 替换对象属性

`patch.object` 允许替换特定对象的属性或方法。

```python
class MyClass:
    def method(self):
        return 'original'

my_obj = MyClass()

with patch.object(my_obj, 'method', return_value='mocked'):
    result = my_obj.method()
    assert result == 'mocked'
```

#### 6. `MagicMock` 类

`MagicMock` 是 `Mock` 的一个子类，它模拟了所有魔法方法（如 `__len__`、`__getitem__` 等）。

```python
from unittest.mock import MagicMock

mock_list = MagicMock()

# 模拟 list 的行为
mock_list.__len__.return_value = 10
assert len(mock_list) == 10

mock_list.__getitem__.return_value = 'mocked'
assert mock_list[0] == 'mocked'
```

#### 7. 自动补全属性和方法

`mock` 对象的任意属性和方法默认都是 `Mock` 对象。

```python
mock_obj = Mock()

# 动态生成属性和方法
result = mock_obj.anything.you.can.imagine()
assert isinstance(result, Mock)
```

#### 8. side_effect 用法

`side_effect` 可以指定调用方法时的副作用。它可以是一个函数、异常或一个返回值序列。

- **抛出异常**:

  ```python
  mock_obj.method.side_effect = ValueError("Error occurred")
  try:
      mock_obj.method()
  except ValueError as e:
      assert str(e) == "Error occurred"
  ```

- **根据输入参数返回不同值**:

  ```python
  def side_effect(arg):
      return arg * 2

  mock_obj.method.side_effect = side_effect
  assert mock_obj.method(3) == 6
  ```

- **返回不同的序列值**:

  ```python
  mock_obj.method.side_effect = [1, 2, 3]
  assert mock_obj.method() == 1
  assert mock_obj.method() == 2
  assert mock_obj.method() == 3
  ```

#### 9. mock_open 读取文件

`mock_open` 用于模拟文件读取操作，特别是在测试需要模拟 `open()` 函数时。

```python
from unittest.mock import mock_open, patch

m = mock_open(read_data='file content')

with patch('builtins.open', m):
    with open('file.txt') as f:
        result = f.read()
        assert result == 'file content'
```

#### 10. 重置 Mock 对象

`reset_mock()` 方法用于重置 `mock` 对象的所有调用记录和状态。

```python
mock_obj = Mock()
mock_obj.method()
mock_obj.reset_mock()
assert mock_obj.method.call_count == 0
```

### 总结

`unittest.mock` 模块为测试提供了强大的工具，使得你能够轻松地模拟对象、替换依赖并检查交互。它可以帮助你编写更健壮的单元测试，使测试代码与实际代码解耦。