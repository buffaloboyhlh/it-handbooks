# json 模块

`json` 模块是 Python 标准库中的一个模块，用于处理 JSON（JavaScript Object Notation）数据格式。JSON 是一种轻量级的数据交换格式，常用于 Web 应用程序中传输数据。Python 的 `json` 模块提供了对 JSON 数据的解析和生成功能。

### 1. 基本功能

#### 1.1 `json.dumps()`
- **功能**：将 Python 对象转换为 JSON 字符串。
- **参数**：
  - `obj`：要转换的 Python 对象。
  - `indent`：指定缩进级别，生成格式化的 JSON 字符串。
  - `separators`：元组，指定分隔符（项之间的分隔符，键值对之间的分隔符）。
  - `sort_keys`：是否按字典键排序（默认 `False`）。
- **示例**：
  ```python
  import json
  
  data = {'name': 'Alice', 'age': 25, 'city': 'New York'}
  json_str = json.dumps(data, indent=4, sort_keys=True)
  print(json_str)
  ```

#### 1.2 `json.loads()`
- **功能**：将 JSON 字符串解析为 Python 对象。
- **参数**：
  - `s`：要解析的 JSON 字符串。
  - `cls`：自定义 JSON 解码器类（默认 `None`）。
  - `object_hook`：用于解析字典的自定义函数。
  - `parse_float`：用于解析 JSON 数字为浮点数的自定义函数。
  - `parse_int`：用于解析 JSON 数字为整数的自定义函数。
- **示例**：
  ```python
  import json
  
  json_str = '{"name": "Alice", "age": 25, "city": "New York"}'
  data = json.loads(json_str)
  print(data)  # 输出: {'name': 'Alice', 'age': 25, 'city': 'New York'}
  ```

#### 1.3 `json.dump()`
- **功能**：将 Python 对象转换为 JSON 格式，并将结果写入文件。
- **参数**：
  - `obj`：要转换的 Python 对象。
  - `fp`：文件对象，用于写入 JSON 数据。
  - `indent`、`separators`、`sort_keys`：同 `json.dumps()`。
- **示例**：
  ```python
  import json
  
  data = {'name': 'Alice', 'age': 25, 'city': 'New York'}
  with open('data.json', 'w') as f:
      json.dump(data, f, indent=4)
  ```

#### 1.4 `json.load()`
- **功能**：从文件读取 JSON 数据并解析为 Python 对象。
- **参数**：
  - `fp`：文件对象，包含 JSON 数据。
  - `cls`、`object_hook`、`parse_float`、`parse_int`：同 `json.loads()`。
- **示例**：
  ```python
  import json
  
  with open('data.json', 'r') as f:
      data = json.load(f)
  print(data)  # 输出: {'name': 'Alice', 'age': 25, 'city': 'New York'}
  ```

### 2. 高级用法

#### 2.1 自定义 JSON 编码器
- **功能**：通过继承 `json.JSONEncoder` 类，自定义对象的 JSON 编码。
- **示例**：
  ```python
  import json
  
  class Person:
      def __init__(self, name, age):
          self.name = name
          self.age = age
  
  class PersonEncoder(json.JSONEncoder):
      def default(self, obj):
          if isinstance(obj, Person):
              return {'name': obj.name, 'age': obj.age}
          return super().default(obj)
  
  person = Person('Alice', 25)
  json_str = json.dumps(person, cls=PersonEncoder)
  print(json_str)  # 输出: {"name": "Alice", "age": 25}
  ```

#### 2.2 自定义 JSON 解码器
- **功能**：通过使用 `object_hook` 参数，自定义 JSON 字符串解析为 Python 对象。
- **示例**：
  ```python
  import json
  
  def dict_to_person(d):
      return Person(d['name'], d['age']) if 'name' in d and 'age' in d else d
  
  json_str = '{"name": "Alice", "age": 25}'
  person = json.loads(json_str, object_hook=dict_to_person)
  print(person.name, person.age)  # 输出: Alice 25
  ```

#### 2.3 处理复杂对象
- **功能**：使用 `default` 参数或自定义编码器处理复杂的 Python 对象（如日期、类实例）。
- **示例**：
  ```python
  import json
  from datetime import datetime
  
  class DateTimeEncoder(json.JSONEncoder):
      def default(self, obj):
          if isinstance(obj, datetime):
              return obj.isoformat()
          return super().default(obj)
  
  now = datetime.now()
  json_str = json.dumps({'time': now}, cls=DateTimeEncoder)
  print(json_str)  # 输出: {"time": "2024-08-28T12:34:56.789012"}
  ```

### 3. 常见错误处理

#### 3.1 `JSONDecodeError`
- **场景**：在解析 JSON 字符串时，如果格式不正确，会引发 `JSONDecodeError`。
- **示例**：
  ```python
  import json
  
  invalid_json_str = '{"name": "Alice", "age": 25'  # 缺少右括号
  try:
      data = json.loads(invalid_json_str)
  except json.JSONDecodeError as e:
      print(f"JSON decode error: {e}")
  ```

### 4. 应用场景

- **数据序列化和反序列化**：将 Python 对象序列化为 JSON 字符串，用于网络传输或存储；反序列化 JSON 字符串为 Python 对象，用于后续处理。
- **配置文件**：使用 JSON 格式存储和读取应用程序的配置数据。
- **API 数据交互**：与基于 REST 的 Web API 进行数据交互时，通常使用 JSON 格式。

`json` 模块为 Python 提供了强大的 JSON 处理功能，广泛应用于数据交换、配置管理和 Web 开发等场景。