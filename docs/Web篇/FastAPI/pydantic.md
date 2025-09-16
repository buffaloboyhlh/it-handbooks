# Pydantic 使用教程


Pydantic 是一个强大的 Python 数据验证和设置管理库，它使用 Python 类型注解来定义数据模型的结构和验证规则。以下是 Pydantic 的详细教程，涵盖从基础到高级的用法。

---

## 1. 安装 Pydantic

```bash
pip install pydantic
```

---

## 2. 基础用法

### 2.1 定义模型
通过继承 `BaseModel` 并定义字段类型来创建模型：

```python
from pydantic import BaseModel
from typing import Optional, List

class User(BaseModel):
    id: int
    name: str = "John Doe"
    email: Optional[str] = None
    friends: List[int] = []
```

### 2.2 创建实例
```python
user = User(id=1, name="Alice", email="alice@example.com")
print(user)  # 输出: id=1 name='Alice' email='alice@example.com' friends=[]
```

### 2.3 自动类型转换
Pydantic 会自动转换类型（如字符串数字转整数）：
```python
user = User(id="123", name="Bob")  # id 会被自动转换为 int
```

### 2.4 验证错误处理
如果数据无效，会抛出 `ValidationError`：
```python
from pydantic import ValidationError

try:
    User(id="not_an_int")
except ValidationError as e:
    print(e.json())  # 输出详细的错误信息
```

---

## 3. 字段类型与验证

### 3.1 常用类型
- 基础类型：`int`, `str`, `float`, `bool`
- 复合类型：`List`, `Dict`, `Set`, `Optional`
- 特殊类型：`EmailStr`, `UrlStr`, `IPvAnyAddress`（需安装 `email-validator`）

### 3.2 字段约束
使用 `Field` 添加额外约束：
```python
from pydantic import Field

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="价格必须大于0")
    tags: List[str] = Field(min_items=1)
```

### 3.3 自定义验证器
使用 `@validator` 装饰器：
```python
from pydantic import validator

class User(BaseModel):
    name: str

    @validator('name')
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('必须包含空格')
        return v
```

---

## 4. 高级特性

### 4.1 嵌套模型
```python
class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    name: str
    address: Address  # 嵌套模型
```

### 4.2 模型继承
```python
class BaseUser(BaseModel):
    email: str

class User(BaseUser):
    name: str
```

### 4.3 配置类（Config）
自定义模型行为：
```python
class User(BaseModel):
    name: str

    class Config:
        allow_mutation = False  # 禁止修改实例
        anystr_lower = True     # 自动转换字符串为小写
```

### 4.4 JSON 序列化与解析
```python
user = User(name="Alice")
json_data = user.json()          # 序列化为 JSON
user_from_json = User.parse_raw(json_data)  # 从 JSON 解析
```

---

## 5. 实用功能

### 5.1 数据导出
```python
user.dict()          # 转换为字典
user.json()          # 转换为 JSON 字符串
user.copy()          # 深拷贝实例
```

### 5.2 模型更新
```python
user.update({"name": "Bob"})  # 更新字段值
```

### 5.3 动态模型创建
使用 `create_model` 动态创建模型：
```python
from pydantic import create_model

DynamicUser = create_model("DynamicUser", name=(str, ...), age=(int, 25))
```

---

## 6. 与 FastAPI 集成
Pydantic 是 FastAPI 的默认数据模型库，用于请求/响应验证：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
def create_item(item: Item):  # 自动验证请求体
    return item
```

---

## 7. 常见问题

### 7.1 循环引用
使用 `defer`（Pydantic v1.10+）或字符串类型注解：
```python
from pydantic import BaseModel
from typing import List

class User(BaseModel):
    name: str
    friends: List['User'] = []  # 字符串注解避免循环引用

User.update_forward_refs()  # 更新前向引用
```

### 7.2 性能优化
- 避免深层嵌套模型。
- 使用 `alias_generator` 处理字段别名。

---

## 8. 总结

Pydantic 通过类型注解提供了简洁而强大的数据验证功能，适用于：
- API 请求/响应验证
- 配置文件管理
- 数据管道中的类型检查
- 与 FastAPI、Django 等框架集成

官方文档：[https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)
