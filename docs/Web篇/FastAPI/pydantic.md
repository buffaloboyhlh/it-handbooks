# Pydantic V2 教程

---

## Pydantic V2 全面教程：从入门到精通

### 一、 Pydantic 是什么？

Pydantic 是一个基于 Python 类型注解的数据验证和设置管理库。它的核心功能是：

1.  **验证**：确保输入数据符合你定义的模型规范。
2.  **转换**：自动将输入数据（来自 JSON、字典等）转换为指定的 Python 类型（如 `datetime`, `UUID` 等）。
3.  **序列化**：将模型实例轻松转换为字典或 JSON。
4.  **设置管理**：非常适合管理应用程序的配置。

Pydantic V2 是一个重大更新，带来了显著的性能提升（基于 Rust）和更清晰的 API。

---

### 二、 核心概念与基础用法

#### 1. 安装

```bash
pip install pydantic
```

#### 2. 第一个模型：`BaseModel`

一切始于从 `pydantic.BaseModel` 继承来定义你的数据模型。

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str = "John Doe"
    email: str | None = None  # 可选字段，默认为 None
    is_active: bool = True    # 带默认值的字段

# 使用字典创建实例（验证和转换在此发生）
user_data = {"id": 123, "name": "Alice", "email": "alice@example.com"}
user = User(**user_data)

print(user)
# > id=123 name='Alice' email='alice@example.com' is_active=True

print(user.id)        # 123
print(user.name)      # 'Alice'
print(user.model_dump()) # 将模型转回字典
# > {'id': 123, 'name': 'Alice', 'email': 'alice@example.com', 'is_active': True}

# 如果数据无效，会抛出 ValidationError
try:
    invalid_user = User(id="not_an_int")
except ValidationError as e:
    print(e)
    """
    1 validation error for User
    id
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='not_an_int', input_type=str]
    """
```

#### 3. 类型验证与自动转换

Pydantic 会自动尝试将输入数据转换为你声明的类型。

```python
from datetime import datetime
from pydantic import BaseModel

class Event(BaseModel):
    name: str
    timestamp: datetime  # 传入字符串，自动转换为 datetime 对象
    participants: list[str]  # 传入元组，自动转换为列表

event_data = {
    "name": "Product Launch",
    "timestamp": "2023-10-27 12:00:00",  # 字符串
    "participants": ("Alice", "Bob")     # 元组
}

event = Event(**event_data)
print(event.timestamp)  # datetime.datetime(2023, 10, 27, 12, 0)
print(type(event.timestamp))  # <class 'datetime.datetime'>
print(event.participants)  # ['Alice', 'Bob']
```

---

### 三、 进阶验证器 (Validators)

Pydantic 提供了强大的装饰器来创建自定义验证逻辑。

#### 1. 字段验证器 (`@field_validator`)

用于验证单个字段的值。

```python
from pydantic import BaseModel, field_validator, ValidationError

class UserProfile(BaseModel):
    username: str
    age: int

    # validator 装饰器，指定要验证的字段
    @field_validator('username')
    @classmethod
    def username_must_contain_letter(cls, v: str) -> str:
        if not any(char.isalpha() for char in v):
            raise ValueError('must contain at least one letter')
        return v.title()  # 验证器还可以对值进行修改

    @field_validator('age')
    @classmethod
    def age_must_be_realistic(cls, v: int) -> int:
        if not 0 < v < 120:
            raise ValueError('age must be between 1 and 119')
        return v

# 测试
try:
    profile = UserProfile(username="123", age=25) # 用户名无效
except ValidationError as e:
    print(e)

good_profile = UserProfile(username="alice123", age=30)
print(good_profile.username)  # 'Alice123' (被 title() 修改了)
```

#### 2. 模型验证器 (`@model_validator`)

用于验证多个字段之间的关系。

```python
from pydantic import BaseModel, model_validator, ValidationError

class PasswordModel(BaseModel):
    password: str
    password_confirm: str

    @model_validator(mode='after') # ‘after’ 表示在初始验证完成后运行
    def check_passwords_match(self) -> 'PasswordModel':
        pw1 = self.password
        pw2 = self.password_confirm
        if pw1 is not None and pw2 is not None and pw1 != pw2:
            raise ValueError('passwords do not match')
        return self

# 测试
try:
    pw = PasswordModel(password="abc", password_confirm="123")
except ValidationError as e:
    print(e)
```

---

### 四、 高级功能与配置

#### 1. 模型配置 (`model_config`)

使用 `model_config` 属性来自定义模型行为。

```python
from pydantic import BaseModel, ConfigDict

class Company(BaseModel):
    # 在 V2 中，这是新的配置方式
    model_config = ConfigDict(
        frozen=True,          # 创建后实例不可变（类似命名元组）
        str_strip_whitespace=True, # 自动去除字符串两端的空格
        extra='forbid'        # 禁止传入额外字段（默认是 'ignore'）
    )

    name: str
    founded: int

company = Company(name="  Tech Corp  ", founded=1999)
print(company.name)  # 'Tech Corp' (空格被去除)

try:
    company.founded = 2000  # 错误：实例是 frozen 的
except AttributeError as e:
    print(e)

try:
    company_data = {"name": "Test", "founded": 2000, "ceo": "Alice"}
    c = Company(**company_data) # 错误：存在额外字段 'ceo'
except ValidationError as e:
    print(e)
```

#### 2. 自定义字段类型 (`Annotated` 与 `Field`)

使用 `Field` 为字段添加元数据和约束。

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated
from datetime import date

class Product(BaseModel):
    # 使用 Annotated 和 Field 提供描述、示例和约束
    name: Annotated[str, Field(description="The name of the product", min_length=1, max_length=50)]
    price: Annotated[float, Field(gt=0, description="Price must be positive")] 
    release_date: date | None = Field(
        default=None,
        examples=["2023-01-01"], # 提供示例值
        json_schema_extra={"format": "date"} # 为 JSON Schema 添加额外信息
    )

# 测试
try:
    p = Product(name="", price=-10) # 两个字段都无效
except ValidationError as e:
    print(e)

# 查看模型的 JSON Schema
print(Product.model_json_schema())
```

#### 3. 嵌套模型

模型可以包含其他模型，实现复杂数据结构的验证。

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    addresses: List[Address]  # 地址列表

person_data = {
    "name": "Bob",
    "addresses": [
        {"street": "123 Main St", "city": "Springfield", "zip_code": "12345"},
        {"street": "456 Oak Ave", "city": "Metropolis", "zip_code": "67890"},
    ]
}

person = Person(**person_data)
print(person.addresses[0].city)  # Springfield
```

#### 4. 序列化与别名 (`alias`)

处理 JSON 键名与 Python 属性名不一致的情况。

```python
from pydantic import BaseModel, Field, ConfigDict

class DataModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True, # 允许同时使用 `alias` 和 属性名
    )

    # 字段在 Python 中叫 `first_name`，在 JSON 中期望叫 `firstName`
    first_name: str = Field(alias="firstName")
    age: int

# 通过别名（JSON键）创建
data = DataModel(firstName="Alice", age=30)
print(data.first_name) # Alice

# 因为设置了 populate_by_name=True，也可以用属性名创建
data2 = DataModel(first_name="Bob", age=25)
print(data2.model_dump(by_alias=True))  # {'firstName': 'Bob', 'age': 25}
```

---

### 五、 与 JSON 和 HTTP 框架的集成

#### 1. 序列化为 JSON

```python
user = User(id=1, name="Alice")
json_str = user.model_dump_json()
print(json_str) # '{"id":1,"name":"Alice","email":null,"is_active":true}'

# 排除默认值
json_str_minimal = user.model_dump_json(exclude_defaults=True)
print(json_str_minimal) # '{"id":1,"name":"Alice"}'
```

#### 2. 在 FastAPI 中使用 (示例)

Pydantic 是 FastAPI 的基石，用于定义请求体和响应模型。

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 定义请求体模型
class ItemCreate(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

# 定义响应模型（可以排除敏感字段如 `tax`）
class ItemResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    price: float

@app.post("/items/", response_model=ItemResponse)
async def create_item(item: ItemCreate):
    # `item` 已经是一个验证过的 Pydantic 模型实例
    # ... 在这里处理业务逻辑，比如保存到数据库 ...
    fake_db_id = 123
    # 返回的数据会自动用 `ItemResponse` 模型验证和序列化
    return {**item.model_dump(), "id": fake_db_id}
```

---

### 六、 总结与最佳实践

1.  **拥抱类型注解**：充分利用 Python 的类型提示，这是 Pydantic 工作的基础。
2.  **从简单开始**：先用基本的 `BaseModel` 和标准类型，需要时再引入验证器。
3.  **善用 `Field`**：使用 `Field` 来声明约束和元数据，而不是把逻辑都写在验证器里。
4.  **配置模型行为**：使用 `model_config` 来全局控制模型的严格性、可变性等。
5.  **区分输入/输出模型**：在 Web API 中，为创建（输入）和读取（输出）定义不同的模型。输出模型可以排除敏感信息（如密码哈希）或包含计算字段（如 `id`）。
6.  **性能**：Pydantic V2 非常快，但复杂的自定义验证器会影响性能。保持验证器逻辑简洁。

Pydantic V2 通过其强类型、自动验证和清晰的 API，极大地提升了 Python 数据处理的可靠性和开发者体验。希望这份教程能帮助你有效地使用它！