# FastAPI  面试手册

在 FastAPI 的面试中，面试官通常会考察候选人对 FastAPI 的基本概念、实际应用以及如何应对各种开发场景的能力。以下是一些典型的 FastAPI 面试题目和解答举例，帮助你更好地理解面试中的问题和解答思路。

### 一、FastAPI 基础概念

#### 1.1 **举例说明：什么是 FastAPI，它的优势是什么？**
- **示例问题**：请简要介绍一下 FastAPI，以及它相比其他 Python Web 框架的优势。
- **详细解答**：
  - **FastAPI 介绍**：FastAPI 是一个基于 Python 3.6+ 的现代 Web 框架，用于构建快速（high-performance）API，基于标准 Python 类型提示。
  - **优势**：
    - **高性能**：基于 Starlette 和 Pydantic，性能与 Node.js 和 Go 相当。
    - **自动生成文档**：支持自动生成 OpenAPI 和 JSON Schema 文档。
    - **依赖注入**：内置的依赖注入系统，便于管理复杂的依赖关系。
    - **类型安全**：利用 Python 类型提示确保参数验证和类型检查。

### 二、路由与请求处理

#### 2.1 **举例说明：如何在 FastAPI 中定义一个简单的路由？**
- **示例问题**：请展示如何使用 FastAPI 定义一个返回 "Hello, World!" 的 GET 请求。
- **详细解答**：
  - **问题分析**：在 FastAPI 中，通过装饰器 `@app.get()` 定义 GET 请求路由。
  - **处理方法**：
    - **定义路由**：通过装饰器定义路由，并在路由函数中返回响应内容。

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
```

### 三、请求参数

#### 3.1 **举例说明：如何在 FastAPI 中处理路径参数和查询参数？**
- **示例问题**：请编写一个 API 接口，它接受一个用户名作为路径参数，并可选地接受一个年龄作为查询参数。
- **详细解答**：
  - **问题分析**：路径参数直接在 URL 路径中定义，查询参数通过 URL 的 `?` 号后面定义。
  - **处理方法**：
    - **路径参数**：在路径中定义参数，并在函数签名中相应定义。
    - **查询参数**：在函数签名中定义可选参数，并设置默认值。

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{username}")
def read_user(username: str, age: int = None):
    return {"username": username, "age": age}
```

### 四、请求体和表单数据

#### 4.1 **举例说明：如何在 FastAPI 中处理 JSON 请求体？**
- **示例问题**：请编写一个 API 接口，它接受一个包含用户信息的 JSON 对象，并返回处理后的用户数据。
- **详细解答**：
  - **问题分析**：FastAPI 使用 Pydantic 模型来定义和验证请求体的数据结构。
  - **处理方法**：
    - **定义 Pydantic 模型**：创建 Pydantic 模型类来描述请求体结构。
    - **请求体处理**：通过函数参数接收并处理 JSON 数据。

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

@app.post("/users/")
def create_user(user: User):
    return {"name": user.name.upper(), "age": user.age}
```

### 五、依赖注入

#### 5.1 **举例说明：如何在 FastAPI 中使用依赖注入？**
- **示例问题**：如何定义一个依赖，它会在每个请求中打印一条日志，并在路由函数中使用它？
- **详细解答**：
  - **问题分析**：FastAPI 允许通过 `Depends` 进行依赖注入，便于代码重用和逻辑分离。
  - **处理方法**：
    - **定义依赖**：创建一个函数，用作依赖。
    - **使用依赖**：在路由函数中通过 `Depends` 注入该依赖。

```python
from fastapi import FastAPI, Depends

app = FastAPI()

def log_dependency():
    print("Log: New request received")
    return "Dependency injected"

@app.get("/items/")
def read_items(log: str = Depends(log_dependency)):
    return {"message": log}
```

### 六、异常处理

#### 6.1 **举例说明：如何在 FastAPI 中处理异常？**
- **示例问题**：如何定义一个自定义异常并在 FastAPI 中处理它？
- **详细解答**：
  - **问题分析**：可以通过 `HTTPException` 抛出标准异常，也可以通过自定义异常处理器处理特殊情况。
  - **处理方法**：
    - **定义自定义异常**：通过继承 Python 的 `Exception` 类创建自定义异常。
    - **注册异常处理器**：使用 `app.exception_handler` 装饰器注册处理器。

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
def custom_exception_handler(request, exc: CustomException):
    return JSONResponse(status_code=418, content={"message": f"Oops! {exc.name} did something wrong."})

@app.get("/items/{name}")
def read_item(name: str):
    if name == "bad":
        raise CustomException(name=name)
    return {"name": name}
```

### 七、测试 FastAPI 应用

#### 7.1 **举例说明：如何测试一个 FastAPI 应用？**
- **示例问题**：请展示如何使用 `TestClient` 测试一个 FastAPI 应用的 API 路由。
- **详细解答**：
  - **问题分析**：FastAPI 提供了 `TestClient` 来模拟请求，测试 API 的行为。
  - **处理方法**：
    - **测试客户端**：使用 `TestClient` 创建客户端，发送请求并验证响应。

```python
from fastapi.testclient import TestClient
from myapp import app  # 假设上面的 FastAPI 应用在 myapp.py 文件中

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}
```

### 八、部署 FastAPI 应用

#### 8.1 **举例说明：如何部署一个 FastAPI 应用？**
- **示例问题**：你会如何将一个 FastAPI 应用部署到生产环境？
- **详细解答**：
  - **问题分析**：部署 FastAPI 应用的方式多种多样，常见的有使用 Uvicorn 或 Gunicorn。
  - **处理方法**：
    - **本地部署**：使用 Uvicorn 运行应用。
    - **生产环境部署**：通过 Gunicorn + Uvicorn 多工作进程的方式提高性能。

```bash
# 使用 Uvicorn 本地运行应用
uvicorn myapp:app --reload

# 使用 Gunicorn + Uvicorn 部署到生产环境
gunicorn -k uvicorn.workers.UvicornWorker myapp:app
```

这些示例涵盖了 FastAPI 面试中可能涉及的多个方面，帮助你系统性地理解和回答相关问题。通过这些详细的解答，你可以更好地准备 FastAPI 面试，展现你的技术能力和项目经验。