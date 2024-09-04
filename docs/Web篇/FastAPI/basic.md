# FastAPI 基础教程

FastAPI 是一个现代的、快速的 Web 框架，用于构建 APIs，具有以下特点：

1. **高性能**：基于 Python 的异步功能，FastAPI 的性能接近于 Node.js 和 Go。
2. **易用性**：设计上简洁优雅，支持类型注解，开发者体验非常好。
3. **自动生成文档**：基于类型提示自动生成交互式 API 文档（Swagger UI 和 ReDoc）。
4. **异步支持**：原生支持 `async` 和 `await`，轻松构建高并发的异步 API。
5. **数据验证**：使用 Pydantic 进行数据验证和解析。

### 1. FastAPI 概念

#### 1.1 请求和响应模型
- **请求**：客户端发送数据到服务器，通常通过 GET、POST、PUT、DELETE 等 HTTP 方法。
- **响应**：服务器处理请求并返回数据，通常为 JSON 格式。

#### 1.2 路由（Routing）
路由是将 URL 路径映射到特定的处理函数。FastAPI 提供了简洁的路由定义方式，通过 Python 函数直接绑定到路由。

#### 1.3 数据模型（Pydantic）
FastAPI 使用 Pydantic 进行数据模型定义和验证。Pydantic 支持基于类型注解的自动数据验证。

#### 1.4 依赖注入（Dependency Injection）
FastAPI 支持依赖注入，通过函数参数的形式将依赖传递给路由处理函数。它允许开发者轻松管理复杂的依赖关系。

### 2. 快速上手：构建一个简单的 FastAPI 应用

#### 2.1 创建应用程序

首先，创建一个最基本的 FastAPI 应用。

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
```

这里我们定义了一个 FastAPI 实例 `app`，并使用 `@app.get("/")` 装饰器定义了一个 GET 请求的处理函数 `read_root`。

#### 2.2 启动应用

使用 Uvicorn 启动 FastAPI 应用：

```bash
uvicorn main:app --reload
```

- `main` 是指文件名 `main.py`。
- `app` 是 FastAPI 实例的名字。
- `--reload` 启用自动重载，代码更新时服务器会自动重启。

打开浏览器访问 `http://127.0.0.1:8000/`，你会看到一个简单的 JSON 响应 `{"message": "Hello, World!"}`。

#### 2.3 路由和路径参数

我们可以定义更多的路由，并使用路径参数来处理动态 URL。

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

- `item_id` 是路径参数，`q` 是查询参数（带默认值）。

访问 `http://127.0.0.1:8000/items/42?q=foo`，响应将是 `{"item_id": 42, "q": "foo"}`。

#### 2.4 数据模型与请求体

使用 Pydantic 定义数据模型，并在请求体中接收 JSON 数据。

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item}
```

- `Item` 是一个 Pydantic 模型，用于验证请求体数据。
- 使用 `@app.post("/items/")` 定义了一个 POST 路由，接收 `Item` 类型的数据。

发送 POST 请求到 `http://127.0.0.1:8000/items/`，携带 JSON 数据 `{"name": "Book", "price": 12.5}`，服务器将返回接收的对象。

#### 2.5 依赖注入

依赖注入允许将依赖的组件或配置注入到路由处理函数中。

```python
from fastapi import Depends

def get_query(q: str = None):
    return q

@app.get("/items/")
async def read_items(q: str = Depends(get_query)):
    return {"q": q}
```

- `get_query` 是一个依赖函数，`Depends` 负责将其注入到路由函数中。

#### 2.6 自动生成文档

FastAPI 自动生成 API 文档，提供交互式界面：

- Swagger UI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`

### 3. 更高级的操作

#### 3.1 中间件（Middleware）

中间件是一个处理请求和响应的钩子，可以用于请求日志记录、跨域处理等。

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 3.2 异步任务（Background Tasks）

FastAPI 支持后台任务，可以在响应返回后继续执行长时间运行的任务。

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", "a") as log:
        log.write(message)

@app.post("/log/")
async def create_log(message: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, message)
    return {"message": "Log will be written"}
```

#### 3.3 WebSockets

FastAPI 原生支持 WebSockets，可以轻松构建实时通信的应用。

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
```

### 4. 部署 FastAPI 应用

FastAPI 可以在多个环境中部署：

- **使用 Uvicorn**：适合开发和小型应用。
- **使用 Gunicorn + Uvicorn**：适合生产环境，支持多个工作进程。
- **容器化部署**：使用 Docker 将应用容器化，方便在各种环境中部署。

### 总结

FastAPI 是一个强大而灵活的 Web 框架，它结合了类型提示、自动文档生成、异步支持等现代特性，使得开发高性能、高可维护性的 API 变得更加容易。通过本文的学习，你应该对 FastAPI 的基本概念、路由、数据模型、依赖注入等有了深入的理解。