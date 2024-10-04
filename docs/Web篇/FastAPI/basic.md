# FastAPI 教程 

### FastAPI 从入门到精通教程（详细版）

#### 目录
1. FastAPI 简介  
   1.1 什么是 FastAPI  
   1.2 FastAPI 的核心特性
2. 环境准备  
   2.1 安装 FastAPI 和 Uvicorn  
   2.2 创建 FastAPI 项目结构
3. 快速开始  
   3.1 Hello World  
   3.2 路由与视图函数
4. 请求方式  
   4.1 GET 请求  
   4.2 POST 请求  
   4.3 PUT 和 DELETE 请求
5. 路径参数和查询参数  
   5.1 路径参数  
   5.2 查询参数
6. 请求体与数据验证  
   6.1 使用 Pydantic 进行数据验证  
   6.2 嵌套模型与复杂数据结构
7. 响应与状态码  
   7.1 自定义响应  
   7.2 返回 JSON、HTML、文件和流数据
8. 中间件与依赖注入  
   8.1 什么是中间件  
   8.2 中间件的使用  
   8.3 依赖注入系统
9. 异步编程支持  
   9.1 异步基础  
   9.2 FastAPI 中的异步处理
10. 表单数据与文件上传  
    10.1 处理表单数据  
    10.2 处理文件上传
11. 安全与认证  
    11.1 OAuth2 实现认证  
    11.2 使用 JWT 进行身份验证
12. 数据库集成  
    12.1 使用 SQLAlchemy  
    12.2 使用 Tortoise ORM
13. WebSocket 支持  
    13.1 WebSocket 实现  
    13.2 WebSocket 聊天室示例
14. 后台任务与调度  
    14.1 后台任务处理  
    14.2 定时任务调度
15. 自动化文档生成  
    15.1 Swagger UI 和 ReDoc 文档
16. 部署 FastAPI 项目  
    16.1 使用 Uvicorn 部署  
    16.2 Docker 部署  
    16.3 部署到云平台

---

### 1. FastAPI 简介

#### 1.1 什么是 FastAPI

FastAPI 是一个基于 Python 的 Web 框架，它以极高的性能和开发效率著称，能够快速构建 API。其核心是使用 Python 的类型注释来进行自动化的输入验证和文档生成。FastAPI 使用 ASGI 标准，允许处理高并发请求，并且支持异步处理。

#### 1.2 FastAPI 的核心特性

- **高性能**：接近 Node.js 和 Go 的性能，适合高并发应用。
- **基于类型提示**：Python 类型提示用于自动生成 API 文档和数据验证。
- **自动生成文档**：支持 Swagger UI 和 ReDoc，方便 API 调试和测试。
- **异步支持**：天然支持异步 `async` 和 `await`，高效处理 I/O 操作。
- **依赖注入系统**：简化复杂应用的开发，尤其是处理数据库和安全相关的逻辑。

---

### 2. 环境准备

#### 2.1 安装 FastAPI 和 Uvicorn

FastAPI 是 ASGI 框架，需要一个 ASGI 服务器来运行。我们将使用 `uvicorn` 作为 ASGI 服务器。

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

#### 2.2 创建 FastAPI 项目结构

一个典型的 FastAPI 项目结构如下：

```
my_fastapi_project/
│
├── app/
│   ├── main.py
│   ├── models.py
│   ├── routers/
│   ├── dependencies.py
│   └── config.py
│
├── requirements.txt
└── README.md
```

- `main.py`：主要入口文件。
- `models.py`：数据库模型。
- `routers/`：路由定义的模块。
- `dependencies.py`：依赖注入的定义。
- `config.py`：配置文件。

---

### 3. 快速开始

#### 3.1 Hello World

创建一个最简单的 FastAPI 应用：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI"}
```

运行应用：

```bash
uvicorn main:app --reload
```

访问 `http://127.0.0.1:8000/`，你将看到：

```json
{
  "message": "Hello, FastAPI"
}
```

#### 3.2 路由与视图函数

FastAPI 使用装饰器来定义路由和视图函数。你可以使用 `@app.get()`、`@app.post()`、`@app.put()` 等装饰器来定义不同的 HTTP 请求方法。

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/items/")
async def create_item(name: str, price: float):
    return {"name": name, "price": price}
```

---

### 4. 请求方式

FastAPI 支持常见的 HTTP 请求方法：GET、POST、PUT、DELETE。

#### 4.1 GET 请求

GET 请求用于从服务器获取数据。可以通过路径参数或者查询参数传递数据。

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

#### 4.2 POST 请求

POST 请求通常用于向服务器提交数据。

```python
@app.post("/items/")
async def create_item(name: str, price: float):
    return {"name": name, "price": price}
```

#### 4.3 PUT 和 DELETE 请求

PUT 用于更新资源，DELETE 用于删除资源。

```python
@app.put("/items/{item_id}")
async def update_item(item_id: int, name: str):
    return {"item_id": item_id, "name": name}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    return {"item_id": item_id, "status": "deleted"}
```

---

### 5. 路径参数和查询参数

#### 5.1 路径参数

路径参数直接嵌入在 URL 中，FastAPI 会根据定义自动解析并验证类型。

```python
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}
```

在上面的例子中，`user_id` 被定义为 `int` 类型，访问 `/users/5` 将返回 `{"user_id": 5}`。

#### 5.2 查询参数

查询参数是通过 URL 中的 `?` 符号传递的，通常用于过滤或者排序等操作。

```python
@app.get("/search/")
async def search_items(q: str = None, page: int = 1):
    return {"q": q, "page": page}
```

如果你访问 `/search/?q=fastapi&page=2`，将会得到 `{"q": "fastapi", "page": 2}`。

---

### 6. 请求体与数据验证

#### 6.1 使用 Pydantic 进行数据验证

FastAPI 使用 Pydantic 模型来进行请求体的验证。Pydantic 可以定义请求体的结构，并自动进行类型检查和数据验证。

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

如果请求体不符合 Pydantic 模型定义的要求，FastAPI 会自动返回 422 错误，并给出详细的错误信息。

#### 6.2 嵌套模型与复杂数据结构

你可以在 Pydantic 模型中嵌套其他模型，来构建复杂的数据结构。

```python
from pydantic import BaseModel

class SubItem(BaseModel):
    name: str
    quantity: int

class Item(BaseModel):
    name: str
    description: str = None
    sub_items: list[SubItem]

@app.post("/items/")
async def create_item(item: Item):
    return item
```

---

### 7. 响应与状态码

#### 7.1 自定义响应

FastAPI 默认会将 Python 的数据结构（如字典、列表）转换为 JSON 格式并返回。你也可以返回自定义的响应，例如 HTML 或纯文本。

```python
from fastapi import Response

@app.get("/custom-response/")
async def custom_response():
    return Response(content="This is a plain text response", media_type="text/plain")
```

#### 7.2 返回 JSON、HTML、文件和流数据

你可以自定义返回的媒体类型，例如返回 HTML 或者文件。

```python


from fastapi.responses import HTMLResponse, FileResponse

@app.get("/html/", response_class=HTMLResponse)
async def get_html():
    return "<html><body><h1>Hello, HTML!</h1></body></html>"

@app.get("/download/")
async def download_file():
    return FileResponse("example.pdf", media_type='application/pdf')
```

---

### 8. 中间件与依赖注入

#### 8.1 什么是中间件

中间件是一种可以拦截请求和响应的组件，允许你在处理请求前或者返回响应前执行一些操作。

#### 8.2 中间件的使用

你可以使用 `add_middleware` 方法添加中间件。例如，你可以添加一个记录每次请求日志的中间件：

```python
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        print(f"Incoming request: {request.url}")
        response = await call_next(request)
        return response

app.add_middleware(LoggingMiddleware)
```

#### 8.3 依赖注入系统

依赖注入是一种将函数所依赖的资源自动传递给函数的机制，FastAPI 提供了一个强大的依赖注入系统。

```python
from fastapi import Depends

def common_parameters(q: str = None):
    return {"q": q}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

---

### 9. 异步编程支持

FastAPI 提供了完整的异步支持，允许开发者高效处理 I/O 密集型操作。

#### 9.1 异步基础

在 FastAPI 中，异步函数通过 `async def` 定义。异步函数会使用 `await` 关键字来等待 I/O 操作的完成。

#### 9.2 FastAPI 中的异步处理

你可以轻松使用异步函数来处理网络请求或数据库查询。例如：

```python
import httpx

@app.get("/external-api/")
async def call_external_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

---

### 10. 表单数据与文件上传

#### 10.1 处理表单数据

FastAPI 可以轻松处理表单数据，使用 `Form` 作为请求参数类型。

```python
from fastapi import Form

@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}
```

#### 10.2 处理文件上传

文件上传也非常简单，可以使用 `File` 和 `UploadFile` 处理大文件。

```python
from fastapi import File, UploadFile

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}
```

---

### 11. 安全与认证

FastAPI 内置支持 OAuth2 和 JWT（JSON Web Token）等常见的认证方式。

#### 11.1 OAuth2 实现认证

使用 OAuth2 来处理身份认证：

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me/")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"token": token}
```

#### 11.2 使用 JWT 进行身份验证

你可以结合 JWT 来实现用户认证和授权。

---

### 12. 数据库集成

#### 12.1 使用 SQLAlchemy

SQLAlchemy 是 Python 的一个流行 ORM 库，常与 FastAPI 一起使用。

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
```

#### 12.2 使用 Tortoise ORM

Tortoise ORM 是一个现代的异步 ORM，适合与 FastAPI 一起使用。

---

### 13. WebSocket 支持

FastAPI 支持 WebSocket 协议，适合实时应用开发。

#### 13.1 WebSocket 实现

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello WebSocket")
    await websocket.close()
```

#### 13.2 WebSocket 聊天室示例

你可以使用 WebSocket 创建一个简单的多人聊天室。

---

### 14. 后台任务与调度

#### 14.1 后台任务处理

你可以使用 `BackgroundTasks` 处理后台任务。

```python
from fastapi import BackgroundTasks

async def write_log(message: str):
    with open("log.txt", "a") as log_file:
        log_file.write(message + "\n")

@app.post("/log/")
async def create_log(background_tasks: BackgroundTasks, message: str):
    background_tasks.add_task(write_log, message)
    return {"message": "Log will be written"}
```

#### 14.2 定时任务调度

FastAPI 也可以结合其他库如 `APScheduler` 来实现定时任务。

---

### 15. 自动化文档生成

#### 15.1 Swagger UI 和 ReDoc 文档

FastAPI 会自动生成基于 OpenAPI 的文档，默认提供 Swagger UI 和 ReDoc 两种文档页面：

- Swagger UI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`

---

### 16. 部署 FastAPI 项目

#### 16.1 使用 Uvicorn 部署

你可以直接使用 Uvicorn 部署 FastAPI 应用。

```bash
uvicorn main:app --host 0.0.0.0 --port 80
```

#### 16.2 Docker 部署

创建一个 `Dockerfile`：

```Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

然后运行：

```bash
docker build -t fastapi-app .
docker run -d -p 80:80 fastapi-app
```

#### 16.3 部署到云平台

FastAPI 可以轻松部署到 AWS、Google Cloud、Heroku 等云平台，具体操作请参考平台文档。

---

这个详细的教程介绍了 FastAPI 的核心功能和高级特性，提供了全面的示例和用法，帮助你从基础入门，直到掌握复杂应用的开发与部署。