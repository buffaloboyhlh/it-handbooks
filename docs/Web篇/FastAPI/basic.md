# FastAPI 基础教程

FastAPI 是一个现代、快速（高性能）、基于 Python 的 Web 框架，用于构建 API。它通过 Python 的类型提示（Type Hints）提供高效的开发体验，具有自动生成 API 文档、异步支持和强大的数据验证功能。下面将详细介绍 FastAPI 的基础概念、核心功能以及常见的应用场景。

---

### 1. 快速开始

#### 安装 FastAPI 和 Uvicorn

FastAPI 本身不包含服务器，通常与 Uvicorn 一起运行。安装命令如下：

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

Uvicorn 是一个 ASGI 服务器，FastAPI 依赖它来启动和运行。

#### 最简单的 FastAPI 应用

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}
```

#### 运行应用

```bash
uvicorn main:app --reload
```

- `main` 是 Python 文件的名称。
- `app` 是 FastAPI 实例。
- `--reload` 启动代码自动热加载（开发环境下使用）。

#### 查看 API 文档

FastAPI 自动生成 API 文档，默认路径为：
- Swagger UI 文档：`http://127.0.0.1:8000/docs`
- ReDoc 文档：`http://127.0.0.1:8000/redoc`

---

### 2. 路由与请求方法

FastAPI 支持常见的 HTTP 请求方法：GET、POST、PUT、DELETE 等。你可以通过装饰器定义路由。

#### 示例：基本请求方法

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(name: str, price: float):
    return {"name": name, "price": price}

@app.put("/items/{item_id}")
async def update_item(item_id: int, name: str, price: float):
    return {"item_id": item_id, "name": name, "price": price}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    return {"item_id": item_id}
```

#### 详解

- `@app.get()`、`@app.post()` 等是定义不同 HTTP 请求方法的装饰器。
- 路径参数（如 `item_id`）通过 URL 直接传递。
- 查询参数（如 `q`）通过 URL 中的 `?` 传递，是可选参数。

---

### 3. 数据验证与模型

FastAPI 通过 **Pydantic** 自动进行数据验证，使用 Pydantic 的 `BaseModel` 定义数据模型，模型会自动进行类型检查和校验。

#### 示例：数据模型

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    description: str = None
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return {"name": item.name, "price": item.price, "description": item.description}
```

#### 详解

- `BaseModel`：数据模型基于 Pydantic 的 `BaseModel`，可以定义字段类型和默认值。
- FastAPI 会自动将请求体的数据转换为模型实例，并且自动执行数据校验。请求数据不符合模型时，返回 422 错误。

---

### 4. 路径参数与查询参数

FastAPI 支持通过路径传递参数（路径参数），以及通过 URL 查询字符串传递参数（查询参数）。

#### 示例：路径参数与查询参数

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

- `item_id` 是路径参数，必须传递。
- `q` 是查询参数，带默认值为 `None`，可选。

---

### 5. 请求体与响应体

FastAPI 允许你轻松处理请求体数据，并自动处理 JSON 响应体。

#### 示例：POST 请求体数据

```python
from fastapi import Body

@app.post("/items/")
async def create_item(name: str = Body(...), price: float = Body(...)):
    return {"name": name, "price": price}
```

#### 详解

- **Body**：通过 `Body` 传递请求体数据，`Body(...)` 表示必填字段。
- FastAPI 自动将请求体的 JSON 数据解析为 Python 数据类型。

---

### 6. 表单与文件上传

FastAPI 支持表单数据和文件上传，方便处理前端提交的文件。

#### 示例：表单数据

```python
from fastapi import Form

@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}
```

#### 示例：文件上传

```python
from fastapi import File, UploadFile

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}
```

- **UploadFile**：提供文件的 `filename`、`content_type` 等信息，支持异步文件读取。

---

### 7. 异步支持

FastAPI 天生支持异步处理请求，使用 `async` 和 `await` 关键字，可以实现异步编程，提升应用性能。

```python
@app.get("/async/")
async def async_example():
    return {"message": "This is async!"}
```

#### 详解

- FastAPI 原生支持异步请求处理，适合构建高并发应用，特别是在 I/O 密集型任务中具有显著优势。

---

### 8. 依赖注入（Dependency Injection）

FastAPI 提供了依赖注入机制，允许开发者将依赖的函数或对象注入到路由中，简化代码的复用性和可测试性。

#### 示例：依赖注入

```python
from fastapi import Depends

async def common_parameters(q: str = None):
    return {"q": q}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

#### 详解

- **Depends**：FastAPI 使用 `Depends()` 来声明一个依赖的函数。FastAPI 自动执行该函数，并将返回值作为参数传递给路由函数。

---

### 9. 中间件

中间件可以拦截每个请求并在请求到达路由之前或之后处理逻辑。常见的中间件包括请求日志、跨域资源共享（CORS）等。

#### 示例：中间件

```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Custom-Header"] = "Value"
        return response

app.add_middleware(CustomMiddleware)
```

#### 详解

- 中间件会拦截请求并允许对请求和响应进行自定义处理。`BaseHTTPMiddleware` 提供了基础实现。

---

### 10. 错误处理

FastAPI 提供了方便的错误处理机制，允许你自定义响应错误信息。

#### 示例：自定义错误

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}
```

- **HTTPException**：通过抛出 `HTTPException` 可以返回自定义的 HTTP 状态码和错误信息。

---

### 11. FastAPI 自动生成文档

FastAPI 自动为所有 API 生成 OpenAPI 文档，开发者可以通过 `/docs` 查看 Swagger UI，通过 `/redoc` 查看 ReDoc 文档。

- **Swagger UI**：API 调试和文档查看工具，默认路径：`http://127.0.0.1:8000/docs`
- **ReDoc**：API 文档生成工具，默认路径：`http://127.0.0.1:8000/redoc`

---

### 12. FastAPI 与数据库集成

FastAPI 支持与任意数据库集成，例如 SQLAlchemy、Tortoise ORM 或 MongoDB。以下是与 SQLAlchemy 的集成示例：

#### 示例：SQLAlchemy 集成

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

Base.metadata.create_all(bind=engine)
```

#### 详解

- **SQLAlchemy**：使用 SQLAlchemy 创建数据库连接、定义模型，并进行基本的数据库操作。
- 通过 FastAPI 的依赖注入，可以将数据库会话对象注入到路由函数中。

---

### 13. FastAPI 项目结构

当项目变大时，推荐采用模块化的结构来组织代码。下面是一个典型的 FastAPI 项目结构：

```
.
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── routers
│   │   ├── __init__.py
│   │   ├── items.py
│   │   └── users.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── item.py
│   │   └── user.py
│   ├── database.py
│   └── schemas
│       ├── __init__.py
│       ├── item.py
│       └── user.py
├── tests
│   ├── __init__.py
│   └── test_main.py
└── requirements.txt
```

#### 目录结构说明

- `app/main.py`: 应用的入口文件，创建 FastAPI 实例并运行应用。
- `app/routers/`: 存放应用的路由，每个文件对应一个模块（例如 `items.py`, `users.py`）。
- `app/models/`: 数据库模型文件，定义数据库中的表和字段。
- `app/schemas/`: Pydantic 模型，定义请求和响应的格式。
- `app/database.py`: 数据库配置和会话管理。
- `tests/`: 存放测试文件。

#### 示例：`app/main.py`

```python
from fastapi import FastAPI
from app.routers import items, users

app = FastAPI()

app.include_router(items.router)
app.include_router(users.router)
```

#### 示例：`app/routers/items.py`

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/items/")
async def read_items():
    return [{"name": "Item 1"}, {"name": "Item 2"}]
```

#### 详解

- **APIRouter**：可以将不同功能的路由分模块管理，每个模块通过 `APIRouter` 来定义路由，并在 `main.py` 中注册。
- **include_router**：将不同的路由模块导入到主应用中，形成完整的路由结构。

---

### 14. FastAPI 中间件详解

中间件是处理请求和响应的钩子函数，可以在进入路由之前或者返回响应之后进行额外的操作，例如日志记录、修改响应头、执行身份验证等。

#### 示例：记录请求日志的中间件

```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request

app = FastAPI()

class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"Request URL: {request.url}")
        response = await call_next(request)
        return response

app.add_middleware(LogMiddleware)
```

#### 详解

- **BaseHTTPMiddleware**：实现自定义中间件的基类。你可以覆盖 `dispatch` 方法，在请求到达路由之前进行处理。
- **call_next**：该方法会调用下一个中间件或路由。

---

### 15. FastAPI 异常处理

FastAPI 提供了一种统一的方式来处理异常。你可以使用 `HTTPException` 来抛出标准 HTTP 错误，也可以自定义异常处理器。

#### 示例：自定义异常处理器

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"Oops! {exc.detail}"}
    )
```

#### 详解

- **exception_handler**：用于定义自定义异常处理器，处理特定类型的异常。
- **JSONResponse**：返回自定义格式的 JSON 响应。

---

### 16. FastAPI 测试

FastAPI 支持编写测试，测试通常使用 `TestClient` 提供的客户端模拟请求。

#### 示例：使用 `TestClient` 进行测试

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}
```

#### 详解

- **TestClient**：用于模拟客户端请求，适合编写自动化测试。
- **assert**：测试响应的状态码和内容是否符合预期。

---

### 17. FastAPI 部署

FastAPI 应用可以通过多种方式部署到生产环境。以下是常见的部署方式：

#### 1. 使用 Uvicorn 部署

Uvicorn 是一个轻量级、高性能的 ASGI 服务器，适合运行 FastAPI。

```bash
uvicorn app.main:app --host 0.0.0.0 --port 80
```

#### 2. 使用 Gunicorn 和 Uvicorn 部署

Gunicorn 是一个流行的 WSGI 服务器，结合 Uvicorn 可以提高稳定性。

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

- `-w 4`: 表示启动 4 个工作进程。
- `-k uvicorn.workers.UvicornWorker`: 使用 Uvicorn 的 ASGI worker。

#### 3. 使用 Docker 部署

可以使用 Docker 将 FastAPI 应用打包成镜像并进行部署。

#### Dockerfile 示例

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

#### 详解

- 使用 Docker 将应用容器化，可以方便地在不同环境中运行，保证一致性。

---

### 18. FastAPI 性能优化

#### 1. 使用异步函数

FastAPI 本身支持异步编程，使用 `async` 和 `await` 可以显著提高 I/O 密集型任务的性能。

#### 2. 使用缓存

对于计算量大、频繁调用的接口，可以考虑使用缓存机制，如 Redis 或内存缓存。

#### 3. 数据库连接池

使用数据库连接池（如 `asyncpg`）可以减少频繁建立数据库连接的开销，提高数据库访问性能。

#### 4. 使用 CDN

对于静态资源（如图片、CSS、JS 文件），可以通过 CDN 加速访问，减轻服务器的压力。

#### 5. 使用 Gunicorn 管理多个进程

通过 `Gunicorn` 启动多个工作进程，可以有效利用多核 CPU 提升吞吐量。

---

### 19. FastAPI 权限验证和安全机制

FastAPI 提供了一些内置的工具，用于处理身份验证和权限验证。支持 OAuth2、JWT、API Key 等常见的认证方式。

#### 1. OAuth2 和 JWT 认证

OAuth2 是常见的授权协议，JWT (JSON Web Tokens) 用于身份验证和授权，FastAPI 对 OAuth2 及 JWT 的支持非常强大。

##### 示例：OAuth2 JWT 认证

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

app = FastAPI()

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

@app.post("/token")
async def login():
    # 模拟登录返回 JWT token
    return {"access_token": "fake_jwt_token", "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = decode_jwt(token)
    return {"user": user}
```

##### 详解：

- **OAuth2PasswordBearer**：用于声明一个 OAuth2 认证的依赖，传递的参数是认证的 URL。
- **JWT**：通过 `jose` 库对 JWT 进行编码和解码。
- **decode_jwt**：一个解码 JWT 并验证的函数，抛出异常表示认证失败。

---

#### 2. API Key 认证

API Key 是一种常见的认证方式，通常在请求头中传递。

##### 示例：API Key 认证

```python
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

app = FastAPI()

API_KEY = "your_api_key"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@app.get("/protected-route")
async def protected_route(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return {"message": "You are authorized!"}
```

##### 详解：

- **APIKeyHeader**：声明一个 API Key 依赖，使用自定义请求头来传递 API Key。
- **Security**：类似于 `Depends`，用于将认证依赖传递给路由。

---

### 20. FastAPI 的 WebSocket 支持

FastAPI 原生支持 WebSocket，适合构建实时通信应用。WebSocket 是一种双向通信协议，通常用于即时消息传递、实时数据推送等场景。

#### 示例：WebSocket 简单示例

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
```

##### 详解：

- **WebSocket**：通过 `WebSocket` 对象实现双向通信。`accept()` 表示接受连接，`receive_text()` 表示接收客户端消息，`send_text()` 用于向客户端发送消息。
- WebSocket 适用于高频数据传输和实时交互应用，如聊天室、在线游戏等。

---

### 21. FastAPI 的 CORS（跨域资源共享）设置

在 Web 应用开发中，CORS 是处理跨域请求的一个常见问题。FastAPI 提供了方便的中间件来解决跨域问题。

#### 示例：设置 CORS 中间件

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

@app.get("/")
async def main():
    return {"message": "Hello World"}
```

##### 详解：

- **CORSMiddleware**：通过 FastAPI 的中间件机制解决跨域问题。
- **allow_origins**：允许跨域的来源列表。`["*"]` 表示允许所有来源。
- **allow_methods**：允许的 HTTP 方法，例如 `["GET", "POST"]`。
- **allow_headers**：允许的请求头。

---

### 22. FastAPI 与数据库事务管理

在使用数据库时，事务管理是保证数据一致性的重要机制。FastAPI 支持通过依赖注入和数据库会话对象实现事务管理。

#### 示例：使用 SQLAlchemy 管理数据库事务

```python
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from .database import SessionLocal, engine

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/items/")
async def create_item(name: str, db: Session = Depends(get_db)):
    db_item = Item(name=name)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

##### 详解：

- **get_db**：通过依赖注入获取数据库会话，使用 `yield` 保证在请求结束后关闭会话。
- **Session**：SQLAlchemy 的数据库会话对象，用于管理数据库事务（`add`、`commit`、`refresh` 等操作）。

---

### 23. FastAPI 与异步数据库交互

FastAPI 支持异步数据库操作，可以结合异步数据库驱动（如 `asyncpg`）和异步 ORM（如 Tortoise ORM）实现高效的数据库交互。

#### 示例：使用 Tortoise ORM

```python
from fastapi import FastAPI
from tortoise.contrib.fastapi import register_tortoise
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str

@app.post("/items/")
async def create_item(item: Item):
    item_obj = await Item.create(**item.dict())
    return item_obj

register_tortoise(
    app,
    db_url="sqlite://db.sqlite3",
    modules={"models": ["models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)
```

##### 详解：

- **Tortoise ORM**：一个异步的 Python ORM，支持与 FastAPI 结合进行异步数据库操作。
- **register_tortoise**：用于注册数据库配置和自动生成数据库表。

---

### 24. FastAPI 异步任务队列

FastAPI 支持将一些需要长时间运行的任务异步处理，可以使用任务队列（如 Celery）结合 FastAPI 实现后台任务处理。

#### 示例：FastAPI 与 Celery 结合

1. 安装 Celery：

```bash
pip install celery
```

2. 创建 `celery.py`：

```python
from celery import Celery

celery_app = Celery("worker", broker="redis://localhost:6379/0")

@celery_app.task
def add(x, y):
    return x + y
```

3. FastAPI 中调用异步任务：

```python
from fastapi import FastAPI
from .celery import celery_app

app = FastAPI()

@app.post("/tasks/")
async def create_task(x: int, y: int):
    task = celery_app.send_task("celery.add", args=[x, y])
    return {"task_id": task.id}
```

##### 详解：

- **Celery**：一个流行的分布式任务队列，适合处理异步任务和后台处理。
- **send_task**：调用 Celery 的任务队列，异步执行任务。

---

### 25. FastAPI WebSocket 的高级应用

FastAPI 的 WebSocket 支持不仅适合简单的双向通信，还可以处理复杂的场景，如 WebSocket 组、广播消息等。

#### 示例：WebSocket 组广播

```python
from fastapi import FastAPI, WebSocket
from typing import List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client says: {data}")
    finally:
        await manager.disconnect(websocket)
```

##### 详解：

- **ConnectionManager**：管理所有的 WebSocket 连接，实现广播消息的功能。
- **broadcast**：将接收到的消息发送给所有连接的客户端，实现组消息推送。

---
