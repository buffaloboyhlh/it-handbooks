# FastAPI 进阶教程

### 1. FastAPI 中的依赖注入（Dependency Injection）

依赖注入是 FastAPI 的核心功能之一，可以将某些共享逻辑（如数据库连接、验证逻辑等）抽象成依赖项，减少代码重复并提高可维护性。

#### 示例：依赖注入的基本用法

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def get_query_param(q: str = None):
    return q

@app.get("/items/")
async def read_items(q: str = Depends(get_query_param)):
    return {"query_param": q}
```

#### 详解：

- **Depends**：通过 `Depends` 函数声明依赖项，`get_query_param` 的返回值会被注入到 `read_items` 路由中。
- **依赖复用**：相同的依赖函数可以在多个路由中复用，大幅减少重复代码。

---

### 2. 自定义依赖项

FastAPI 的依赖项不仅可以是简单的函数，还可以是带有复杂逻辑的类或者协程。

#### 示例：类作为依赖项

```python
from fastapi import Depends, FastAPI

app = FastAPI()

class QueryChecker:
    def __init__(self, q: str = None):
        self.q = q

    def check(self):
        if self.q:
            return f"Query param is: {self.q}"
        return "No query param provided"

@app.get("/items/")
async def read_items(checker: QueryChecker = Depends()):
    return {"message": checker.check()}
```

#### 详解：

- **类作为依赖**：通过定义类，可以封装更复杂的业务逻辑，依赖项会自动实例化并注入到路由中。
- **状态维护**：类中的 `self.q` 可以保存状态，适用于一些需要维护内部状态的业务场景。

---

### 3. 依赖项的作用域控制

依赖项可以设置不同的作用域，包括：
- **`request`**：每次请求都会创建新的依赖实例。
- **`session`**：生命周期内维持一个依赖实例。

#### 示例：依赖项作用域

```python
from fastapi import Depends, FastAPI
from contextlib import contextmanager

app = FastAPI()

@contextmanager
def db_session():
    db = "database_session"
    try:
        yield db
    finally:
        print("Close the DB session")

@app.get("/items/")
async def read_items(db: str = Depends(db_session)):
    return {"db_session": db}
```

#### 详解：

- **contextmanager**：通过 Python 的 `contextmanager` 实现依赖项的生命周期管理。
- **依赖项作用域**：每次请求都会启动一个数据库会话，并在请求结束后自动关闭。

---

### 4. FastAPI 与 SQLAlchemy 深入结合

在实际项目中，通常会使用 ORM 框架进行数据库操作，SQLAlchemy 是一个常用的 ORM 框架。通过依赖注入和 FastAPI，SQLAlchemy 可以高效管理数据库会话。

#### 示例：SQLAlchemy 数据库会话管理

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends, FastAPI

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
async def read_items(db: Session = Depends(get_db)):
    # 使用 db 进行数据库操作
    return {"message": "Database session in use"}
```

#### 详解：

- **sessionmaker**：创建一个数据库会话工厂，每次请求都会生成一个新的会话。
- **get_db**：通过依赖注入 `get_db` 函数，可以在请求生命周期内管理数据库会话，避免手动处理会话关闭。

---

### 5. FastAPI 与异步数据库操作

FastAPI 支持异步操作数据库，使用诸如 `asyncpg` 和 `Tortoise ORM` 等异步数据库驱动，可以大幅提高 I/O 密集型操作的性能。

#### 示例：使用 Tortoise ORM

```python
from fastapi import FastAPI
from tortoise.contrib.fastapi import register_tortoise

app = FastAPI()

class ItemModel:
    # 定义数据库模型
    pass

@app.get("/items/")
async def read_items():
    items = await ItemModel.all()
    return items

register_tortoise(
    app,
    db_url="sqlite://db.sqlite3",
    modules={"models": ["your_project.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)
```

#### 详解：

- **Tortoise ORM**：一个异步的 ORM，支持与 FastAPI 完美结合，提供自动化的数据库迁移和模型管理。
- **register_tortoise**：通过 `register_tortoise` 函数，将 Tortoise ORM 注册到 FastAPI 应用中，并自动生成数据库表。

---

### 6. FastAPI 的后台任务处理

在某些情况下，我们需要在处理完请求后执行一些异步任务。FastAPI 提供了后台任务处理的功能，允许任务在响应后异步执行。

#### 示例：使用 BackgroundTasks 处理后台任务

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def write_log(message: str):
    with open("log.txt", "a") as log_file:
        log_file.write(message)

@app.post("/send-notification/")
async def send_notification(background_tasks: BackgroundTasks, message: str):
    background_tasks.add_task(write_log, message)
    return {"message": "Notification sent in background"}
```

#### 详解：

- **BackgroundTasks**：FastAPI 提供的后台任务管理器，可以通过 `add_task` 方法将任务放入后台异步执行。
- **后台任务**：任务可以在响应返回后继续执行，不会阻塞主线程。

---

### 7. FastAPI 中的中间件（Middleware）

中间件可以在请求到达路由之前或者响应返回客户端之前进行额外的操作，如修改请求、响应或者进行身份验证。

#### 示例：自定义中间件

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Custom-Header"] = "Value"
        return response

app.add_middleware(CustomMiddleware)
```

#### 详解：

- **BaseHTTPMiddleware**：实现自定义中间件的基类，通过 `dispatch` 方法处理请求和响应。
- **add_middleware**：在 FastAPI 中注册自定义中间件，每次请求会经过这个中间件进行处理。

---

### 8. FastAPI 与异步任务队列 Celery 的集成

在高并发场景下，FastAPI 可以与 Celery 等任务队列系统结合，处理一些需要长时间执行的任务，如发送邮件、生成报告等。

#### 示例：FastAPI 与 Celery 集成

1. 配置 Celery：

```python
from celery import Celery

celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)
```

2. FastAPI 中调用 Celery 任务：

```python
from fastapi import FastAPI
from .celery import celery_app

app = FastAPI()

@celery_app.task
def send_email(to: str):
    # 模拟发送邮件
    return f"Email sent to {to}"

@app.post("/send-email/")
async def send_email_endpoint(to: str):
    task = send_email.delay(to)
    return {"task_id": task.id}
```

#### 详解：

- **Celery**：一个流行的分布式任务队列系统，用于处理后台任务。
- **send_email.delay()**：通过 `delay()` 方法将任务放入队列，由 Celery 异步执行任务。

---

### 9. FastAPI 中的事件处理（Event Handling）

FastAPI 支持应用启动和关闭时执行特定的事件处理函数，通常用于在应用启动时建立数据库连接，或在关闭时释放资源。

#### 示例：启动和关闭事件

```python
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("Application is starting")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application is shutting down")
```

#### 详解：

- **@app.on_event**：通过 `on_event` 装饰器声明启动和关闭事件，可以在应用启动或关闭时执行一些初始化或清理操作。

---

### 10. FastAPI 的安全性提升

FastAPI 提供了多种安全相关的功能，如 OAuth2、API Key 认证、JWT 认证等，适合保护敏感的 API。

#### 示例：使用 OAuth2 密码流和 JWT

```python
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload

@app.get("/users/me")
async def read_users_me(token: str = Depends(verify_token)):
    return {"user": token}
```

#### 详解：

- **OAuth2PasswordBearer**：FastAPI 提供的 OAuth2 密码流，用于保护路由。
- **JWT**：JSON Web Token，用于认证用户身份，通过 `jose` 库进行编码和解码。

---

### 11. FastAPI 中的 WebSocket 实现

WebSocket 是一种在客户端和服务器之间建立全双工通信的协议，适合实时应用场景。FastAPI 支持 WebSocket 的实现，可以用于在线聊天、实时通知等场景。

#### 示例：实现 WebSocket 连接

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message received: {data}")
```

#### 详解：

- **WebSocket**：`WebSocket` 类提供了建立 WebSocket 连接的方法，`receive_text()` 接收客户端发送的消息，`send_text()` 返回响应。
- **全双工通信**：WebSocket 允许服务器与客户端之间的双向通信，无需每次发送 HTTP 请求。

---

### 12. FastAPI 的后台定时任务

FastAPI 并不直接支持定时任务，但是可以通过结合 `apscheduler` 等第三方库，实现定时任务功能。

#### 示例：使用 APScheduler 实现定时任务

```python
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI()

scheduler = BackgroundScheduler()

def periodic_task():
    print("This task runs periodically")

scheduler.add_job(periodic_task, "interval", seconds=10)
scheduler.start()

@app.get("/")
async def read_root():
    return {"message": "FastAPI with periodic tasks"}
```

#### 详解：

- **APScheduler**：用于在 FastAPI 应用中调度定时任务，支持以固定的时间间隔执行任务。
- **BackgroundScheduler**：后台调度器，可以在应用启动后自动运行。

---

### 13. FastAPI 与 Redis 缓存的结合

在处理高并发请求时，使用 Redis 进行缓存是提升性能的有效方式。通过 FastAPI 结合 Redis，可以实现数据缓存、会话管理等功能。

#### 示例：使用 Redis 缓存 API 响应

```python
import redis
from fastapi import FastAPI

app = FastAPI()
r = redis.Redis(host="localhost", port=6379, db=0)

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    cached_item = r.get(item_id)
    if cached_item:
        return {"item": cached_item.decode("utf-8")}
    item = f"Item with id {item_id}"  # 模拟从数据库获取的数据
    r.set(item_id, item)
    return {"item": item}
```

#### 详解：

- **Redis**：使用 Redis 作为缓存层，减少数据库查询次数，提升响应速度。
- **r.get() 和 r.set()**：从 Redis 获取和存储缓存的数据，支持多种数据类型的缓存。

---

### 14. FastAPI 中的依赖注入与数据库连接池结合

对于高并发场景，单次连接数据库可能成为性能瓶颈。通过连接池（如 `asyncpg`），可以高效管理数据库连接。

#### 示例：使用 asyncpg 连接池

```python
import asyncpg
from fastapi import FastAPI

app = FastAPI()

async def get_db_pool():
    pool = await asyncpg.create_pool(dsn="postgres://user:password@localhost:5432/dbname")
    return pool

@app.on_event("startup")
async def startup_event():
    app.state.db_pool = await get_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.db_pool.close()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    async with app.state.db_pool.acquire() as connection:
        item = await connection.fetchrow("SELECT * FROM items WHERE id = $1", item_id)
        return item
```

#### 详解：

- **asyncpg**：一个高效的异步 PostgreSQL 客户端库，支持连接池管理。
- **create_pool()**：创建数据库连接池，应用启动时创建，关闭时释放。

---

### 15. FastAPI 的测试和调试

为了确保应用的稳定性和正确性，编写单元测试非常重要。FastAPI 提供了与 `TestClient` 集成的简单方法，用于快速测试 API。

#### 示例：使用 TestClient 进行测试

```python
from fastapi.testclient import TestClient
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items():
    return {"items": ["item1", "item2"]}

client = TestClient(app)

def test_read_items():
    response = client.get("/items/")
    assert response.status_code == 200
    assert response.json() == {"items": ["item1", "item2"]}
```

#### 详解：

- **TestClient**：FastAPI 自带的测试客户端，基于 `requests`，可以模拟发送请求并检查响应结果。
- **单元测试**：通过编写测试函数，可以确保每个 API 的行为符合预期。

---

### 16. FastAPI 中的文件上传和下载

FastAPI 支持文件上传和下载操作，特别适合处理大文件和多文件上传。

#### 示例：实现文件上传

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "content": content.decode("utf-8")}
```

#### 详解：

- **UploadFile**：FastAPI 提供的文件上传类，支持异步处理上传文件，`read()` 方法读取文件内容。
- **File()**：声明上传文件参数，并可以指定文件的多种元数据。

---

### 17. FastAPI 与 GraphQL 的集成

FastAPI 支持与 GraphQL 集成，通过 `graphene` 库实现基于 GraphQL 的 API 开发，适合需要灵活查询和数据结构的项目。

#### 示例：使用 GraphQL 创建 API

```python
import graphene
from starlette.graphql import GraphQLApp
from fastapi import FastAPI

class Query(graphene.ObjectType):
    hello = graphene.String(name=graphene.String(default_value="stranger"))

    def resolve_hello(self, info, name):
        return f"Hello {name}"

app = FastAPI()
app.add_route("/graphql", GraphQLApp(schema=graphene.Schema(query=Query)))
```

#### 详解：

- **GraphQL**：一种查询语言，允许客户端灵活指定需要的数据，避免数据过多或不足的问题。
- **GraphQLApp**：FastAPI 提供了 `GraphQLApp`，可以快速集成 GraphQL 服务。

---

### 18. FastAPI 的 CORS 支持

在跨域请求的场景下，FastAPI 支持通过配置 CORS（跨源资源共享）规则来允许某些来源访问 API。

#### 示例：配置 CORS

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://myfrontendapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 详解：

- **CORSMiddleware**：FastAPI 的 CORS 中间件，允许配置跨域请求的来源、方法、头部等参数，确保前端应用能够正确访问 API。

---

### 19. FastAPI 与 JWT 结合的用户认证

通过 JWT（JSON Web Token）进行用户认证是 RESTful API 中常见的方式。FastAPI 与 JWT 可以很方便地结合，实现安全的用户身份验证。

#### 示例：使用 JWT 进行用户认证

```python
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "mysecret"
ALGORITHM = "HS256"

def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return decode_jwt(token)
```

#### 详解：

- **JWT**：用于认证和授权的令牌，包含用户的加密信息。
- **OAuth2PasswordBearer**：FastAPI 提供的 OAuth2 认证流，使用 JWT 进行令牌验证。

---

### 20. FastAPI 的自定义异常处理

自定义异常处理可以帮助你管理和处理特定的错误条件，从而提供更友好的用户反馈和更有效的错误追踪。

#### 示例：自定义异常处理

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Custom exception occurred with name: {exc.name}"},
    )

@app.get("/raise-custom-exception/{name}")
async def raise_custom_exception(name: str):
    raise CustomException(name=name)
```

#### 详解：

- **自定义异常类**：创建自定义异常类 `CustomException`。
- **异常处理器**：使用 `@app.exception_handler` 装饰器注册自定义异常处理器。
- **JSONResponse**：返回自定义格式的 JSON 响应。

---

### 21. FastAPI 的分页处理

分页是一种处理大量数据时的常见方法。FastAPI 可以结合查询参数实现简单的分页功能。

#### 示例：实现分页

```python
from fastapi import FastAPI, Query

app = FastAPI()

fake_items = [{"item": f"Item {i}"} for i in range(100)]

@app.get("/items/")
async def read_items(skip: int = Query(0, alias="page", ge=0), limit: int = Query(10, le=100)):
    start = skip * limit
    end = start + limit
    return fake_items[start:end]
```

#### 详解：

- **Query**：使用 `Query` 声明查询参数，并设置默认值和验证条件。
- **分页逻辑**：通过 `skip` 和 `limit` 参数计算数据的起始和结束位置，返回相应的分页数据。

---

### 22. FastAPI 的请求体模型（Pydantic）

FastAPI 使用 Pydantic 作为数据验证和序列化的工具。通过 Pydantic 的模型，可以确保请求数据符合预期的结构和类型。

#### 示例：使用 Pydantic 模型验证请求数据

```python
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item}
```

#### 详解：

- **BaseModel**：Pydantic 的基类，通过继承 `BaseModel` 定义数据模型。
- **数据验证**：请求体的数据会自动验证是否符合模型定义的类型和结构。

---

### 23. FastAPI 的路由参数和路径参数

FastAPI 支持路径参数和查询参数的灵活使用，可以通过这些参数接收和处理客户端的请求数据。

#### 示例：路径参数与查询参数

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

#### 详解：

- **路径参数**：通过 `{item_id}` 语法定义路径参数，并在函数中接收。
- **查询参数**：通过函数参数接收查询参数，支持默认值和类型检查。

---

### 24. FastAPI 的请求和响应模型

通过定义请求和响应模型，可以确保 API 的输入和输出数据符合预期的结构，增强代码的自文档化。

#### 示例：请求和响应模型

```python
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

class ResponseModel(BaseModel):
    item: Item
    message: str

@app.post("/items/", response_model=ResponseModel)
async def create_item(item: Item):
    return {"item": item, "message": "Item created successfully"}
```

#### 详解：

- **请求模型**：通过 `Item` 类定义请求体的结构。
- **响应模型**：通过 `ResponseModel` 类定义响应体的结构，并在 `@app.post` 装饰器中指定 `response_model`。

---

### 25. FastAPI 的静态文件服务

FastAPI 可以通过配置静态文件服务来处理静态资源，如图片、CSS 文件、JavaScript 文件等。

#### 示例：配置静态文件服务

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Static files are served at /static"}
```

#### 详解：

- **StaticFiles**：FastAPI 的静态文件服务类，用于指定静态文件的目录。
- **app.mount**：将静态文件目录挂载到指定的路径上。

---

### 26. FastAPI 的请求体和响应体示例

在开发 API 时，提供清晰的请求体和响应体示例可以帮助前端开发者更好地理解 API 接口的使用。

#### 示例：定义示例数据

```python
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    return item

@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    return {"name": "Item", "description": "Item description", "price": 123.45}
```

#### 详解：

- **示例数据**：在 `response_model` 中定义的模型会自动生成示例数据，这些示例数据会在 API 文档中显示，帮助用户了解请求和响应的数据格式。

---

### 27. FastAPI 的异步编程支持

FastAPI 原生支持异步编程，可以使用 `async/await` 语法进行异步 I/O 操作，提高应用的性能和响应速度。

#### 示例：异步数据库查询

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

async def async_query():
    await asyncio.sleep(1)  # 模拟异步 I/O 操作
    return {"data": "Async data"}

@app.get("/async-items/")
async def read_async_items():
    data = await async_query()
    return data
```

#### 详解：

- **async/await**：通过 `async/await` 语法进行异步操作，适合 I/O 密集型任务，如数据库查询、网络请求等。
- **async_query**：一个模拟的异步查询函数，演示如何处理异步操作。

---

### 28. FastAPI 的中间件和插件扩展

FastAPI 支持通过中间件和插件进行功能扩展，如日志记录、请求限流等。

#### 示例：使用中间件记录请求日志

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger = logging.getLogger("uvicorn")
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        return response

app.add_middleware(LoggingMiddleware)
```

#### 详解：

- **BaseHTTPMiddleware**：自定义中间件基类，允许在请求和响应之间进行处理。
- **LoggingMiddleware**：一个简单的日志中间件，记录请求的基本信息。

---

### 29. FastAPI 的错误日志和监控

错误日志和监控对于生产环境中的应用至关重要。通过集成日志记录和监控工具，可以实时了解应用的运行状态。

#### 示例：集成日志记录

```python
import logging
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_logger")

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id < 0:
        logger.error(f"Invalid item_id: {item_id}")
        raise HTTPException(status_code=400, detail="Invalid item ID")
    return {"item_id": item_id}
```

#### 详解：

- **logging**：Python 的标准库用于记录应用的日志。
- **logger.error**：记录错误信息，并在发生错误时输出到日志中。

---

### 30. FastAPI 的文档自定义

FastAPI 自动生成的 API 文档非常强大，但你也可以通过自定义文档来满足特定需求。

#### 示例：自定义 OpenAPI 文档

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="This is a very fancy project",
    version="1.0.0",
)

@app.get("/items/")
async def read_items():
    return {"message": "Custom API Documentation"}
```

#### 详解：

- **FastAPI 初始化参数**：通过 `title`、`description` 和 `version` 参数自定义 API 文档的标题、描述和版本。
- **文档生成**：FastAPI 会自动生成并提供 OpenAPI 和 Swagger UI 文档。

---

### 31. FastAPI 的数据库集成

FastAPI 可以与多种数据库集成，包括关系型数据库和非关系型数据库。这里我们以 SQLAlchemy 和 Tortoise ORM 为例，演示如何在 FastAPI 中使用它们。

#### 使用 SQLAlchemy

**示例：SQLAlchemy 与 FastAPI 集成**

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI()

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, index=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/items/")
async def create_item(name: str, description: str, db: Depends(get_db)):
    db_item = Item(name=name, description=description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/{item_id}")
async def read_item(item_id: int, db: Depends(get_db)):
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item
```

**详解：**

- **create_engine**：创建数据库引擎，连接到指定的数据库。
- **SessionLocal**：数据库会话工厂，用于创建和管理数据库会话。
- **Base**：声明基础模型类，所有模型继承自该类。
- **get_db**：依赖注入函数，管理数据库会话的生命周期。

#### 使用 Tortoise ORM

**示例：Tortoise ORM 与 FastAPI 集成**

```python
from fastapi import FastAPI, HTTPException
from tortoise import Tortoise, fields
from tortoise.models import Model
from tortoise.contrib.fastapi import HTTPNotFoundError, register_tortoise

app = FastAPI()

class Item(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)

register_tortoise(app, db_url='sqlite://db.sqlite3', modules={'models': ['__main__']}, generate_schemas=True)

@app.post("/items/")
async def create_item(name: str, description: str = None):
    item = await Item.create(name=name, description=description)
    return item

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    item = await Item.get_or_none(id=item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

**详解：**

- **register_tortoise**：注册 Tortoise ORM 与 FastAPI 集成，自动处理数据库连接和模式生成。
- **Item**：模型类，定义数据表结构。

---

### 32. FastAPI 的安全性最佳实践

确保 API 的安全性是开发过程中至关重要的部分。这里介绍一些 FastAPI 的安全性最佳实践，包括使用 HTTPS、限制请求速率等。

#### 使用 HTTPS

**示例：通过 Uvicorn 启动 HTTPS**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem
```

**详解：**

- **--ssl-keyfile 和 --ssl-certfile**：指定 SSL 证书和密钥文件，实现 HTTPS 加密通信。

#### 限制请求速率

**示例：使用 `slowapi` 限制请求速率**

```python
from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/limited/")
@limiter.limit("5/minute")
async def limited_route():
    return {"message": "This route is rate limited"}
```

**详解：**

- **Limiter**：配置速率限制器，限制特定路由的请求速率。
- **get_remote_address**：获取客户端 IP 地址作为速率限制的依据。

---

### 33. FastAPI 的多语言支持

如果你的应用需要支持多语言，可以使用 FastAPI 与 `gettext` 结合，实现国际化（i18n）功能。

#### 示例：使用 `gettext` 实现多语言支持

**安装依赖：**

```bash
pip install gettext
```

**代码示例：**

```python
from fastapi import FastAPI
from gettext import translation

app = FastAPI()

def get_translations(lang: str):
    return translation(domain="messages", localedir="locale", languages=[lang])

@app.get("/greet/{lang}")
async def greet(lang: str):
    translations = get_translations(lang)
    return {"message": translations.gettext("Hello, world!")}
```

**详解：**

- **gettext**：用于加载和管理翻译文件，支持多语言文本。
- **translations.gettext**：根据当前语言返回翻译后的文本。

---

### 34. FastAPI 的异步任务队列

对于长时间运行的任务或批量处理，可以使用异步任务队列，如 `Celery`。

#### 示例：使用 Celery 实现异步任务

**安装依赖：**

```bash
pip install celery[redis] fastapi
```

**代码示例：**

```python
from fastapi import FastAPI, BackgroundTasks
from celery import Celery

app = FastAPI()
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task
def background_task(param):
    # 处理任务
    return f"Processed {param}"

@app.post("/tasks/")
async def create_task(param: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(background_task, param)
    return {"message": "Task has been started"}
```

**详解：**

- **Celery**：一个异步任务队列，支持任务调度和执行。
- **BackgroundTasks**：FastAPI 提供的后台任务处理器，用于管理和调度后台任务。

---

### 35. FastAPI 的 OAuth2 和 OpenID Connect 支持

使用 OAuth2 和 OpenID Connect 实现更复杂的认证和授权机制。

#### 示例：OAuth2 认证

```python
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel

app = FastAPI()

oauth2_scheme = OAuth2AuthorizationCodeBearer(authorizationUrl="authorization_url", tokenUrl="token_url")

class User(BaseModel):
    username: str

@app.get("/users/me", response_model=User)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # 验证 token 并获取用户信息
    return {"username": "user"}
```

**详解：**

- **OAuth2AuthorizationCodeBearer**：处理 OAuth2 授权码流的安全认证。
- **token 验证**：使用依赖注入对 token 进行验证和解析。

---

### 36. FastAPI 的 Swagger 和 ReDoc 自定义

FastAPI 自动生成的 API 文档（Swagger 和 ReDoc）可以进行自定义，以满足特定的需求。

#### 示例：自定义 Swagger 文档

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="This is a custom API",
    version="1.0.0",
    openapi_tags=[{"name": "items", "description": "Operations with items"}]
)

@app.get("/items/", tags=["items"])
async def read_items():
    return {"items": ["item1", "item2"]}
```

**详解：**

- **title、description、version**：自定义 API 文档的基本信息。
- **openapi_tags**：定义 API 文档中的标签和描述信息。

---
