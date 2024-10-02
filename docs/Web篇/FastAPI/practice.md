### FastAPI 实战项目教程：构建完整的博客 API

本教程将带您逐步完成一个博客 API 项目的开发，涵盖用户认证、文章管理、评论系统、数据库操作、测试和部署等功能。整个项目采用 FastAPI、SQLAlchemy 以及 SQLite 数据库，通过 JWT 进行身份验证，并展示如何组织项目结构以保持代码清晰和易于维护。

### 目录
1. 项目概述
2. 环境搭建
3. 项目结构
4. 用户模块（注册、登录、JWT 认证）
5. 文章模块（CRUD 操作）
6. 评论模块
7. 数据库集成与操作
8. 项目测试
9. 项目部署

---

## 1. 项目概述

在本项目中，我们将开发一个 RESTful 风格的博客 API，具备以下功能：
- 用户注册与登录
- 基于 JWT 的用户认证
- 文章的创建、查看、更新、删除（CRUD）
- 评论系统
- 数据库操作集成（SQLAlchemy + SQLite）
- 项目测试
- 部署在生产环境中

---

## 2. 环境搭建

### 2.1 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # 激活虚拟环境
```

### 2.2 安装 FastAPI、Uvicorn 和 SQLAlchemy

```bash
pip install fastapi uvicorn sqlalchemy sqlite3 passlib[bcrypt] python-jose pydantic
```

- `fastapi`：FastAPI 框架
- `uvicorn`：ASGI 服务器
- `sqlalchemy`：ORM（对象关系映射）工具
- `passlib` 和 `python-jose`：用于密码哈希和 JWT 处理
- `pydantic`：用于数据验证

---

## 3. 项目结构

首先，设计一个清晰的项目结构以保持代码模块化和可维护性。

```plaintext
blog_api_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py
│   │   ├── articles.py
│   │   └── comments.py
│   ├── schemas.py
│   └── auth.py
├── tests/
│   └── test_users.py
├── .env
└── Dockerfile
```

- `app/`：主要应用代码，包括数据库、模型、API 路由等。
- `routers/`：不同功能模块的 API 路由。
- `schemas.py`：定义请求和响应数据结构。
- `auth.py`：JWT 认证逻辑。
- `tests/`：单元测试文件。

---

## 4. 用户模块

用户模块负责处理用户的注册、登录以及 JWT 认证。

### 4.1 创建数据库模型

在 `models.py` 中定义用户的数据库模型。

```python
from sqlalchemy import Column, Integer, String
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
```

### 4.2 JWT 认证与密码哈希

在 `auth.py` 中实现 JWT 生成和验证，以及密码的哈希处理。

```python
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

### 4.3 用户注册和登录

在 `routers/users.py` 中实现用户注册和登录逻辑。

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import schemas, models, auth
from ..database import get_db

router = APIRouter()

@router.post("/register", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(email=user.email, username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/login")
def login_user(form_data: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.email).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = auth.create_access_token({"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}
```

### 4.4 数据验证 (schemas.py)

```python
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    username: str
    password: str

class User(UserBase):
    id: int
    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str
```

---

## 5. 文章模块

### 5.1 定义文章模型 (models.py)

```python
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User")
```

### 5.2 CRUD 操作 (routers/articles.py)

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter()

@router.post("/", response_model=schemas.Article)
def create_article(article: schemas.ArticleCreate, db: Session = Depends(get_db)):
    db_article = models.Article(**article.dict())
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

@router.get("/{article_id}", response_model=schemas.Article)
def read_article(article_id: int, db: Session = Depends(get_db)):
    db_article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if db_article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return db_article
```

### 5.3 数据验证 (schemas.py)

```python
class ArticleBase(BaseModel):
    title: str
    content: str

class ArticleCreate(ArticleBase):
    pass

class Article(ArticleBase):
    id: int
    owner_id: int
    class Config:
        orm_mode = True
```

---

## 6. 评论模块

类似于文章模块，评论模块包含创建、查看等功能，并与文章关联。

---

## 7. 数据库集成与操作

`database.py` 用于创建数据库引擎、会话和基础模型类。

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

---

## 8. 项目测试

在 `tests/` 文件夹中编写测试用例，使用 FastAPI 提供的 `TestClient` 进行测试。

---

## 9. 项目部署

使用 `Uvicorn` 或 `Gunicorn` 部署 FastAPI 应用，或者通过 Docker 容器化。

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

通过这份教程，您可以完整构建一个基于 FastAPI 的博客 API 项目。