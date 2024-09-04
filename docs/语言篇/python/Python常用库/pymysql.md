# pymysql 模块

### PyMySQL 详解

`PyMySQL` 是一个纯 Python 实现的 MySQL 客户端库，用于连接 MySQL 数据库并执行 SQL 查询。它与 MySQL 数据库的交互不需要依赖任何 MySQL 的 C 语言库，因此在不同平台上具有良好的兼容性。以下是 `PyMySQL` 的详细讲解。

#### 1. 安装 PyMySQL

在使用 `PyMySQL` 之前，需要通过 `pip` 安装它：

```bash
pip install pymysql
```

#### 2. 基本用法

##### 2.1 连接到 MySQL 数据库

要连接到 MySQL 数据库，首先需要创建一个连接对象：

```python
import pymysql

# 创建数据库连接
connection = pymysql.connect(
    host='localhost',  # MySQL服务器地址
    user='your_username',  # 用户名
    password='your_password',  # 密码
    database='your_database',  # 数据库名称
    charset='utf8mb4',  # 编码格式
    cursorclass=pymysql.cursors.DictCursor  # 可选，返回字典格式的结果
)
```

##### 2.2 创建游标并执行 SQL 语句

一旦连接成功，你可以创建一个游标来执行 SQL 查询：

```python
# 创建游标
cursor = connection.cursor()

# 执行查询
cursor.execute("SELECT VERSION()")

# 获取单条结果
version = cursor.fetchone()
print(f"Database version: {version['VERSION()']}")
```

##### 2.3 插入数据

执行插入操作可以通过 `execute()` 函数来完成：

```python
sql = "INSERT INTO users (name, age) VALUES (%s, %s)"
data = ('Alice', 25)

cursor.execute(sql, data)

# 提交事务
connection.commit()
```

`PyMySQL` 使用参数化查询来防止 SQL 注入攻击。这里的 `%s` 是占位符，后面的数据会自动填充到 SQL 语句中。

##### 2.4 查询数据

你可以使用 `SELECT` 语句来查询数据，并使用 `fetchone()` 或 `fetchall()` 获取结果：

```python
sql = "SELECT id, name, age FROM users WHERE age > %s"
cursor.execute(sql, (20,))

# 获取所有匹配的记录
results = cursor.fetchall()

for row in results:
    print(f"ID: {row['id']}, Name: {row['name']}, Age: {row['age']}")
```

##### 2.5 更新和删除数据

更新和删除操作与插入类似，使用 `execute()` 方法执行 SQL 语句：

```python
# 更新数据
sql = "UPDATE users SET age = %s WHERE name = %s"
cursor.execute(sql, (30, 'Alice'))
connection.commit()

# 删除数据
sql = "DELETE FROM users WHERE name = %s"
cursor.execute(sql, ('Alice',))
connection.commit()
```

#### 3. 事务处理

`PyMySQL` 默认情况下是自动提交模式，但你可以手动管理事务以确保数据的一致性。

```python
try:
    # 禁用自动提交
    connection.autocommit(False)

    # 开始一个新的事务
    cursor.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    cursor.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")

    # 提交事务
    connection.commit()
except Exception as e:
    # 如果出现错误，回滚事务
    connection.rollback()
    print(f"Transaction failed: {e}")
finally:
    # 恢复自动提交
    connection.autocommit(True)
```

#### 4. 连接池

对于高并发的应用，使用连接池可以提高效率，`PyMySQL` 本身不提供连接池功能，但可以结合第三方库 `pymysqlpool` 或 `SQLAlchemy` 等来实现连接池。

```python
from pymysqlpool.pool import Pool

pool = Pool(host='localhost', user='your_username', password='your_password', database='your_database', autocommit=True)
pool.init()

# 从连接池获取连接
connection = pool.get_conn()
```

#### 5. 处理异常

`PyMySQL` 提供了一系列异常类用于处理常见的数据库错误，如 `OperationalError`, `IntegrityError` 等。

```python
try:
    connection = pymysql.connect(
        host='localhost',
        user='your_username',
        password='your_password',
        database='your_database'
    )
except pymysql.MySQLError as e:
    print(f"Error connecting to MySQL: {e}")
```

#### 6. 关闭连接

操作完成后，记得关闭游标和连接以释放资源：

```python
cursor.close()
connection.close()
```

#### 7. 总结

`PyMySQL` 是一个简单易用且功能强大的 MySQL 客户端库，适合在各种 Python 项目中使用。它提供了对 MySQL 数据库的全面支持，并允许开发者在 Python 中以直观的方式执行数据库操作。通过掌握 `PyMySQL` 的基本用法和高级功能，你可以在项目中轻松实现数据库操作。