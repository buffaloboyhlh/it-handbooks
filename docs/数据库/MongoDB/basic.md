# MongoDB 完整教程：从入门到精通

#### 目录：

1. MongoDB 简介
2. MongoDB 的基本概念
3. MongoDB 安装与配置
4. 数据库与集合操作
5. MongoDB CRUD 操作详解
6. MongoDB 高级查询
7. 索引与性能优化
8. 聚合框架详解
9. 复制集与分片
10. 备份与恢复
11. 安全与用户管理
12. MongoDB 性能优化与最佳实践
13. MongoDB 常见问题排查

---

### 1. MongoDB 简介

#### 1.1 什么是 MongoDB？
MongoDB 是一种基于文档的 NoSQL 数据库，采用 BSON（二进制 JSON）的格式存储数据，提供了灵活的、非结构化的文档数据存储方式。与传统的关系型数据库（如 MySQL、PostgreSQL）不同，MongoDB 无需定义严格的表结构，适用于海量、动态变化的数据需求场景。

#### 1.2 MongoDB 的核心特点
- **无模式结构：** 允许每个文档有不同的结构，灵活性极高。
- **高扩展性：** 支持分布式存储和横向扩展，适合大数据场景。
- **高可用性：** 支持复制集和自动故障转移，确保数据可靠性。
- **丰富的查询语言：** 提供强大的查询、排序、投影和聚合功能。

---

### 2. MongoDB 的基本概念

- **文档 (Document)：** 数据库中存储的基本单元，是键值对的集合，类似于 JSON 对象。
- **集合 (Collection)：** 文档的容器，类似于关系数据库中的表，但没有固定的结构。
- **数据库 (Database)：** 一组集合的容器。
- **BSON (Binary JSON)：** MongoDB 使用的文档存储格式，支持更多的数据类型。
- **主键 (_id)：** 每个文档都有一个唯一的 `_id` 字段，类似于关系数据库中的主键。

---

### 3. MongoDB 安装与配置

#### 3.1 安装 MongoDB

##### Windows 安装：
1. 从 [MongoDB 官方网站](https://www.mongodb.com/try/download/community) 下载并安装 MongoDB 社区版。
2. 配置环境变量，将 MongoDB 的 `bin` 目录添加到系统的 `PATH` 中。
3. 在命令提示符中输入 `mongod` 启动 MongoDB 服务器。

##### Ubuntu 安装：
```bash
# 更新包管理器
sudo apt update

# 安装 MongoDB
sudo apt install -y mongodb

# 启动 MongoDB 服务
sudo systemctl start mongodb

# 确认服务已启动
sudo systemctl status mongodb
```

##### macOS 安装 (使用 Homebrew):
```bash
# 更新 Homebrew
brew update

# 安装 MongoDB
brew tap mongodb/brew
brew install mongodb-community@6.0

# 启动 MongoDB
brew services start mongodb-community@6.0
```

#### 3.2 配置文件
MongoDB 使用 `mongod.conf` 文件进行配置。常见配置选项包括：
- **`storage.dbPath`**: 数据库存储路径。
- **`net.port`**: MongoDB 服务端口，默认是 `27017`。
- **`replication.replSetName`**: 设置复制集名称。

修改配置后，可以通过以下命令启动 MongoDB：
```bash
mongod --config /path/to/mongod.conf
```

---

### 4. 数据库与集合操作

#### 4.1 创建/切换数据库
```javascript
use myDatabase; // 创建或切换到数据库 myDatabase
```

#### 4.2 查看当前数据库
```javascript
db; // 显示当前数据库名称
```

#### 4.3 删除数据库
```javascript
db.dropDatabase(); // 删除当前数据库
```

#### 4.4 创建集合
```javascript
db.createCollection("myCollection");
```

#### 4.5 查看集合
```javascript
show collections; // 查看当前数据库下所有集合
```

#### 4.6 删除集合
```javascript
db.myCollection.drop(); // 删除集合
```

---

### 5. MongoDB CRUD 操作详解

#### 5.1 插入文档

- 插入单个文档：
```javascript
db.myCollection.insertOne({name: "Alice", age: 25});
```

- 插入多个文档：
```javascript
db.myCollection.insertMany([
    {name: "Bob", age: 30},
    {name: "Charlie", age: 35}
]);
```

#### 5.2 查询文档

- 查询所有文档：
```javascript
db.myCollection.find();
```

- 条件查询（如：查找年龄大于 30 的文档）：
```javascript
db.myCollection.find({age: {$gt: 30}});
```

- 显示特定字段（如：只显示 name 字段，隐藏 _id）：
```javascript
db.myCollection.find({}, {name: 1, _id: 0});
```

- 查询单个文档：
```javascript
db.myCollection.findOne({name: "Alice"});
```

#### 5.3 更新文档

- 更新单个文档：
```javascript
db.myCollection.updateOne(
    {name: "Alice"}, // 查询条件
    {$set: {age: 26}} // 更新操作
);
```

- 更新多个文档：
```javascript
db.myCollection.updateMany(
    {age: {$lt: 30}}, // 条件：年龄小于 30
    {$set: {status: "young"}} // 更新字段
);
```

- 替换整个文档：
```javascript
db.myCollection.replaceOne(
    {name: "Alice"}, 
    {name: "Alice", age: 27, status: "active"}
);
```

#### 5.4 删除文档

- 删除单个文档：
```javascript
db.myCollection.deleteOne({name: "Bob"});
```

- 删除多个文档：
```javascript
db.myCollection.deleteMany({status: "young"});
```

---

### 6. MongoDB 高级查询

MongoDB 的查询语言功能强大，支持多种条件和表达式。

#### 6.1 比较运算符
- `$eq`: 等于
- `$ne`: 不等于
- `$gt`: 大于
- `$lt`: 小于
- `$gte`: 大于等于
- `$lte`: 小于等于

```javascript
db.myCollection.find({age: {$gte: 30}});
```

#### 6.2 逻辑运算符
- `$and`: 与
- `$or`: 或
- `$not`: 非
- `$nor`: 也不

```javascript
db.myCollection.find({
    $or: [
        {age: {$lt: 20}},
        {age: {$gt: 40}}
    ]
});
```

#### 6.3 正则表达式查询
```javascript
db.myCollection.find({name: /^A/}); // 查找以 A 开头的名称
```

---

### 7. 索引与性能优化

#### 7.1 创建索引
索引用于加快查询速度，可以为一个或多个字段创建索引。

```javascript
db.myCollection.createIndex({name: 1}); // 为 name 字段创建升序索引
```

#### 7.2 查看索引
```javascript
db.myCollection.getIndexes();
```

#### 7.3 删除索引
```javascript
db.myCollection.dropIndex({name: 1});
```

#### 7.4 索引最佳实践
- **频繁查询的字段**：经常用作查询条件的字段应创建索引。
- **唯一索引**：为确保唯一性，可以创建唯一索引。
- **复合索引**：根据查询中涉及的多个字段创建复合索引。

---

### 8. 聚合框架详解

聚合框架提供了类似 SQL 的 `GROUP BY` 功能，通过管道操作符处理和转换数据。

#### 8.1 聚合基本示例
```javascript
db.myCollection.aggregate([
    {$match: {status: "active"}}, // 过滤条件
    {$group: {_id: "$age", total: {$sum: 1}}}, // 按年龄分组，并统计数量
    {$sort: {total: -1}} // 按总数降序排序
]);
```

#### 8.2 聚合阶段
- `$match`: 筛选数据，类似 SQL 中的 `WHERE`。
- `$group`: 按字段分组，类似 SQL 中的 `GROUP BY`。
- `$project`: 投影字段，选择输出的字段。
- `$sort`: 排序结果。

---

### 9. 复制集与分片

#### 9.1 复制集
复制集用于实现高可用性，数据会自动复制到多个节点上。一个复制集至少包含三个节点：**主节点**、**辅助节点** 和 **仲裁节点**。

```bash
#

 初始化复制集
rs.initiate({
  _id: "myReplSet",
  members: [
    {_id: 0, host: "localhost:27017"},
    {_id: 1, host: "localhost:27018"},
    {_id: 2, host: "localhost:27019"}
  ]
});
```

#### 9.2 分片
分片用于横向扩展数据库，MongoDB 将数据分散到多个节点上。

---

### 10. 备份与恢复

#### 10.1 备份数据库
```bash
mongodump --db myDatabase --out /path/to/backup
```

#### 10.2 恢复数据库
```bash
mongorestore /path/to/backup
```

---

### 11. 安全与用户管理

#### 11.1 创建用户
```javascript
db.createUser({
    user: "admin",
    pwd: "password",
    roles: [{role: "readWrite", db: "myDatabase"}]
});
```

#### 11.2 用户认证
启用用户认证后，连接 MongoDB 需要提供用户名和密码：
```bash
mongod --auth
```

---

### 12. MongoDB 性能优化与最佳实践

- **使用适当的索引**：避免为每个字段都创建索引。
- **合理分片**：在大数据量场景下，使用分片分布数据。
- **定期监控性能**：使用 MongoDB 的 `profiler` 和 `explain` 分析慢查询。

---

### 13. MongoDB 常见问题排查

#### 13.1 问题：连接失败
**解决方法：**
- 检查 MongoDB 服务是否启动。
- 确认 MongoDB 正在监听正确的端口（默认 `27017`）。

#### 13.2 问题：查询速度慢
**解决方法：**
- 使用 `explain()` 检查查询性能。
- 为查询字段添加索引。

---

通过本教程，你可以从 MongoDB 的基础知识学起，并逐步掌握其高级功能，最终能够在复杂应用中熟练使用 MongoDB。