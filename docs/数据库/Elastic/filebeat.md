# filebeat 教程

### Filebeat 从入门到精通教程

Filebeat 是一个轻量级的日志传输工具，主要用于将服务器和应用程序生成的日志数据发送到 Logstash、Elasticsearch 或其他存储服务。它是 Elastic Stack（ELK Stack）的一部分，常用于实时日志处理和监控。本文将带你从 Filebeat 的基础入门到一些高级功能的掌握，帮助你全面掌握 Filebeat。

---

## 目录
1. **Filebeat 基础概念**
2. **Filebeat 安装与启动**
3. **Filebeat 配置详解**
   - 输入源（Inputs）
   - 输出目标（Outputs）
4. **Filebeat 模块化配置**
   - 模块简介
   - 启用和管理模块
5. **Filebeat 与 Logstash/Elasticsearch 集成**
6. **Filebeat 高级功能**
   - 多输入与多输出
   - 自定义字段与标签
7. **Filebeat 性能优化**
8. **Filebeat 常见问题排查**
9. **Filebeat 实战案例**

---

## 1. Filebeat 基础概念

Filebeat 是 Elastic Stack 中的“轻量级代理”，专门用于转发和集中化日志数据。它可以通过以下几种方式使用：

- **直接发送到 Elasticsearch**：Filebeat 可以直接将日志数据发送到 Elasticsearch，以供实时查询、分析和可视化。
- **通过 Logstash 处理后发送到 Elasticsearch**：使用 Logstash 处理数据时，可以将 Filebeat 日志数据发送到 Logstash 进行进一步的过滤、处理和增强，然后再传递给 Elasticsearch。
- **通过模块进行日志解析**：Filebeat 包含大量预配置模块，能够自动解析特定应用程序的日志。

---

## 2. Filebeat 安装与启动

Filebeat 支持多种操作系统，下面将以 CentOS 和 Ubuntu 为例，介绍如何安装和启动 Filebeat。

### 2.1 在 CentOS 上安装 Filebeat

```bash
# 导入 Elastic 公钥
sudo rpm --import https://artifacts.elastic.co/GPG-KEY-elasticsearch

# 创建 Filebeat YUM 仓库
sudo tee /etc/yum.repos.d/elastic.repo <<EOF
[elastic-8.x]
name=Elastic repository for 8.x packages
baseurl=https://artifacts.elastic.co/packages/8.x/yum
gpgcheck=1
gpgkey=https://artifacts.elastic.co/GPG-KEY-elasticsearch
enabled=1
autorefresh=1
type=rpm-md
EOF

# 安装 Filebeat
sudo yum install filebeat
```

### 2.2 在 Ubuntu 上安装 Filebeat

```bash
# 导入 Elastic 公钥
curl -fsSL https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elastic-archive-keyring.gpg

# 添加 Elastic 源
echo "deb [signed-by=/usr/share/keyrings/elastic-archive-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list

# 安装 Filebeat
sudo apt-get update
sudo apt-get install filebeat
```

### 2.3 启动和启用 Filebeat 服务

```bash
# 启动 Filebeat 服务
sudo systemctl start filebeat

# 设置 Filebeat 开机自动启动
sudo systemctl enable filebeat
```

---

## 3. Filebeat 配置详解

Filebeat 的配置文件是 `filebeat.yml`，位于 `/etc/filebeat/` 目录。Filebeat 的主要配置包括 **输入源（inputs）** 和 **输出目标（outputs）**。

### 3.1 输入源配置（inputs）

Filebeat 支持从多种类型的日志文件、系统日志、服务日志中收集日志数据。常见的输入源类型包括文件、系统日志等。

#### 示例：收集日志文件

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/*.log
    - /path/to/custom/logs/*.log
```

#### 示例：收集系统日志

```yaml
filebeat.inputs:
- type: syslog
  enabled: true
  protocol.udp:
    host: "0.0.0.0:514"
```

### 3.2 输出目标配置（outputs）

Filebeat 支持多种输出方式，可以将数据发送到 Elasticsearch、Logstash、Kafka 或其他系统。最常见的输出目标是 **Elasticsearch** 和 **Logstash**。

#### 直接输出到 Elasticsearch

```yaml
output.elasticsearch:
  hosts: ["http://localhost:9200"]
  username: "elastic"
  password: "your_password"
```

#### 输出到 Logstash

```yaml
output.logstash:
  hosts: ["localhost:5044"]
```

---

## 4. Filebeat 模块化配置

Filebeat 提供了许多预配置的模块，支持采集和解析不同类型的服务日志，如 Nginx、Apache、MySQL 等。

### 4.1 启用模块

使用以下命令可以启用一个 Filebeat 模块：

```bash
sudo filebeat modules enable nginx
```

### 4.2 查看所有可用模块

```bash
sudo filebeat modules list
```

### 4.3 模块配置文件

模块的配置文件位于 `/etc/filebeat/modules.d/` 目录下。例如，Nginx 模块的配置文件是 `nginx.yml`，你可以修改此文件来调整采集和解析的行为。

---

## 5. Filebeat 与 Logstash/Elasticsearch 集成

Filebeat 通常与 Elasticsearch 或 Logstash 一起使用，提供强大的日志处理能力。

### 5.1 通过 Logstash 处理日志

在 Logstash 中配置 `beats` 插件，监听来自 Filebeat 的日志：

```bash
input {
  beats {
    port => 5044
  }
}

output {
  elasticsearch {
    hosts => ["http://localhost:9200"]
    index => "filebeat-%{+YYYY.MM.dd}"
  }
}
```

### 5.2 配置 Filebeat 输出到 Logstash

在 `filebeat.yml` 中配置输出到 Logstash：

```yaml
output.logstash:
  hosts: ["localhost:5044"]
```

---

## 6. Filebeat 高级功能

### 6.1 多输入与多输出

你可以在 `filebeat.inputs` 中定义多个输入源，并且可以同时输出到多个目标（例如，输出到 Elasticsearch 和 Logstash）。

#### 多输入源示例

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/nginx/access.log

- type: log
  enabled: true
  paths:
    - /var/log/nginx/error.log
```

#### 多输出源示例

```yaml
output.elasticsearch:
  hosts: ["http://localhost:9200"]

output.logstash:
  hosts: ["localhost:5044"]
```

### 6.2 自定义字段与标签

你可以为每个输入源添加自定义字段，便于在 Elasticsearch 中进行筛选和查询。

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/nginx/access.log
  fields:
    environment: production
    app: nginx
```

---

## 7. Filebeat 性能优化

Filebeat 在处理大量日志时，需要对性能进行优化：

- **批量发送**：通过 `bulk_max_size` 设置批量发送日志的数量。
- **队列大小**：通过 `queue.mem` 设置内存队列大小，避免日志丢失。
- **数据压缩**：启用 `compression_level` 减少带宽使用。

```yaml
output.elasticsearch:
  hosts: ["http://localhost:9200"]
  bulk_max_size: 1024
  compression_level: 3
```

---

## 8. Filebeat 常见问题排查

### 8.1 查看 Filebeat 日志

查看 Filebeat 的运行日志以排查问题：

```bash
sudo journalctl -u filebeat
```

或查看具体的日志文件：

```bash
sudo tail -f /var/log/filebeat/filebeat.log
```

### 8.2 测试配置文件

使用命令检查 `filebeat.yml` 配置文件是否正确：

```bash
sudo filebeat test config
```

测试 Filebeat 是否能够连接到输出目标：

```bash
sudo filebeat test output
```

---

## 9. Filebeat 实战案例

### 9.1 集成 Filebeat 和 Elasticsearch，收集 Web 日志

步骤：
1. 配置 Filebeat 采集 Nginx 日志。
2. 启用 Nginx 模块。
3. 配置输出到 Elasticsearch。
4. 在 Kibana 中查看收集到的日志并创建仪表板。

### 9.2 Filebeat 和 Logstash 集成日志过滤

步骤：
1. Filebeat 将日志发送到 Logstash。
2. 在 Logstash 中使用 `grok` 过滤和解析日志。
3. 将处理后的日志发送到 Elasticsearch。

---

通过本文档的学习，你应该已经掌握了 Filebeat 的基本使用和配置，了解了如何与 Logstash 和 Elasticsearch 集成以及进行一些性能优化。在实际场景中，Filebeat 可以灵活应用于不同的日志采集需求，为你的系统和应用提供强大的日志分析能力。