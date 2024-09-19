# LogStash 教程

**Logstash** 是 Elastic Stack 中的一个核心组件，用于实时的数据收集、转换和存储。Logstash 可以从各种来源（如日志文件、数据库、消息队列等）收集数据，对数据进行过滤、转换，然后将数据发送到多个目的地（如 Elasticsearch、Kafka、文件等）。Logstash 的强大之处在于它的灵活性，支持复杂的数据处理流程和丰富的插件系统。

本文将详细介绍 Logstash 的功能、架构、使用方法以及常见配置示例。

---

## 1. **Logstash 架构概览**

Logstash 的工作流程可以分为三个阶段：

- **Input**：定义数据源，Logstash 从这些数据源读取数据。
- **Filter**：对数据进行处理、过滤或转换。
- **Output**：将处理后的数据发送到目的地，如 Elasticsearch、文件、数据库等。

Logstash 还支持 `codec` 插件，用于处理数据的序列化与反序列化。它允许在数据进入和输出时，对数据进行格式转换，例如 JSON、CSV 等格式。

---

## 2. **Logstash 安装**

Logstash 可以在各种操作系统上安装，以下介绍在 Linux 和 Windows 平台上的安装方法。

### 2.1 **在 Linux 上安装**

以 Ubuntu 为例：

1. 添加 Elastic APT 仓库：

```bash
   curl -L -O https://artifacts.elastic.co/downloads/logstash/logstash-8.10.0.deb
   sudo dpkg -i logstash-8.10.0.deb
```

2. 安装 Logstash：

```bash
   sudo apt-get update
   sudo apt-get install logstash
```

3. 启动 Logstash 服务：

```bash
   sudo systemctl start logstash
   sudo systemctl enable logstash
```

### 2.2 **在 Windows 上安装**

1. 下载 Logstash 安装包：[Logstash 下载页面](https://www.elastic.co/downloads/logstash)
2. 解压文件，并进入 Logstash 目录。
3. 使用命令行启动 Logstash：

```bash
   bin\logstash.bat -f logstash.conf
```

---

## 3. **Logstash 配置详解**

Logstash 的配置文件是基于管道（Pipeline）的。一个典型的 Logstash 配置文件包含三个部分：`input`（输入）、`filter`（过滤）和 `output`（输出）。

### 3.1 **基本配置结构**

Logstash 的配置文件是纯文本文件，通常以 `.conf` 为后缀，每个配置文件可以定义多个 `input`、`filter` 和 `output` 块。下面是一个简单的 Logstash 配置示例：

```yaml
input {
  beats {
    port => 5044  # 接收来自 Filebeat 的日志
  }
}

filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "logs-%{+YYYY.MM.dd}"
  }
  stdout {
    codec => rubydebug  # 在终端打印处理后的数据，用于调试
  }
}
```

### 3.2 **Input 插件**

Logstash 支持多种数据源的输入，通过 `input` 插件定义数据采集方式。

常用的 `input` 插件包括：

- **file**：从文件中读取日志。
- **beats**：接收来自 Beats（如 Filebeat）的数据。
- **stdin**：从标准输入（控制台）读取数据。
- **syslog**：从 syslog 接收日志。

示例：

```yaml
input {
  file {
    path => "/var/log/syslog"
    start_position => "beginning"  # 从文件开头开始读取
  }
}
```

### 3.3 **Filter 插件**

`filter` 插件用于对数据进行处理和转换。Logstash 提供了丰富的 `filter` 插件，常见的有：

- **grok**：用于解析和提取日志中的结构化数据。
- **mutate**：修改字段（如添加、重命名、移除字段）。
- **date**：将时间戳转换为统一格式。
- **geoip**：根据 IP 地址查找地理位置信息。
- **kv**：将键值对解析为结构化字段。

#### 3.3.1 **Grok 过滤器**

`grok` 插件是 Logstash 最常用的过滤器之一，用于解析非结构化数据。它基于正则表达式，将日志信息解析成结构化字段。

例如，解析 Nginx 访问日志：

```yaml
filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
}
```

Logstash 提供了许多预定义的 `grok` 模式（如 `%{COMBINEDAPACHELOG}`），你可以根据需要自定义。

#### 3.3.2 **Mutate 过滤器**

`mutate` 插件用于修改字段，例如重命名、删除、转换类型等。

示例：将字段 `status_code` 从字符串转换为整数：

```yaml
filter {
  mutate {
    convert => { "status_code" => "integer" }
  }
}
```

#### 3.3.3 **Date 过滤器**

`date` 插件将日志中的时间戳转换为 Logstash 内部使用的标准时间格式。

示例：

```yaml
filter {
  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
  }
}
```

### 3.4 **Output 插件**

`output` 插件用于定义日志的输出目的地，Logstash 支持多种目标，包括 Elasticsearch、文件、Kafka、数据库等。

常见的 `output` 插件有：

- **elasticsearch**：将日志发送到 Elasticsearch。
- **file**：将日志写入文件。
- **kafka**：将日志发送到 Kafka。
- **stdout**：将日志输出到标准输出（控制台）。

#### 3.4.1 **输出到 Elasticsearch**

最常见的输出方式是将日志发送到 Elasticsearch，以便后续通过 Kibana 进行可视化分析。

```yaml
output {
  elasticsearch {
    hosts => ["http://localhost:9200"]
    index => "logs-%{+YYYY.MM.dd}"  # 按日期生成索引
  }
}
```

#### 3.4.2 **输出到控制台（stdout）**

在开发和调试时，可以将日志输出到控制台，便于观察处理效果：

```yaml
output {
  stdout {
    codec => rubydebug  # 以可读格式输出日志
  }
}
```

---

## 4. **Logstash 实例：处理 Apache 日志**

下面是一个处理 Apache 日志的完整 Logstash 配置示例，演示了如何从文件读取日志、解析日志内容并将其存储到 Elasticsearch。

```yaml
input {
  file {
    path => "/var/log/apache2/access.log"
    start_position => "beginning"
  }
}

filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    target => "@timestamp"
  }
  geoip {
    source => "clientip"  # 根据 IP 查找地理位置信息
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "apache-logs-%{+YYYY.MM.dd}"
  }
  stdout {
    codec => rubydebug
  }
}
```

这个配置文件的工作流程是：

1. **输入**：从 `/var/log/apache2/access.log` 文件中读取日志。
2. **过滤**：使用 `grok` 解析日志，提取字段如 `clientip`、`request`、`response` 等；使用 `date` 插件解析时间戳；使用 `geoip` 插件根据 `clientip` 获取地理位置。
3. **输出**：将日志发送到 Elasticsearch，并按日期创建索引，同时将日志输出到控制台。

---

## 5. **Logstash 管道管理与性能优化**

在生产环境中，Logstash 处理大量的数据，可能需要进行性能调优和多管道管理。

### 5.1 **多管道配置**

Logstash 支持多管道，可以同时处理多个不同类型的日志。你可以在 Logstash 配置目录中创建多个配置文件，每个文件对应不同的管道。

在 `logstash.yml` 中启用多管道配置：

```yaml
path.config: "/etc/logstash/conf.d/*.conf"
```

将每个日志源的处理逻辑拆分成多个配置文件，例如 `nginx.conf`、`mysql.conf`。

### 5.2 **性能优化**

- **Pipeline Workers**：调整 Logstash 管道的工作线程数以提高并发处理能力，默认情况下，Logstash 的 `pipeline.workers` 设置为 CPU 核心数。

  ```yaml
  pipeline.workers: 4
  ```

- **批处理大小**：调整 `pipeline.batch.size` 以提高数据处理效率，默认是 125。

  ```yaml
  pipeline.batch.size: 200
  ```

---

## 6. **Logstash 可视化与监控**

通过 Elastic Stack 的 Kibana，你可以监控 Logstash 的运行状态，查看数据处理的延迟、错误等。

1. 在 Kibana 中导航到 **Stack Monitoring**。
2. 启用监控后，你可以实时查看 Logstash 的性能指标，包括吞吐量、延迟、CPU 使用率等。

---

## Filter 详解

`Logstash` 的 `filter` 是数据处理管道中的中间步骤，用于转换、增强、解析或过滤传入的数据。`filter` 插件可以帮助你对数据执行复杂的操作，常见的过滤插件包括 `grok`、`date`、`mutate`、`geoip` 等。

### Logstash Filter 插件类别及常用插件
#### 1. **grok** 插件
`grok` 是 Logstash 中最常用的插件之一，用于从未结构化的文本（如日志文件）中提取结构化数据。它基于正则表达式来匹配并解析文本。
```ruby
filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
}
```
**示例解释：**
- `match`：定义了 `grok` 匹配模式，其中 `"message"` 是日志字段，`%{COMBINEDAPACHELOG}` 是 `grok` 模式，它代表 Apache 日志格式。

你也可以自定义 `grok` 模式，例如：
```ruby
filter {
  grok {
    match => { "message" => "%{IP:client_ip} %{WORD:method} %{URIPATHPARAM:request} %{NUMBER:status_code}" }
  }
}
```
这里定义了提取 `IP`、`HTTP 方法`、`请求路径` 和 `状态码`。

#### 2. **mutate** 插件
`mutate` 插件用于修改事件中的字段，可以进行增删改操作，包括重命名字段、转换数据类型、拆分或合并字符串等操作。
```ruby
filter {
  mutate {
    rename => { "host" => "server_host" }
    convert => { "status_code" => "integer" }
    remove_field => [ "unnecessary_field" ]
  }
}
```
**常用操作：**
- `rename`：重命名字段。
- `convert`：将字段转换为不同的数据类型（如 `integer`, `float`, `string`）。
- `remove_field`：删除不必要的字段。

#### 3. **date** 插件
`date` 插件用于解析日期字段，并将它们转换为 `Logstash` 的 `@timestamp` 字段。这样可以确保日志时间被正确解析并与其他系统时间保持一致。
```ruby
filter {
  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    target => "@timestamp"
  }
}
```
**示例解释：**
- `match`：用于匹配输入日志的日期格式。`"timestamp"` 是日志字段，后面的 `"dd/MMM/yyyy:HH:mm:ss Z"` 是日期格式模板。
- `target`：指定输出的目标字段，默认是 `@timestamp`。

#### 4. **geoip** 插件
`geoip` 插件用于将 IP 地址解析为地理位置信息（如国家、城市、纬度、经度等）。
```ruby
filter {
  geoip {
    source => "client_ip"
  }
}
```
**示例解释：**
- `source`：要解析的 IP 地址字段。插件会根据该字段自动生成地理位置信息（如 `geoip.location`）。

#### 5. **json** 插件
`json` 插件用于将 JSON 字符串解析为结构化数据（字段）。
```ruby
filter {
  json {
    source => "message"
  }
}
```
**示例解释：**
- `source`：指定要解析的 JSON 字符串字段，`Logstash` 会自动解析该字段并生成相应的结构化字段。

#### 6. **kv** 插件
`kv` 插件用于从键值对字符串中提取字段。它非常适用于从 URL 查询参数或其他带有键值对的日志格式中提取信息。
```ruby
filter {
  kv {
    source => "message"
    field_split => "&"
    value_split => "="
  }
}
```
**示例解释：**
- `source`：指定要解析的字段。
- `field_split`：指定字段之间的分隔符（如 `&`）。
- `value_split`：指定键值之间的分隔符（如 `=`）。

#### 7. **split** 插件
`split` 插件用于将一个字段拆分为多个事件。这在你有数组或多个记录嵌套在同一个事件中的时候非常有用。
```ruby
filter {
  split {
    field => "items"
  }
}
```
**示例解释：**
- `field`：指定要拆分的数组字段。

#### 8. **drop** 插件
`drop` 插件用于丢弃不需要的事件，通常用于过滤掉不需要的日志。
```ruby
filter {
  if [status_code] == 200 {
    drop { }
  }
}
```
**示例解释：**
- 这里指定当 `status_code` 为 `200` 时，丢弃该事件。

### Logstash Filter 的基本结构
```ruby
filter {
  filter_plugin {
    option1 => value1
    option2 => value2
  }
}
```
其中 `filter_plugin` 是你选择的过滤插件（如 `grok`, `mutate` 等），`option1` 和 `value1` 是该插件的配置选项和值。

### 综合示例
```ruby
filter {
  grok {
    match => { "message" => "%{IPORHOST:client_ip} %{USER:ident} %{USER:auth} \[%{HTTPDATE:timestamp}\] \"%{WORD:method} %{URIPATHPARAM:request} HTTP/%{NUMBER:http_version}\" %{NUMBER:status_code} %{NUMBER:bytes}" }
  }

  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    target => "@timestamp"
  }

  geoip {
    source => "client_ip"
  }

  mutate {
    convert => { "status_code" => "integer" }
    convert => { "bytes" => "integer" }
    remove_field => [ "ident", "auth" ]
  }
}
```
**示例解释：**
1. 使用 `grok` 从日志中提取客户端 IP、HTTP 方法、请求路径、状态码等字段。
2. 使用 `date` 插件将 `timestamp` 字段转换为 `@timestamp`。
3. 使用 `geoip` 插件根据 IP 提取地理位置信息。
4. 使用 `mutate` 插件转换 `status_code` 和 `bytes` 字段为整数，并删除不需要的字段。

### 总结
`Logstash` 的 `filter` 插件为日志处理提供了强大的能力，可以通过 `grok` 提取文本，使用 `mutate` 修改字段，使用 `geoip` 获取地理信息等。根据不同需求，选择合适的插件进行配置，将有效提升日志处理的效率和灵活性。