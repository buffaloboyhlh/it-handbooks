# logstash 配置文件

Logstash 配置文件是用来定义数据的**输入（Input）**、**过滤（Filter）**和**输出（Output）**的。它通常以 `.conf` 结尾，具有如下三个核心部分：

- **Input**：指定数据的来源。
- **Filter**：对数据进行处理和过滤。
- **Output**：指定数据的去向，如 Elasticsearch、文件或其他目标。

一个典型的 Logstash 配置文件结构如下：

```plaintext
input {
  # 定义输入
}

filter {
  # 定义过滤和处理
}

output {
  # 定义输出
}
```

接下来我们详细讲解这三部分的配置，并结合具体示例来说明如何配置 Logstash。

### 1. Input（输入）部分详解

`input` 定义了 Logstash 从哪些地方获取数据。常见的输入插件有：`beats`（接收来自 Filebeat 的数据）、`tcp`、`http`、`stdin` 等。

#### 示例 1：接收来自 Filebeat 的数据
```ruby
input {
  beats {
    port => 5044  # 接收来自 Filebeat 的数据，默认端口是 5044
  }
}
```

#### 示例 2：从 TCP 端口接收数据
```ruby
input {
  tcp {
    port => 5000  # 从 TCP 5000 端口接收数据
    codec => json_lines  # 数据格式为 JSON，每行一个对象
  }
}
```

#### 示例 3：从标准输入接收数据（用于测试）
```ruby
input {
  stdin { }
}
```

### 2. Filter（过滤器）部分详解

`filter` 部分是 Logstash 的核心，定义如何处理和转换数据。常见的过滤插件有 `grok`（提取模式匹配）、`date`（时间戳解析）、`mutate`（修改字段）等。

#### 示例 1：使用 `grok` 提取日志信息
```ruby
filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }  # 使用预定义的模式解析 Apache 日志
  }
}
```

#### 示例 2：使用 `mutate` 修改字段
```ruby
filter {
  mutate {
    rename => { "host" => "server_host" }  # 将字段 "host" 重命名为 "server_host"
    add_field => { "env" => "production" }  # 添加一个新字段 "env"
    remove_field => ["agent", "user"]  # 删除不需要的字段
  }
}
```

#### 示例 3：使用 `date` 解析时间戳
```ruby
filter {
  date {
    match => [ "timestamp", "ISO8601" ]  # 将 timestamp 字段解析为 Logstash 的 @timestamp 字段
    target => "@timestamp"
  }
}
```

#### 示例 4：处理多行日志
```ruby
filter {
  multiline {
    pattern => "^ERROR"  # 匹配 ERROR 开头的日志
    negate => true
    what => "previous"  # 将当前行附加到前一行日志
  }
}
```

### 3. Output（输出）部分详解

`output` 部分定义了数据的去向，可以输出到 Elasticsearch、文件、Kafka 等。

#### 示例 1：输出到 Elasticsearch
```ruby
output {
  elasticsearch {
    hosts => ["localhost:9200"]  # Elasticsearch 地址
    index => "logstash-%{+YYYY.MM.dd}"  # 指定索引格式
  }
}
```

#### 示例 2：输出到文件
```ruby
output {
  file {
    path => "/var/log/logstash/logstash_output.log"  # 将处理后的日志写入文件
  }
}
```

#### 示例 3：输出到标准输出（用于调试）
```ruby
output {
  stdout {
    codec => rubydebug  # 以调试模式输出日志
  }
}
```

#### 示例 4：输出到 Kafka
```ruby
output {
  kafka {
    bootstrap_servers => "localhost:9092"  # Kafka 服务器地址
    topic_id => "logstash_logs"  # Kafka 主题名称
  }
}
```

### 4. Logstash 配置文件完整示例

假设我们要从 Filebeat 收集 Apache 日志，经过过滤后将其发送到 Elasticsearch，并且需要过滤掉不必要的字段并解析日志时间戳。以下是一个完整的配置示例：

```ruby
input {
  beats {
    port => 5044  # 接收来自 Filebeat 的数据
  }
}

filter {
  # 解析 Apache 日志格式
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
  
  # 将日志中的时间字段转换为 @timestamp
  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    target => "@timestamp"
  }
  
  # 删除不必要的字段
  mutate {
    remove_field => [ "agent", "auth" ]
  }
}

output {
  # 输出到 Elasticsearch
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "apache-logs-%{+YYYY.MM.dd}"
  }

  # 同时将日志输出到标准输出，用于调试
  stdout {
    codec => rubydebug
  }
}
```

### 配置文件详解：
- **input**: 配置 Logstash 从 Filebeat 接收日志，监听端口 5044。
- **filter**: 
  - 使用 `grok` 解析 Apache 日志格式。
  - 使用 `date` 解析日志中的时间戳。
  - 使用 `mutate` 删除不需要的字段（如 `agent` 和 `auth`）。
- **output**: 
  - 将处理后的日志数据输出到 Elasticsearch，创建名为 `apache-logs-YYYY.MM.dd` 的索引。
  - 同时将日志输出到标准输出，方便调试。

### 5. 多输入、多输出示例

Logstash 支持多个输入、过滤器和输出插件，可以组合使用。

```ruby
input {
  beats {
    port => 5044  # 接收来自 Filebeat 的数据
  }
  tcp {
    port => 5000  # 从 TCP 5000 端口接收数据
    codec => json_lines
  }
}

filter {
  # 针对 Filebeat 数据的过滤器
  if [type] == "filebeat" {
    grok {
      match => { "message" => "%{COMBINEDAPACHELOG}" }
    }
  }

  # 针对 TCP 数据的过滤器
  if [type] == "tcp" {
    json {
      source => "message"
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "logstash-%{+YYYY.MM.dd}"
  }
}
```

### 总结
Logstash 的配置文件非常灵活，可以根据需要自定义输入、过滤和输出。通过不同的输入插件，可以从不同来源接收数据；通过过滤插件可以对数据进行丰富的处理；输出插件可以将处理后的数据发送到多种目标，如 Elasticsearch、文件、Kafka 等。