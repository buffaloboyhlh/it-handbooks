# filebeat.yml 配置文件

`filebeat.yml` 是 Filebeat 的主配置文件，用于定义如何从日志源收集数据、处理数据并将其发送到指定的目标系统（如 Elasticsearch、Logstash 等）。以下是对 `filebeat.yml` 的详细解读，包括配置项的解释以及示例。

### Filebeat 配置结构
典型的 `filebeat.yml` 配置结构主要包含以下几个部分：
- **Inputs（输入）**：定义日志来源。
- **Outputs（输出）**：定义日志的输出目标。
- **Processors（处理器）**：日志的预处理配置。
- **Logging（日志记录）**：Filebeat 自身的日志记录设置。
- **Setup（初始化）**：设置 Filebeat 的索引模板、仪表板等。

### 1. Inputs（输入）
`filebeat.inputs` 配置项用于定义 Filebeat 从哪些源收集日志数据。

```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/mysql/mysql.log  # 指定日志文件路径
      - /var/log/mysql/error.log
    encoding: utf-8  # 设置日志文件的编码类型
    exclude_lines: ['^#']  # 排除匹配此正则的行（如以 # 开头的注释）
    include_lines: ['^ERR', '^WARN']  # 仅包含匹配此正则的行
    exclude_files: ['\.gz$']  # 排除匹配此正则的文件（如 .gz 压缩文件）
    scan_frequency: 10s  # 每隔 10 秒检查新日志
    multiline.pattern: '^ERROR'  # 处理多行日志，匹配行的开头为 ERROR
    multiline.negate: true  # 启用非匹配模式
    multiline.match: after  # 将多行日志合并为一条日志
```

#### 关键配置说明：
- **paths**：日志文件的路径，可以使用通配符，如 `/var/log/*.log`。
- **exclude_lines/include_lines**：通过正则表达式排除或包含特定日志行。
- **multiline**：用于处理多行日志，例如 Java 异常堆栈，或复杂的错误日志。
- **scan_frequency**：控制 Filebeat 扫描新日志的频率。

### 2. Outputs（输出）
Filebeat 支持多种输出目标，包括 Elasticsearch、Logstash、Kafka 等。你需要根据具体场景配置。

#### 输出到 Elasticsearch：
```yaml
output.elasticsearch:
  hosts: ["http://localhost:9200"]  # Elasticsearch 地址
  username: "elastic"  # Elasticsearch 用户名
  password: "changeme"  # Elasticsearch 密码
  index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"  # 索引的命名格式
  pipeline: "my-pipeline"  # 指定 Ingest pipeline 来处理日志
```

#### 输出到 Logstash：
```yaml
output.logstash:
  hosts: ["localhost:5044"]  # Logstash 地址
  ssl.enabled: true  # 启用 SSL 加密传输
  ssl.certificate_authorities: ["/etc/pki/tls/certs/logstash-ca.pem"]  # SSL 证书路径
```

#### 输出到 Kafka：
```yaml
output.kafka:
  hosts: ["kafka:9092"]
  topic: "filebeat"  # Kafka 主题
  partition.round_robin:  # 使用轮询的方式发送到分区
    reachable_only: true
  required_acks: 1  # Kafka 应答级别
```

#### 关键配置说明：
- **output.elasticsearch**：直接将日志发送到 Elasticsearch，常用于无需额外处理的场景。
- **output.logstash**：将日志发送到 Logstash 进行进一步处理，然后由 Logstash 发往 Elasticsearch 或其他系统。
- **output.kafka**：日志发送到 Kafka 队列，适用于分布式系统。

### 3. Processors（处理器）
`processors` 用于对日志进行预处理，处理器可以按顺序应用于每条日志事件。

```yaml
processors:
  - add_fields:
      target: ''  # 添加新字段到日志中
      fields:
        log_type: mysql  # 设置 log_type 字段为 "mysql"
  - drop_fields:
      fields: ["agent", "host"]  # 删除不需要的字段
  - rename:
      fields:
        - from: "message"
          to: "mysql_log"  # 重命名字段
  - dissect:
      tokenizer: "%{timestamp} %{loglevel} %{message}"  # 使用 dissect 解析字段
      field: "message"
      target_prefix: "parsed"  # 将解析的字段放入 "parsed" 下
```

#### 关键配置说明：
- **add_fields**：为每条日志添加自定义字段，便于后续查询或过滤。
- **drop_fields**：删除不需要的字段，减少数据量。
- **dissect**：通过简单的模式匹配来提取字段，适合结构化日志。

### 4. Logging（日志记录）
控制 Filebeat 自身的日志记录，帮助调试和排错。

```yaml
logging.level: info  # 日志级别，可以是 debug, info, warning, error
logging.to_files: true  # 是否写入文件
logging.files:
  path: /var/log/filebeat  # Filebeat 日志文件的存放路径
  name: filebeat.log  # 日志文件名
  keepfiles: 7  # 保留的日志文件数量
  permissions: 0644  # 文件权限
```

#### 关键配置说明：
- **logging.level**：日志级别，调试时建议使用 `debug` 级别，以便获取更多详细信息。
- **logging.files**：定义日志的保存路径和文件名，默认日志存储在标准输出。

### 5. Setup（初始化）
`setup` 部分用于初始化 Filebeat 在 Elasticsearch 中的索引模板、仪表板等。

```yaml
setup.kibana:
  host: "localhost:5601"  # Kibana 地址

setup.template.settings:
  index.number_of_shards: 1  # 设置索引的分片数量
  index.codec: best_compression  # 使用最佳压缩方式
setup.dashboards.enabled: true  # 是否自动加载 Filebeat 仪表板
```

#### 关键配置说明：
- **setup.kibana**：用于将 Filebeat 与 Kibana 集成。
- **setup.template**：用于设置 Elasticsearch 索引模板的参数。
- **setup.dashboards**：是否自动导入 Filebeat 提供的 Kibana 仪表板。

### 6. 示例配置
以下是一个完整的 `filebeat.yml` 示例，它从 MySQL 日志中收集日志，处理后发送到 Elasticsearch：

```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/mysql/mysql.log
      - /var/log/mysql/error.log
    exclude_lines: ['^#']  # 排除注释行
    multiline.pattern: '^ERROR'
    multiline.negate: true
    multiline.match: after

processors:
  - add_fields:
      target: ''
      fields:
        log_type: mysql

output.elasticsearch:
  hosts: ["http://localhost:9200"]
  username: "elastic"
  password: "changeme"
  index: "filebeat-mysql-%{+yyyy.MM.dd}"

setup.kibana:
  host: "localhost:5601"

setup.template.settings:
  index.number_of_shards: 1
```

### 总结
`filebeat.yml` 配置文件具有很强的灵活性，能够根据不同的日志源和处理需求进行定制。通过定义不同的输入、处理器和输出，可以满足多种日志采集场景。