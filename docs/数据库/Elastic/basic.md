# ElasticSearch 教程

### <a name="part1">一、Elasticsearch 简介与安装</a>

#### 1.1 什么是 Elasticsearch？
Elasticsearch 是一个分布式的开源搜索和分析引擎。它基于 Lucene 构建，提供了快速的全文检索功能，并能够处理大量的数据。Elasticsearch 主要用于日志分析、数据存储、实时数据搜索等场景。

#### 1.2 安装步骤（详细版）

##### 1.2.1 使用 Docker 安装 Elasticsearch
1. 运行 Elasticsearch 容器：
```bash
   docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.5.0
```

2. 检查 Elasticsearch 状态：
```bash
   curl http://localhost:9200/
```

##### 1.2.2 手动安装（适用于 Linux）
1. 下载并解压：
```bash
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.5.0-linux-x86_64.tar.gz
   tar -xzf elasticsearch-8.5.0-linux-x86_64.tar.gz
   cd elasticsearch-8.5.0
```

2. 配置 Elasticsearch（可选）：
   编辑 `config/elasticsearch.yml` 文件进行配置，例如设置集群名称、节点名称等。

3. 启动 Elasticsearch：
```bash
   ./bin/elasticsearch
```

4. 验证安装：
```bash
   curl http://localhost:9200/
```

---

### <a name="part2">二、Elasticsearch 核心概念</a>

#### 2.1 节点（Node）
每个节点可以扮演不同的角色，如主节点、数据节点等。节点的配置可以通过 `elasticsearch.yml` 文件中的设置来控制。

##### 主节点配置示例：
```yaml
node.master: true
node.data: true
```

##### 数据节点配置示例：
```yaml
node.master: false
node.data: true
```

#### 2.2 集群（Cluster）
集群的状态和健康状况可以通过 `_cluster/health` 接口获取：
```bash
GET /_cluster/health
```

响应示例：
```json
{
  "cluster_name": "my-cluster",
  "status": "green",
  "timed_out": false,
  "number_of_nodes": 3,
  "number_of_data_nodes": 2,
  "active_primary_shards": 10,
  "active_shards": 20,
  "relocating_shards": 0,
  "initializing_shards": 0,
  "unassigned_shards": 0,
  "delayed_unassigned_shards": 0,
  "number_of_pending_tasks": 0,
  "number_of_in_flight_fetch": 0,
  "task_max_waiting_in_queue_millis": 0,
  "active_shards_percent_as_number": 100.0
}
```

#### 2.3 索引（Index）
索引的设置和映射可以通过创建索引时定义，或者在索引创建后通过更新映射进行调整。

##### 更新索引设置示例：
```bash
PUT /my_index/_settings
{
  "index": {
    "number_of_replicas": 2
  }
}
```

#### 2.4 文档（Document）
文档在 Elasticsearch 中存储为 JSON 格式，支持多种数据类型，如文本、数字、日期等。

##### 插入文档示例：
```bash
POST /my_index/_doc/2
{
  "title": "Elasticsearch 高级特性",
  "content": "深入了解 Elasticsearch 的高级功能和性能优化。",
  "tags": ["高级", "优化"],
  "published_at": "2024-09-15"
}
```

#### 2.5 映射（Mapping）
映射定义了索引中文档字段的数据类型及其属性，如是否需要被索引、是否需要存储原始值等。

##### 修改映射示例：
```bash
PUT /my_index/_mapping
{
  "properties": {
    "description": {
      "type": "text",
      "analyzer": "ik_max_word"
    }
  }
}
```

---

### <a name="part3">三、Elasticsearch 的基本操作（CRUD）</a>

#### 3.1 创建索引
创建索引时可以设置分片和副本数，以控制索引的性能和容错性。

##### 创建索引示例：
```bash
PUT /my_blog
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "content": { "type": "text" },
      "author": { "type": "keyword" },
      "published_at": { "type": "date" }
    }
  }
}
```

#### 3.2 插入文档
文档插入时，Elasticsearch 会自动生成一个唯一的 `_id`，或者你可以自定义 `_id`。

##### 插入文档示例：
```bash
POST /my_blog/_doc/3
{
  "title": "深入 Elasticsearch",
  "content": "本篇文章详细讲解了 Elasticsearch 的高级功能和性能优化。",
  "author": "李四",
  "published_at": "2024-09-16"
}
```

#### 3.3 查询文档
Elasticsearch 提供多种查询方式，如 `GET`、`POST`、`search` API。支持精确查询、模糊查询、范围查询等。

##### 精确查询示例：
```bash
GET /my_blog/_search
{
  "query": {
    "term": {
      "author": "李四"
    }
  }
}
```

##### 模糊查询示例：
```bash
GET /my_blog/_search
{
  "query": {
    "fuzzy": {
      "title": "深入 Elasticsearch"
    }
  }
}
```

#### 3.4 更新文档
更新文档时，你可以使用 `doc` 或 `script` 来部分更新文档。

##### 部分更新示例：
```bash
POST /my_blog/_update/3
{
  "doc": {
    "title": "深入学习 Elasticsearch"
  }
}
```

#### 3.5 删除文档
删除文档时，指定文档的 `_id` 来执行删除操作。

##### 删除文档示例：
```bash
DELETE /my_blog/_doc/3
```

---

### <a name="part4">四、倒排索引与全文搜索</a>

#### 4.1 倒排索引
倒排索引在 Elasticsearch 中用于高效的全文搜索。它会将词语与文档的关系记录下来，以便快速检索。

#### 4.2 全文搜索示例
使用 `match` 查询进行全文搜索，Elasticsearch 会自动分词并进行相关性评分。

##### 基本全文搜索示例：
```bash
GET /my_blog/_search
{
  "query": {
    "match": {
      "content": "Elasticsearch 高级"
    }
  }
}
```

#### 4.3 精确匹配查询
`term` 查询用于精确匹配，不进行分析。

##### 精确匹配示例：
```bash
GET /my_blog/_search
{
  "query": {
    "term": {
      "author": "李四"
    }
  }
}
```

#### 4.4 复合查询（Boolean Query）
`bool` 查询允许组合多个查询条件，如 `must`、`should` 和 `must_not`。

##### 复合查询示例：
```bash
GET /my_blog/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "Elasticsearch" } },
        { "range": { "published_at": { "gte": "2024-01-01" } } }
      ],
      "must_not": [
        { "match": { "tags": "入门" } }
      ]
    }
  }
}
```

---

### <a name="part5">五、映射（Mapping）和分析（Analysis）</a>

#### 5.1 映射定义（详细版）
映射定义了字段的数据类型及其属性。映射可以在创建索引时定义，也可以在后续进行更新。

##### 自定义映射示例：
```bash
PUT /my_index
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard"
      },
      "tags": {
        "type": "keyword"
      },
      "published_at": {
        "type": "date",
        "format": "yyyy-MM-dd"
      }
    }
  }
}
```

#### 5.2 分词器与分析器
分词器将文本分割为词语，分析器则控制文本的索引和搜索方式。

##### 自定义分词器：
```bash
PUT /my_index
{
  "settings": {
    "analysis": {
      "analyzer": {
        "my_custom_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "asciifolding"]
        }
      }
    }
  }
}
```

##### 中文分词器（IK 分词器）：
```bash
PUT /my_index
{
  "settings": {
    "analysis": {
      "analyzer": {
        "ik_analyzer": {
          "type": "custom",
          "tokenizer": "ik_max_word"
        }
      }
    }
  }
}
```

##### 分词分析示例：
```bash
GET /_analyze
{
  "analyzer": "ik_analyzer",
  "text": "Elasticsearch 中文分词示例"
}
```

---

### <a name="part6">六、分布式特性与分片</a>

#### 6.1 分片（Shards）
分片是 Elasticsearch 中的存储和搜索单位。主分片负责数据的存储和操作，副本分片用于冗余和负载均衡。

##### 查询分片状态：
```bash
GET /_cat/shards
```

#### 6.2 副本（Replicas）
副本是主分片的备份，用于提高数据的可靠性和性能。

##### 配置副本数示例：
```bash
PUT /my_index/_settings
{
  "index": {
    "number_of_replicas": 2
  }
}
```

---

### <a name="part7">七、Elasticsearch 高级功能</a>

#### 7.1 聚合（Aggregations）
聚合用于对数据进行分析和总结。常见的聚合有 `terms`、`histogram`、`range` 等。

##### 按字段值聚合示例：
```bash
GET /my_blog/_search
{
  "aggs": {
    "tags_count": {
      "terms": {
        "field": "tags"
      }
    }
  }
}
```

##### 按日期范围聚合示例：
```bash
GET /my_blog/_search
{
  "aggs": {
    "date_ranges": {
      "date_range": {
        "field": "published_at",
        "ranges": [
          { "from": "2024-01-01", "to": "2024-06-30" },
          { "from": "2024-07-01", "to": "2024-12-31" }
        ]
      }
    }
  }
}
```

#### 7.2 脚本与动态聚合
Elasticsearch 允许使用脚本来进行动态聚合和计算。

##### 脚本聚合示例：
```bash
GET /my_blog/_search
{
  "aggs": {
    "average_length": {
      "avg": {
        "script": {
          "source": "doc['content.keyword'].value.length()"
        }
      }
    }
  }
}
```

---

### <a name="part8">八、性能优化与监控</a>

#### 8.1 性能优化

##### 分片配置
合理配置分片数量以优化性能。可以使用 Elasticsearch 的 `_cat` API 监控分片的状态。

##### 查询优化
- 使用 `filter` 代替 `query` 进行非评分查询。
- 使用 `doc_values` 提升聚合和排序性能。

##### 索引优化
- 定期执行 `force_merge` 操作以减少段文件的数量。
  ```bash
  POST /my_index/_forcemerge?max_num_segments=1
  ```

#### 8.2 监控工具

##### Kibana 监控
Kibana 提供了多种监控功能，如集群健康、节点状态、索引性能等。

##### Elasticsearch 集群监控
可以使用 Elasticsearch 自带的监控功能，如 `/_cluster/stats` 和 `/_nodes/stats` API 进行监控。

---

### <a name="part9">九、备份与恢复</a>

#### 9.1 快照与恢复

##### 创建快照
Elasticsearch 支持使用快照来备份数据。需要配置一个快照仓库。

1. 配置快照仓库（以文件系统为例）：
```bash
   PUT /_snapshot/my_backup
   {
     "type": "fs",
     "settings": {
       "location": "/mount/backups/my_backup"
     }
   }
```

2. 创建快照：
```bash
   PUT /_snapshot/my_backup/snapshot_1
   {
     "indices": "my_index",
     "ignore_unavailable": true,
     "include_global_state": false
   }
```

##### 恢复快照
恢复快照时，需要指定快照的名称。

```bash
POST /_snapshot/my_backup/snapshot_1/_restore
{
  "indices": "my_index"
}
```

---

### <a name="part10">十、Elasticsearch 与 Kibana、Logstash 的集成</a>

#### 10.1 Kibana
Kibana 提供了可视化和分析 Elasticsearch 数据的功能。它支持创建图表、仪表盘等。

##### 安装 Kibana
```bash
docker run -d --name kibana -p 5601:5601 --link elasticsearch kibana:8.5.0
```

##### 使用 Kibana
访问 `http://localhost:5601` 即可进入 Kibana 界面。你可以在 Kibana 中创建索引模式、仪表盘和可视化图表。

#### 10.2 Logstash
Logstash 是一个数据收集和处理工具，可以将数据从各种来源发送到 Elasticsearch。

##### Logstash 配置文件示例：
```bash
input {
  file {
    path => "/var/log/my_log.log"
    start_position => "beginning"
  }
}

filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "weblogs"
  }
}
```

##### 启动 Logstash
```bash
bin/logstash -f logstash.conf
```

---

