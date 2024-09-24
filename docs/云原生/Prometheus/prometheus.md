# Prometheus 配置文件

**`prometheus.yml`** 是 Prometheus 的核心配置文件，它定义了 Prometheus 的全局配置、抓取监控目标、告警规则和告警接收器等。下面是 `prometheus.yml` 配置文件的详细讲解，包括各个部分的功能和示例配置。

---

## 一、基本结构

一个典型的 `prometheus.yml` 配置文件包括以下几个部分：

1. **Global Config**（全局配置）：定义全局抓取间隔、评估规则等。
2. **Scrape Configs**（抓取配置）：定义 Prometheus 如何抓取监控目标的数据。
3. **Rule Files**（告警规则文件）：指定告警规则所在的文件路径。
4. **Alerting Config**（告警配置）：定义告警管理器的接收地址。
5. **Remote Write & Remote Read**（远程存储配置）：定义如何将数据写入或从远程存储读取数据。

---

## 二、配置项详解

### 1. 全局配置（Global Config）

全局配置定义了 Prometheus 如何以默认方式执行抓取和评估。通常包括抓取间隔和告警规则评估间隔。

```yaml
global:
  scrape_interval: 15s  # 抓取间隔，默认为 1m
  evaluation_interval: 15s  # 告警规则评估间隔，默认为 1m
  scrape_timeout: 10s  # 单个抓取任务的超时时间
  external_labels:
    monitor: 'codelab-monitor'  # 当多个 Prometheus 服务器抓取相同数据时，用于标识数据来源
```

- **scrape_interval**：定义 Prometheus 抓取所有目标的默认时间间隔。
- **evaluation_interval**：定义 Prometheus 评估告警规则的默认时间间隔。
- **scrape_timeout**：设置抓取任务的超时时间。
- **external_labels**：在向外部存储发送数据时添加的标签，通常用于区分多个 Prometheus 实例。

---

### 2. 抓取配置（Scrape Configs）

`scrape_configs` 定义了 Prometheus 应抓取的目标（如节点、服务、应用程序）及其抓取方法。可以使用静态目标或动态服务发现方式。

#### 基本结构：
```yaml
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

#### 参数解释：
- **job_name**：用于标识抓取任务的名称，每个任务可以包含多个目标。
- **static_configs**：静态定义抓取目标。
- **targets**：要抓取的目标地址及端口。

#### 示例：
```yaml
scrape_configs:
  - job_name: 'node_exporter'  # 任务名称
    scrape_interval: 10s  # 任务特定的抓取间隔
    static_configs:
      - targets: ['localhost:9100']  # 要监控的目标地址

  - job_name: 'my-app'  # 监控某个自定义应用
    metrics_path: '/metrics'  # 监控目标的指标路径，默认为 /metrics
    static_configs:
      - targets: ['192.168.1.10:8080', '192.168.1.11:8080']
```

---

### 3. 规则文件（Rule Files）

`rule_files` 用于定义告警规则和记录规则。可以将告警规则存放在一个或多个外部文件中。

#### 语法：
```yaml
rule_files:
  - "rules.yml"  # 定义告警规则的文件
```

例如：
```yaml
groups:
  - name: example
    rules:
    - alert: HighMemoryUsage
      expr: node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.2
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Instance {{ $labels.instance }} has high memory usage"
        description: "Memory usage on {{ $labels.instance }} is above 80% for the last 5 minutes."
```

- **alert**：告警名称。
- **expr**：告警条件，使用 PromQL 表达式。
- **for**：条件持续时间。
- **labels**：告警的标签信息，用于分类告警。
- **annotations**：告警注释，提供额外信息。

---

### 4. 告警配置（Alerting Config）

`alerting` 配置用于定义 Prometheus 如何与 `Alertmanager` 进行通信，以便发送告警通知。

#### 配置示例：
```yaml
alerting:
  alertmanagers:
  - static_configs:
      - targets: ['localhost:9093']  # Alertmanager 服务器地址
```

这里，Prometheus 会将告警事件发送到位于 `localhost:9093` 的 Alertmanager 实例。

---

### 5. 远程写入和读取（Remote Write & Remote Read）

Prometheus 可以将数据写入远程存储，或从远程存储读取历史数据。

#### Remote Write（远程写入）：
```yaml
remote_write:
  - url: "http://remote-storage.local/api/v1/write"
```

#### Remote Read（远程读取）：
```yaml
remote_read:
  - url: "http://remote-storage.local/api/v1/read"
```

---

## 三、动态服务发现

除了静态配置外，Prometheus 还支持通过服务发现动态地抓取监控目标。这些服务发现机制包括 Kubernetes、Consul、EC2 等。

### Kubernetes 服务发现：
当 Prometheus 在 Kubernetes 中运行时，可以通过 `kubernetes_sd_configs` 动态发现 Kubernetes Pods、服务或节点。

```yaml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: $1
```

- **kubernetes_sd_configs**：指定通过 Kubernetes API 进行服务发现。
- **role**：定义服务发现的角色，如 `pod`（发现 Pod）、`service`（发现服务）。
- **relabel_configs**：用于过滤或修改抓取的目标，可以根据标签或元数据动态调整。

---

## 四、完整配置示例

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'prometheus-monitor'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: $1

rule_files:
  - 'rules.yml'

alerting:
  alertmanagers:
  - static_configs:
      - targets: ['localhost:9093']

remote_write:
  - url: 'http://remote-storage.local/api/v1/write'

remote_read:
  - url: 'http://remote-storage.local/api/v1/read'
```

---

## 五、总结

`prometheus.yml` 是 Prometheus 最重要的配置文件，它定义了 Prometheus 如何与各个监控目标进行交互、如何抓取数据、如何配置告警等。掌握它的结构和配置方式能够让你灵活地监控分布式系统和应用。