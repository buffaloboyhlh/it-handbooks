# Prometheus 告警

Prometheus 的告警处理系统包括两个核心部分：**告警规则 (Alerting Rules)** 和 **Alertmanager**。Prometheus 通过告警规则触发告警，将其发送到 Alertmanager，Alertmanager 则负责处理告警通知的发送、分组、去重、抑制等操作。

下面将通过一个完整的告警配置示例，详细讲解 Prometheus 告警处理的流程。

### 1. Prometheus 告警规则配置

Prometheus 告警规则用于定义触发告警的条件，告警规则会基于 Prometheus 的时间序列数据定期进行评估。

#### 1.1 告警规则示例

假设我们有以下几个需求：
- 当某个实例宕机超过 5 分钟时触发告警。
- 当某实例的 CPU 使用率超过 80% 且持续 2 分钟时触发告警。
- 当某实例的内存使用率超过 90% 且持续 5 分钟时触发告警。

告警规则可以配置在 Prometheus 的规则文件中，例如 `alert.rules.yml`：

```yaml
groups:
  - name: example_alerts  # 告警组名称
    rules:
      - alert: InstanceDown  # 告警名称
        expr: up == 0  # 告警触发条件
        for: 5m  # 告警触发前需持续 5 分钟
        labels:
          severity: critical  # 告警标签，表示严重性
        annotations:
          summary: "Instance {{ $labels.instance }} down"  # 简要描述
          description: "Instance {{ $labels.instance }} has been down for more than 5 minutes."  # 详细描述

      - alert: HighCpuUsage  # 告警名称
        expr: sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance) > 0.8  # 告警条件
        for: 2m  # 条件需满足 2 分钟
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage on instance {{ $labels.instance }} has been above 80% for more than 2 minutes."

      - alert: HighMemoryUsage
        expr: node_memory_Active_bytes / node_memory_MemTotal_bytes > 0.9  # 内存使用率超过 90%
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage on instance {{ $labels.instance }} has been above 90% for more than 5 minutes."
```

#### 1.2 告警规则字段解释
- **alert**：告警的名称。
- **expr**：使用 PromQL 表达式定义告警条件。
- **for**：条件需要持续多长时间才会触发告警。
- **labels**：用于对告警打标签（如 `severity` 表示告警严重性）。
- **annotations**：附加的告警描述信息，通常用于生成告警通知。

#### 1.3 引用告警规则

需要在 Prometheus 的主配置文件 `prometheus.yml` 中引用告警规则文件：

```yaml
rule_files:
  - 'alert.rules.yml'
```

### 2. Alertmanager 配置

Alertmanager 负责接收 Prometheus 发出的告警，并根据预设规则处理告警通知。你可以为不同的告警定义不同的通知方式和接收者。

#### 2.1 Alertmanager 配置示例

以下是一个基本的 `alertmanager.yml` 配置文件示例，包含邮件和 Slack 两种通知渠道。

```yaml
global:
  resolve_timeout: 5m  # 告警解决的超时时间

route:
  receiver: 'default'  # 默认接收器
  group_by: ['alertname', 'instance']  # 告警分组依据
  group_wait: 30s  # 第一次通知前的等待时间
  group_interval: 5m  # 组内告警发送的最小间隔时间
  repeat_interval: 1h  # 相同告警重复发送的最小间隔时间

  routes:
    - match:
        severity: critical  # 匹配告警严重性为 critical
      receiver: 'critical-team'  # 将严重告警发送到特定接收者
    - match:
        severity: warning  # 匹配告警严重性为 warning
      receiver: 'warning-team'  # 将 warning 告警发送到另一个接收者

receivers:
  - name: 'default'
    email_configs:
      - to: 'admin@example.com'  # 默认告警接收者邮箱
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alertmanager@example.com'
        auth_password: 'password'

  - name: 'critical-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/T000/B000/XXXX'  # Slack Webhook URL
        channel: '#critical-alerts'  # Slack 频道

  - name: 'warning-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/T000/B000/YYYY'
        channel: '#warning-alerts'
```

#### 2.2 关键字段解释
- **global**：全局配置，包含告警的超时等设置。
- **route**：告警路由规则，决定如何发送告警及其分组方式。
- **routes**：定义子路由，根据告警的标签匹配不同的接收器。例如，`severity` 为 `critical` 的告警可以发送到指定团队。
- **receivers**：定义接收告警的目标，可以配置多个接收器，例如电子邮件、Slack 等。

#### 2.3 配置多个接收渠道

除了邮件，Alertmanager 还支持其他渠道，比如 Slack、PagerDuty、Webhook 等。你可以在 `receivers` 中根据需要添加多种接收方式。

### 3. 部署和运行

#### 3.1 启动 Prometheus

确保 Prometheus 的配置文件中引用了告警规则文件，并启动 Prometheus：

```bash
./prometheus --config.file=prometheus.yml
```

#### 3.2 启动 Alertmanager

确保 `alertmanager.yml` 配置正确，并启动 Alertmanager：

```bash
./alertmanager --config.file=alertmanager.yml
```

### 4. 高级功能

#### 4.1 告警抑制（Silencing）

告警抑制用于在特定时间段内（例如维护期间）静默告警，避免收到不必要的告警通知。你可以在 Alertmanager 的 UI 中手动添加静默规则，也可以通过 API 动态创建。

通过 API 创建静默规则的示例：

```bash
curl -XPOST "http://alertmanager/api/v1/silences" -d '{
  "matchers": [{"name": "instance", "value": "web-server-01"}],
  "startsAt": "2024-09-24T10:00:00Z",
  "endsAt": "2024-09-24T12:00:00Z",
  "createdBy": "admin",
  "comment": "Scheduled maintenance"
}'
```

#### 4.2 告警分组与去重

Alertmanager 支持告警的分组与去重。在 `route` 中的 `group_by` 字段定义告警的分组依据。例如，按 `alertname` 和 `instance` 进行分组时，同类告警将合并为一个通知，以减少通知次数。去重功能确保相同的告警不会被多次通知。

### 5. 完整告警处理流程

1. **Prometheus 触发告警**：Prometheus 根据定义的告警规则定期评估指标数据。当告警条件满足时，Prometheus 会生成告警，并将告警信息发送给 Alertmanager。
   
2. **Alertmanager 处理告警**：
   - **去重**：Alertmanager 对重复的告警进行去重。
   - **分组**：根据配置，将相似的告警进行分组。
   - **抑制**：根据静默规则，抑制不必要的告警。
   - **发送通知**：根据路由规则，Alertmanager 将告警发送到预定义的接收者（如邮件、Slack 等）。

3. **告警恢复**：当问题解决时，Prometheus 会自动更新告警状态，Alertmanager 将告警恢复信息发送给相关接收者。

### 6. 实践注意事项
- **告警规则设计**：设计告警规则时，尽量避免告警风暴。应适当使用 `for` 确保短暂的指标波动不会触发告警。
- **静默策略**：在进行计划内维护或大规模部署时，可以提前设置静默规则，避免收到大量无用的告警。
- **多渠道告警**：对于重要的告警，建议配置多渠道通知（如邮件、短信、即时通讯工具），以确保告警信息能够及时被处理。

### 结论

通过配置 Prometheus 的告警规则和 Alertmanager，你可以构建一个强大的告警系统，帮助及时监控服务状态并处理各种异常。