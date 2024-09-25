# PromQL 

PromQL（Prometheus Query Language）是 Prometheus 的查询语言，用于从 Prometheus 时序数据库中查询和分析监控数据。以下是 PromQL 的详细教程：

### 1. **基本概念**
Prometheus 中存储的是时间序列数据，数据是以 metric 名称、label 和时间戳为主的键值对。PromQL 用于查询这些时间序列数据，并可以通过不同的函数进行聚合、处理和计算。

- **时间序列**：由一个 `metric`（度量名称）和多个 `label`（标签）唯一标识。例如：`http_requests_total{job="api-server", method="GET"}`。
- **度量类型**：主要有四种：
  - **Counter**：累加器，只增不减，适用于累计的计数器（如请求数）。
  - **Gauge**：可增可减的数值，适用于瞬时值（如温度、CPU 使用率）。
  - **Histogram**：直方图，用于记录一组数据的分布情况（如请求延时的分布）。
  - **Summary**：与 Histogram 类似，但它可以计算出百分位数。

### 2. **PromQL 基础语法**
PromQL 的基本组成有四种类型的表达式：**瞬时向量**、**区间向量**、**标量**、**字符串**。

- **瞬时向量 (Instant Vector)**：返回在指定时间点上的一组时间序列。例如：`up` 查询当前所有 `up` 度量。
  
- **区间向量 (Range Vector)**：返回在指定时间范围内的时间序列。例如：`up[5m]` 表示过去 5 分钟内的 `up` 数据。
  
- **标量 (Scalar)**：返回单个数值，例如：`5`。
  
- **字符串 (String)**：返回文本字符串值，但较少使用。

### 3. **查询示例**
#### 3.1 查询某个指标
```promql
http_requests_total
```
返回 `http_requests_total` 度量的所有时间序列。

#### 3.2 过滤标签
```promql
http_requests_total{method="GET", status="200"}
```
返回 `method` 为 `GET` 且 `status` 为 `200` 的时间序列。

#### 3.3 使用区间查询
```promql
rate(http_requests_total[5m])
```
计算过去 5 分钟内 HTTP 请求数的变化速率。

### 4. **PromQL 操作符**

PromQL 支持以下几种操作符：

- **算术运算符**：`+`, `-`, `*`, `/`, `%`（用于度量之间的计算）。
  ```promql
  rate(http_requests_total[5m]) / rate(cpu_usage_seconds_total[5m])
  ```

- **比较运算符**：`==`, `!=`, `>`, `<`, `>=`, `<=`（可用于过滤符合条件的时间序列）。
  ```promql
  cpu_usage_seconds_total > 0.5
  ```

- **逻辑运算符**：`and`, `or`, `unless`（用于两个度量之间的逻辑组合）。
  ```promql
  http_requests_total{status="200"} and cpu_usage_seconds_total
  ```

### 5. **PromQL 聚合函数**
PromQL 支持多种聚合函数来处理查询结果。常见的聚合函数有：

- **sum()**：对多个时间序列的值进行求和。
  ```promql
  sum(rate(http_requests_total[5m]))
  ```

- **avg()**：对多个时间序列的值进行求平均。
  ```promql
  avg(cpu_usage_seconds_total)
  ```

- **max()** 和 **min()**：分别返回最大和最小值。
  ```promql
  max(http_requests_total)
  ```

- **count()**：统计匹配的时间序列数量。
  ```promql
  count(http_requests_total)
  ```

### 6. **常见 PromQL 示例**

#### 6.1 查询 CPU 使用率
```promql
rate(cpu_usage_seconds_total[5m])
```
查询过去 5 分钟内 CPU 使用率的变化速率。

#### 6.2 查询特定服务的 HTTP 请求数
```promql
sum(rate(http_requests_total{job="web"}[5m])) by (method)
```
查询 `web` 服务过去 5 分钟内的 HTTP 请求数，并按 HTTP 方法分组。

#### 6.3 查询内存使用率超过 80% 的主机
```promql
node_memory_Active_bytes / node_memory_MemTotal_bytes > 0.8
```
查询内存使用率超过 80% 的所有主机。

### 7. **PromQL 函数列表**

PromQL 提供了丰富的内置函数，用于对时间序列进行复杂操作。以下是一些常用函数：

- **rate()**：计算速率，常用于 Counter 类型的时间序列。
- **irate()**：类似 `rate()`，但仅使用最近两个数据点计算速率，适合对高分辨率的监控数据进行短期观察。
- **increase()**：计算 Counter 在给定时间范围内的增量。
- **delta()**：计算区间内的数值变化，适用于 Gauge 类型的时间序列。
- **histogram_quantile()**：用于计算 Histogram 数据的特定百分位数。

### 8. **进阶使用：子查询**

PromQL 支持子查询，即可以对一个查询结果再进行进一步的查询和计算。子查询的语法是 `(<expression>)[<range>:<resolution>]`。

```promql
avg_over_time(rate(http_requests_total[5m])[1h:])
```
该查询首先计算 HTTP 请求的 5 分钟变化速率，然后对过去 1 小时的数据进行平均。

### 9. **总结**

PromQL 是一个功能强大且灵活的查询语言，能够对 Prometheus 中存储的时间序列数据进行复杂的分析和处理。通过掌握基本的查询语法、操作符、聚合函数以及高级用法（如子查询），你可以从 Prometheus 中获取丰富的监控数据洞察。

