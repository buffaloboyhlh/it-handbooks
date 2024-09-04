# time 模块 

`time` 模块是 Python 标准库中的一个模块，用于处理时间相关的任务。它提供了获取当前时间、暂停程序执行、格式化时间字符串等功能。以下是 `time` 模块的详细介绍。

### 1. 获取当前时间

#### 1.1 `time.time()`
- **功能**：返回当前时间的时间戳（自1970年1月1日00:00:00 UTC以来的秒数）。
- **示例**：
  ```python
  import time
  print(time.time())  # 输出类似于 1693205091.6098497
  ```

#### 1.2 `time.localtime([secs])`
- **功能**：将时间戳（秒数）转换为本地时间的 `struct_time` 对象。如果不提供 `secs` 参数，则使用当前时间。
- **示例**：
  ```python
  local_time = time.localtime()
  print(local_time)  # 输出类似于 time.struct_time(tm_year=2023, tm_mon=8, ...)
  ```

#### 1.3 `time.gmtime([secs])`
- **功能**：将时间戳（秒数）转换为UTC（世界协调时间）的 `struct_time` 对象。如果不提供 `secs` 参数，则使用当前时间。
- **示例**：
  ```python
  utc_time = time.gmtime()
  print(utc_time)  # 输出类似于 time.struct_time(tm_year=2023, tm_mon=8, ...)
  ```

### 2. 时间格式化

#### 2.1 `time.strftime(format, [t])`
- **功能**：将 `struct_time` 对象格式化为字符串。`format` 是格式字符串，`t` 是可选的 `struct_time` 对象（默认为当前时间）。
- **示例**：
  ```python
  formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
  print(formatted_time)  # 输出类似于 "2023-08-28 10:34:56"
  ```

#### 2.2 `time.strptime(string, format)`
- **功能**：将时间字符串解析为 `struct_time` 对象。`string` 是要解析的字符串，`format` 是解析格式。
- **示例**：
  ```python
  time_str = "2023-08-28 10:34:56"
  parsed_time = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
  print(parsed_time)  # 输出类似于 time.struct_time(tm_year=2023, tm_mon=8, ...)
  ```

### 3. 程序暂停和延时

#### 3.1 `time.sleep(secs)`
- **功能**：暂停程序执行指定的秒数。
- **示例**：
  ```python
  print("Start")
  time.sleep(2)  # 暂停2秒
  print("End")
  ```

### 4. 时间戳与结构化时间的相互转换

#### 4.1 `time.mktime(t)`
- **功能**：将本地时间的 `struct_time` 对象转换为时间戳（秒数）。
- **示例**：
  ```python
  local_time = time.localtime()
  timestamp = time.mktime(local_time)
  print(timestamp)  # 输出类似于 1693205091.0
  ```

#### 4.2 `time.asctime([t])`
- **功能**：将 `struct_time` 对象格式化为字符串。如果不提供 `t` 参数，则使用当前时间。
- **示例**：
  ```python
  print(time.asctime())  # 输出类似于 "Mon Aug 28 10:34:56 2023"
  ```

#### 4.3 `time.ctime([secs])`
- **功能**：将时间戳（秒数）转换为字符串。如果不提供 `secs` 参数，则使用当前时间。
- **示例**：
  ```python
  print(time.ctime())  # 输出类似于 "Mon Aug 28 10:34:56 2023"
  ```

### 5. 高精度时间函数

#### 5.1 `time.monotonic()`
- **功能**：返回一个单调递增的时钟时间点，不受系统时间修改的影响。
- **示例**：
  ```python
  start = time.monotonic()
  time.sleep(1)
  end = time.monotonic()
  print(f"Elapsed time: {end - start} seconds")  # 输出类似于 "Elapsed time: 1.00008759 seconds"
  ```

#### 5.2 `time.perf_counter()`
- **功能**：返回一个高精度的计时器时间点，适合测量时间间隔。
- **示例**：
  ```python
  start = time.perf_counter()
  time.sleep(1)
  end = time.perf_counter()
  print(f"Elapsed time: {end - start} seconds")  # 输出类似于 "Elapsed time: 1.00008759 seconds"
  ```

### 6. `struct_time` 对象

`struct_time` 是一种命名元组，包含以下属性：
- `tm_year`: 年份（如 2023）
- `tm_mon`: 月份（1 到 12）
- `tm_mday`: 日期（1 到 31）
- `tm_hour`: 小时（0 到 23）
- `tm_min`: 分钟（0 到 59）
- `tm_sec`: 秒（0 到 61，闰秒）
- `tm_wday`: 星期几（0 到 6，0 表示星期一）
- `tm_yday`: 一年中的第几天（1 到 366）
- `tm_isdst`: 是否夏令时（1 表示夏令时，0 表示非夏令时，-1 表示未知）

### 7. 应用场景

- **程序计时**：使用 `time.perf_counter()` 和 `time.monotonic()` 进行精确的程序计时。
- **日志记录**：使用 `time.strftime()` 格式化时间戳，用于日志记录。
- **时间解析**：使用 `time.strptime()` 解析时间字符串。

`time` 模块为 Python 提供了丰富的时间处理功能，适用于各种场景中的时间操作。