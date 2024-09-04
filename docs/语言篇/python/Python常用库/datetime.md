# datetime 模块

### Python `datetime` 模块详解

`datetime` 模块提供了用于处理日期和时间的类。它允许你创建、操作和格式化日期和时间数据。该模块是处理时间戳、日期和时间相关任务的核心工具。

#### 1. 导入模块

```python
import datetime
```

#### 2. `datetime` 模块中的主要类

- **`datetime.date`**: 表示日期（年、月、日）。
- **`datetime.time`**: 表示时间（时、分、秒、微秒）。
- **`datetime.datetime`**: 表示日期和时间。
- **`datetime.timedelta`**: 表示两个日期或时间之间的时间差。
- **`datetime.tzinfo`**: 用于处理时区信息的抽象基类。

#### 3. `datetime.date` 类

- **获取当前日期**

```python
today = datetime.date.today()
print(today)  # 输出例如 2024-08-27
```

- **创建日期对象**

```python
date_obj = datetime.date(2024, 8, 27)
print(date_obj)  # 输出 2024-08-27
```

- **获取日期的年、月、日**

```python
year = date_obj.year
month = date_obj.month
day = date_obj.day
print(year, month, day)  # 输出 2024 8 27
```

- **获取星期几**

```python
weekday = date_obj.weekday()  # Monday = 0, Sunday = 6
print(weekday)  # 输出 1 表示星期二
```

#### 4. `datetime.time` 类

- **创建时间对象**

```python
time_obj = datetime.time(14, 30, 45)
print(time_obj)  # 输出 14:30:45
```

- **获取时间的时、分、秒、微秒**

```python
hour = time_obj.hour
minute = time_obj.minute
second = time_obj.second
microsecond = time_obj.microsecond
print(hour, minute, second, microsecond)  # 输出 14 30 45 0
```

#### 5. `datetime.datetime` 类

- **获取当前日期和时间**

```python
now = datetime.datetime.now()
print(now)  # 输出例如 2024-08-27 14:30:45.123456
```

- **创建日期时间对象**

```python
datetime_obj = datetime.datetime(2024, 8, 27, 14, 30, 45)
print(datetime_obj)  # 输出 2024-08-27 14:30:45
```

- **从日期和时间对象创建日期时间对象**

```python
date_obj = datetime.date(2024, 8, 27)
time_obj = datetime.time(14, 30)
datetime_obj = datetime.datetime.combine(date_obj, time_obj)
print(datetime_obj)  # 输出 2024-08-27 14:30:00
```

- **获取日期时间的各个部分**

```python
year = datetime_obj.year
month = datetime_obj.month
day = datetime_obj.day
hour = datetime_obj.hour
minute = datetime_obj.minute
second = datetime_obj.second
print(year, month, day, hour, minute, second)  # 输出 2024 8 27 14 30 45
```

#### 6. `datetime.timedelta` 类

- **创建时间差对象**

```python
delta = datetime.timedelta(days=5, hours=2, minutes=30)
print(delta)  # 输出 5 days, 2:30:00
```

- **日期时间运算**

```python
future_date = datetime_obj + delta
print(future_date)  # 输出 2024-09-01 17:00:45

past_date = datetime_obj - delta
print(past_date)  # 输出 2024-08-22 12:00:45
```

- **获取两个日期之间的差异**

```python
start_date = datetime.datetime(2024, 8, 20)
end_date = datetime.datetime(2024, 8, 27)
delta = end_date - start_date
print(delta)  # 输出 7 days, 0:00:00
```

#### 7. 日期时间格式化和解析

- **格式化日期时间为字符串**

`strftime()` 函数用于格式化日期和时间对象。

```python
formatted_date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_date)  # 输出 2024-08-27 14:30:45
```

常用的格式化符号：

- `%Y`: 四位数的年份
- `%m`: 两位数的月份（01-12）
- `%d`: 两位数的日期（01-31）
- `%H`: 24小时制的小时（00-23）
- `%M`: 两位数的分钟（00-59）
- `%S`: 两位数的秒数（00-59）

- **将字符串解析为日期时间对象**

`strptime()` 函数用于将字符串解析为日期时间对象。

```python
date_str = "2024-08-27 14:30:45"
parsed_date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
print(parsed_date)  # 输出 2024-08-27 14:30:45
```

#### 8. `datetime.tzinfo` 类和时区处理

- **带时区的日期时间**

Python 提供了 `pytz` 库来处理时区信息，可以结合 `datetime.datetime` 使用。

```python
import pytz

timezone = pytz.timezone('Asia/Shanghai')
aware_datetime = datetime.datetime.now(timezone)
print(aware_datetime)  # 输出 2024-08-27 14:30:45.123456+08:00
```

- **转换时区**

```python
new_timezone = pytz.timezone('America/New_York')
new_timezone_datetime = aware_datetime.astimezone(new_timezone)
print(new_timezone_datetime)  # 输出 2024-08-27 02:30:45.123456-04:00
```

### 总结

`datetime` 模块是处理日期和时间的基础工具，提供了创建、操作和格式化日期时间的多种方法。掌握这个模块的使用可以大大简化你在程序中处理时间相关任务的复杂度，尤其是在处理跨时区的应用时，结合 `pytz` 等库可以实现更强大的功能。