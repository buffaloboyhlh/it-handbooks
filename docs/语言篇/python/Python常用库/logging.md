# logging 模块

### Python `logging` 模块详解

`logging` 模块是 Python 内置的日志记录模块，提供了灵活且功能强大的日志记录系统，用于跟踪和记录程序的运行状态、调试信息、错误等内容。相比于简单的 `print()` 调试，`logging` 模块提供了更好的控制和管理日志的方式。

#### 1. 基本用法

`logging` 模块允许你通过日志记录器（Logger）来生成日志信息。每条日志消息都有一个严重性级别，默认从最低到最高依次为 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。

```python
import logging

# 设置基本配置
logging.basicConfig(level=logging.DEBUG)

# 记录不同级别的日志
logging.debug("这是调试信息")
logging.info("这是一般信息")
logging.warning("这是警告信息")
logging.error("这是错误信息")
logging.critical("这是严重错误信息")
```

#### 2. 配置日志

`logging.basicConfig()` 是快速设置日志记录器的最简单方法，可以通过它设置日志级别、格式、输出位置等。

- **设置日志级别**: 控制最低记录的日志级别。

  ```python
  logging.basicConfig(level=logging.INFO)
  ```

- **设置日志格式**: 格式化日志消息的输出方式。

  ```python
  logging.basicConfig(
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      level=logging.DEBUG
  )
  ```

- **将日志输出到文件**: 默认情况下，日志输出到控制台，你也可以指定输出到文件。

  ```python
  logging.basicConfig(
      filename='app.log',
      filemode='w',
      format='%(asctime)s - %(levelname)s - %(message)s',
      level=logging.DEBUG
  )
  ```

#### 3. Logger 对象

`Logger` 是日志系统的主体，用于记录日志消息。默认情况下，Python 提供了一个名为 `root` 的全局 Logger，但通常我们会创建自定义的 Logger。

- **创建 Logger 对象**:

  ```python
  logger = logging.getLogger('my_logger')
  logger.setLevel(logging.DEBUG)
  ```

- **Logger 继承性**: 子 Logger 继承父 Logger 的配置。

  ```python
  logger = logging.getLogger('my_logger.child')
  ```

#### 4. Handler 处理器

`Handler` 决定日志信息的输出位置。常用的 `Handler` 有以下几种：

- **StreamHandler**: 将日志输出到控制台。

  ```python
  console_handler = logging.StreamHandler()
  logger.addHandler(console_handler)
  ```

- **FileHandler**: 将日志输出到文件。

  ```python
  file_handler = logging.FileHandler('app.log')
  logger.addHandler(file_handler)
  ```

- **RotatingFileHandler**: 将日志输出到文件，并在文件达到一定大小时进行轮换。

  ```python
  rotating_handler = logging.handlers.RotatingFileHandler(
      'app.log', maxBytes=2000, backupCount=5
  )
  logger.addHandler(rotating_handler)
  ```

- **TimedRotatingFileHandler**: 将日志输出到文件，并在设定的时间间隔后进行轮换。

  ```python
  timed_handler = logging.handlers.TimedRotatingFileHandler(
      'app.log', when='midnight', interval=1, backupCount=7
  )
  logger.addHandler(timed_handler)
  ```

#### 5. Formatter 格式化器

`Formatter` 用于控制日志信息的最终输出格式。可以通过以下格式字符串来定制输出：

- `%(asctime)s`：日志记录的时间
- `%(name)s`：Logger 的名称
- `%(levelname)s`：日志级别
- `%(message)s`：日志消息

```python
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
```

#### 6. Filter 过滤器

`Filter` 用于更细粒度地控制哪些日志记录能够通过 Handler 进行输出。

- **简单示例**:

  ```python
  class CustomFilter(logging.Filter):
      def filter(self, record):
          return 'special' in record.msg

  logger.addFilter(CustomFilter())
  ```

#### 7. 配置日志：配置文件与字典配置

- **配置文件**:

  你可以通过 `configparser` 格式的配置文件来设置日志。

  ```ini
  [loggers]
  keys=root,simpleExample

  [handlers]
  keys=consoleHandler

  [formatters]
  keys=sampleFormatter

  [logger_root]
  level=DEBUG
  handlers=consoleHandler

  [logger_simpleExample]
  level=DEBUG
  handlers=consoleHandler
  qualname=simpleExample

  [handler_consoleHandler]
  class=StreamHandler
  level=DEBUG
  formatter=sampleFormatter
  args=(sys.stdout,)

  [formatter_sampleFormatter]
  format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
  datefmt=
  ```

  使用配置文件:

  ```python
  import logging
  import logging.config

  logging.config.fileConfig('logging.conf')
  logger = logging.getLogger('simpleExample')
  ```

- **字典配置**:

  ```python
  import logging.config

  dictConfig = {
      'version': 1,
      'formatters': {
          'default': {
              'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
          },
      },
      'handlers': {
          'console': {
              'class': 'logging.StreamHandler',
              'formatter': 'default',
          },
      },
      'root': {
          'level': 'DEBUG',
          'handlers': ['console']
      },
  }

  logging.config.dictConfig(dictConfig)
  logger = logging.getLogger()
  ```

#### 8. 日志捕获外部库的日志

你可以控制外部库的日志记录行为，例如将第三方库的日志记录级别设置为 `WARNING` 或更高：

```python
logging.getLogger('some_library').setLevel(logging.WARNING)
```

#### 9. 日志异常信息

`logging` 可以自动捕获和记录异常堆栈信息。

```python
try:
    1 / 0
except ZeroDivisionError:
    logging.error("除以零的错误", exc_info=True)
```

#### 10. 高级用法

- **多进程安全**:

  Python 的 `logging` 模块支持多进程环境中的日志记录，可以通过 `logging.handlers.QueueHandler` 和 `logging.handlers.QueueListener` 来实现日志记录的进程安全。

- **多线程安全**:

  `logging` 模块本身是线程安全的，适合在多线程应用中使用。

- **性能优化**:

  如果日志记录量大，可以使用 `NullHandler` 关闭不必要的日志输出，或使用高效的日志记录方法。

### 总结

`logging` 模块是 Python 强大而灵活的日志记录系统。通过 `Logger`、`Handler`、`Formatter` 和 `Filter` 等组件，你可以非常细致地控制日志的生成、格式化、输出和过滤。这使得它成为开发和调试复杂应用程序时不可或缺的工具。