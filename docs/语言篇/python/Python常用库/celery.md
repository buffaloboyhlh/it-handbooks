# celery 教程

### Celery 教程：概念和使用方法

Celery 是一个广泛使用的分布式任务队列系统，支持实时任务处理和调度任务。它能够轻松地处理大量并发任务，常用于 Web 应用程序的后台任务处理。下面我们来详细介绍 Celery 的核心概念和基本使用方法。

### 1. **Celery 核心概念**
- **任务（Task）**：Celery 中的基本单位，即一个函数或操作。可以被 worker 异步执行。
- **队列（Queue）**：存放任务的地方，worker 会从队列中获取任务并执行。
- **Worker**：执行任务的进程。通常会启动多个 worker 来处理不同队列中的任务。
- **消息代理（Broker）**：负责传递任务消息，常用的代理包括 RabbitMQ 和 Redis。
- **结果后端（Result Backend）**：用于存储任务的执行结果，便于后续查询。

### 2. **安装 Celery 和消息代理**
首先，使用 `pip` 安装 Celery 以及消息代理（以 Redis 为例）：

```bash
pip install celery[redis]
```

### 3. **创建 Celery 应用**
在项目中创建一个名为 `tasks.py` 的文件，定义一个简单的 Celery 应用和任务函数：

```python
from celery import Celery

# 创建 Celery 实例
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# 定义一个简单的任务
@app.task
def add(x, y):
    return x + y
```

- **broker**：指定使用 Redis 作为消息代理。
- **backend**：指定使用 Redis 存储任务的执行结果。

### 4. **运行 Celery Worker**
在命令行中启动 Celery worker 进程，以便监听任务并执行：

```bash
celery -A tasks worker --loglevel=info
```

- **-A tasks**：指定 Celery 应用所在的模块，即 `tasks.py`。
- **--loglevel=info**：设置日志输出的级别。

### 5. **调用任务**
在另一个 Python 会话中异步调用任务：

```python
from tasks import add

result = add.delay(4, 6)
print(result.id)  # 输出任务的唯一 ID
print(result.get(timeout=10))  # 获取任务结果，最多等待 10 秒
```

- **delay**：这是 Celery 中异步调用任务的方式。
- **result.get()**：用于获取任务的返回值，可以设置超时时间。

### 6. **定时任务调度**
Celery 支持定时任务调度（类似于 cron）。你可以使用 Celery 的 `beat` 组件来调度任务：

```python
from celery import Celery
from celery.schedules import crontab

app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def scheduled_task():
    print("This is a scheduled task.")

# 配置定时任务
app.conf.beat_schedule = {
    'add-every-30-seconds': {
        'task': 'tasks.scheduled_task',
        'schedule': 30.0,  # 每 30 秒执行一次
    },
}
```

启动 Celery worker 和 beat：

```bash
celery -A tasks worker --loglevel=info
celery -A tasks beat --loglevel=info
```

### 7. **任务重试**
如果任务执行失败，Celery 可以自动重试任务：

```python
@app.task(bind=True, max_retries=3)
def unreliable_task(self):
    try:
        # 可能会失败的操作
        pass
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)  # 60 秒后重试
```

### 8. **监控和管理**
Celery 提供了 `Flower` 监控工具，可以实时监控任务的执行状态：

```bash
pip install flower
celery -A tasks flower
```

通过 `Flower` 的 Web 界面，可以查看任务执行情况、队列状态等。

### 9. **优化与扩展**
在实际生产环境中，可以考虑以下优化措施：
- **任务路由**：将不同类型的任务分配到不同的队列，提高系统的吞吐量。
- **并发控制**：使用多进程或多线程 worker 来处理大量并发任务。
- **任务预取**：通过调整预取数量来控制 worker 预取任务的数量，避免任务堆积。


# celery 进阶

这里是一些关于 Celery 的更多进阶教程和使用技巧，帮助你更深入地理解和使用 Celery。

### 1. **链式任务（Chord）**
链式任务允许将多个任务按顺序执行，其中每个任务的输出作为下一个任务的输入。你可以使用 `chain` 方法实现链式任务：

```python
from celery import chain
from tasks import add, multiply

result = chain(add.s(4, 6) | multiply.s(3))()
print(result.get())
```

- `chain(add.s(4, 6) | multiply.s(3))`：将 `add` 和 `multiply` 任务链接在一起，`add` 的结果将传递给 `multiply`。

### 2. **分组任务（Group）**
分组任务允许并行执行多个任务，并在所有任务完成后返回结果。你可以使用 `group` 来创建任务组：

```python
from celery import group
from tasks import add

result = group(add.s(i, i) for i in range(10))()
print(result.get())
```

- `group(add.s(i, i) for i in range(10))`：并行执行多个 `add` 任务。

### 3. **任务重试与回退机制**
当任务失败时，Celery 可以自动重试，并在多次失败后执行特定的回退操作：

```python
@app.task(bind=True, max_retries=5)
def send_email(self, email_address):
    try:
        # 发送邮件的代码
        pass
    except SomeSpecificException as exc:
        # 重试3次后执行回退操作
        self.retry(exc=exc, countdown=60)
```

- `max_retries`：定义最大重试次数。
- `countdown`：在重试之前等待的秒数。

### 4. **使用自定义队列**
你可以创建和使用自定义队列来处理不同类型的任务。可以通过在 Celery 应用配置中定义队列：

```python
app.conf.task_queues = (
    Queue('default', routing_key='task.#'),
    Queue('emails', routing_key='email.#'),
)
```

并通过 `queue` 参数指定任务的队列：

```python
@app.task(queue='emails')
def send_email(email_address):
    pass
```

### 5. **任务路由**
任务路由允许你将特定的任务发送到指定的队列中。可以通过 `task_routes` 参数配置路由：

```python
app.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.send_email': {'queue': 'emails'},
}
```

### 6. **使用 Celery Signals**
Celery 提供了信号机制，允许你在任务执行的不同阶段执行一些特定操作。常用的信号包括：

- **task_prerun**：任务开始执行前触发。
- **task_postrun**：任务执行结束后触发。
- **task_failure**：任务失败时触发。

示例：

```python
from celery.signals import task_prerun, task_postrun

@task_prerun.connect
def task_start_handler(sender=None, **kwargs):
    print(f"Task {sender.name} is starting...")

@task_postrun.connect
def task_end_handler(sender=None, **kwargs):
    print(f"Task {sender.name} has finished.")
```

### 7. **Celery 与 Django 集成**
Celery 与 Django 框架集成非常紧密，可以方便地处理异步任务和定时任务。以下是 Django 项目中使用 Celery 的步骤：

1. **安装 Celery 和 Django-Celery-Beat**：
   ```bash
   pip install celery django-celery-beat
   ```

2. **配置 Celery**：
   在 Django 项目根目录创建 `celery.py` 文件：

   ```python
   from __future__ import absolute_import, unicode_literals
   import os
   from celery import Celery

   os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')

   app = Celery('your_project')
   app.config_from_object('django.conf:settings', namespace='CELERY')
   app.autodiscover_tasks()
   ```

3. **在 `__init__.py` 中加载 Celery**：
   ```python
   from __future__ import absolute_import, unicode_literals
   from .celery import app as celery_app

   __all__ = ('celery_app',)
   ```

4. **创建任务**：
   在任意 Django app 中的 `tasks.py` 文件中创建任务：

   ```python
   from celery import shared_task

   @shared_task
   def my_task():
       # 执行一些操作
       pass
   ```

5. **启动 Celery Worker**：
   ```bash
   celery -A your_project worker -l info
   ```

6. **定时任务**：
   使用 `django-celery-beat` 可以在 Django Admin 中管理定时任务。


这里提供更多关于 Celery 的进阶使用技巧和特性，帮助你在复杂场景中更加高效地使用 Celery。

### 1. **链式任务与回调（Chaining and Callbacks）**
Celery 允许你将多个任务链接起来执行，支持任务之间的依赖关系。你可以通过 `chain`、`chord`、`group` 等方式实现复杂的任务链。

- **链式任务**：按顺序执行多个任务，每个任务的输出将作为下一个任务的输入。

  ```python
  from celery import chain

  result = chain(task1.s(arg1), task2.s(), task3.s())()
  ```

- **任务组（Group）**：并行执行一组任务。

  ```python
  from celery import group

  result = group(task1.s(), task2.s(), task3.s())()
  ```

- **回调（Callback）**：在任务组执行完成后执行的任务。

  ```python
  from celery import chord

  result = chord(group(task1.s(), task2.s()))(callback_task.s())
  ```

### 2. **任务重试策略**
在处理不稳定的外部服务时，任务可能会偶尔失败。Celery 提供了自动重试机制来处理这些失败情况。

```python
@app.task(bind=True, max_retries=3)
def send_email(self, to):
    try:
        # 发送邮件操作
        pass
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)  # 60秒后重试
```

- `max_retries`：最大重试次数。
- `countdown`：在重试之前等待的秒数。

### 3. **任务优先级**
Celery 支持任务优先级，允许你控制任务在队列中的处理顺序。任务优先级的范围通常是 0（最高优先级）到 9（最低优先级）。

```python
@app.task(priority=1)
def high_priority_task():
    pass
```

### 4. **任务超时控制**
对于需要限制执行时间的任务，可以设置任务的超时时间，防止任务长期占用资源。

```python
@app.task(time_limit=30)
def long_running_task():
    # 任务逻辑
    pass
```

- `time_limit`：任务的执行时间限制，单位为秒。

### 5. **配置任务预取（Prefetching）**
预取控制 worker 预取任务的数量，这对性能优化非常重要。可以通过 `worker_prefetch_multiplier` 配置项进行设置。

```python
app.conf.worker_prefetch_multiplier = 1
```

### 6. **任务的路由与交换机**
你可以根据任务类型将任务发送到不同的队列，以实现更好的资源隔离和调度控制。

```python
app.conf.task_routes = {
    'myapp.tasks.add': {'queue': 'low_priority'},
    'myapp.tasks.multiply': {'queue': 'high_priority'},
}
```

- **自定义交换机与队列**：通过定义不同的交换机和队列，可以实现更复杂的路由机制。

  ```python
  from kombu import Exchange, Queue

  app.conf.task_queues = (
      Queue('default', Exchange('default'), routing_key='default'),
      Queue('emails', Exchange('emails'), routing_key='email.#'),
  )
  ```

### 7. **分布式锁**
在某些情况下，你可能需要确保某一时间点只有一个任务实例在运行，这时可以使用分布式锁。例如，通过 Redis 实现一个简单的分布式锁：

```python
import redis
from celery import Task

class LockedTask(Task):
    _lock = None

    def __call__(self, *args, **kwargs):
        self._lock = redis.StrictRedis().lock(self.name, timeout=10)
        if self._lock.acquire(blocking=False):
            try:
                return super().__call__(*args, **kwargs)
            finally:
                self._lock.release()
        else:
            raise Exception("Task is already running")

@app.task(base=LockedTask)
def my_locked_task():
    # 任务逻辑
    pass
```

### 8. **监控与健康检查**
Celery 提供了多种工具和机制来监控任务执行状态，并确保系统的健康运行。

- **Flower**：实时监控和管理 Celery 的任务和队列。

  ```bash
  celery -A your_project flower
  ```

- **健康检查**：通过 Celery 的 `ping` 任务或自定义健康检查任务来定期检查 worker 的健康状态。

  ```python
  app.control.ping()
  ```

### 9. **使用 Celery Signals**
Celery Signals 提供了任务生命周期中的各种钩子，允许你在任务执行的不同阶段插入自定义逻辑。

```python
from celery.signals import task_success, task_failure

@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    print(f'Task {sender.name} succeeded with result: {result}')

@task_failure.connect
def task_failure_handler(sender=None, exception=None, **kwargs):
    print(f'Task {sender.name} failed with exception: {exception}')
```

### 10. **动态调整 Worker**
在高峰期，你可能需要增加 worker 的数量来处理更多的任务，Celery 支持动态调整 worker 的数量。

```bash
celery -A your_project worker --autoscale=10,3
```

- `--autoscale=10,3`：动态调整 worker 数量，最多 10 个 worker，最少 3 个。

这些进阶教程和使用技巧可以帮助你在生产环境中更加灵活、高效地使用 Celery，处理更复杂的任务调度和执行场景。

