# queue 模块

`queue` 模块是 Python 标准库中的一个非常实用的工具，主要用于实现线程安全的队列。它可以在多线程环境中安全地使用，用于在生产者-消费者模式下在线程之间传递数据。`queue` 模块提供了几种不同类型的队列，包括先进先出（FIFO）、后进先出（LIFO）以及优先级队列（Priority Queue）。

### 1. 基本概念

`queue` 模块提供了以下几种主要的类：

- **`Queue`**: 先进先出队列（FIFO）
- **`LifoQueue`**: 后进先出队列（LIFO，类似于栈）
- **`PriorityQueue`**: 优先级队列，按照元素的优先级进行排序

这些队列都具备线程安全性，支持多线程环境下的并发操作。

### 2. `Queue` 类详解

#### 2.1 创建队列

```python
import queue

q = queue.Queue(maxsize=10)  # 创建一个最大长度为 10 的队列
```

- `maxsize`: 队列的最大长度。如果不指定或为 `0`，则表示队列大小没有限制。

#### 2.2 基本操作

- **`put(item)`**: 将元素 `item` 放入队列。如果队列已满，会阻塞直到队列有空间。
- **`put_nowait(item)`**: 将元素放入队列，如果队列已满则抛出 `queue.Full` 异常。

```python
q.put(1)
q.put_nowait(2)
```

- **`get()`**: 从队列中取出一个元素。如果队列为空，会阻塞直到有元素可取。
- **`get_nowait()`**: 从队列中取出一个元素，如果队列为空则抛出 `queue.Empty` 异常。

```python
item = q.get()
item = q.get_nowait()
```

- **`task_done()`**: 表示一个队列中的任务完成。当调用 `get()` 获取的任务处理完毕时，调用此方法。
- **`join()`**: 阻塞主线程，直到队列中所有任务处理完毕（即所有 `task_done()` 都被调用）。

```python
q.task_done()
q.join()
```

### 3. `LifoQueue` 类详解

`LifoQueue` 类实现了后进先出（LIFO）的队列，类似于栈。

```python
import queue

lifo_q = queue.LifoQueue()
lifo_q.put(1)
lifo_q.put(2)

print(lifo_q.get())  # 输出 2
print(lifo_q.get())  # 输出 1
```

操作方法与 `Queue` 类相同，只是取出的顺序是后进先出。

### 4. `PriorityQueue` 类详解

`PriorityQueue` 类实现了优先级队列，元素根据优先级排序，优先级越小的元素越早被取出。

```python
import queue

pq = queue.PriorityQueue()
pq.put((1, "task 1"))  # (优先级, 元素)
pq.put((3, "task 3"))
pq.put((2, "task 2"))

print(pq.get())  # 输出 (1, 'task 1')
print(pq.get())  # 输出 (2, 'task 2')
```

### 5. 阻塞与超时

- **阻塞模式**: `put()` 和 `get()` 方法在默认情况下会阻塞线程，直到可以执行操作。
- **超时**: 可以为 `put()` 和 `get()` 方法设置 `timeout` 参数，指定等待的最长时间。

```python
q.put(1, timeout=5)  # 最多等待5秒
item = q.get(timeout=5)
```

### 6. 常见使用场景

- **生产者-消费者模型**: 生产者线程向队列中放入任务，消费者线程从队列中取出任务进行处理。
- **多线程任务调度**: 队列可以用于调度和管理多个线程之间的任务。

### 7. `queue` 模块的异常

- **`queue.Full`**: 当使用 `put_nowait()` 往已满的队列中放入元素时抛出。
- **`queue.Empty`**: 当使用 `get_nowait()` 从空队列中取出元素时抛出。

### 总结

Python 的 `queue` 模块提供了强大的工具，用于在线程安全的环境中管理任务和数据流。无论是简单的 FIFO 队列，还是更复杂的 LIFO 或优先级队列，都可以根据实际需要进行选择并应用。