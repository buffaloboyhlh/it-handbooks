# RabbitMQ 基础

RabbitMQ 是一个开源的消息代理软件，用于在分布式系统中传递消息。它通过高级消息队列协议（AMQP）来进行消息的存储、传输和处理。以下是详细的 RabbitMQ 使用教程。

### 1. **RabbitMQ 简介**
RabbitMQ 是一个消息队列系统，它允许应用程序、服务和系统之间异步地发送和接收消息。RabbitMQ 支持多种消息传递协议，并可以在不同的编程语言和操作系统中使用。

### 2. **RabbitMQ 的核心概念**
- **Producer（生产者）**：发送消息到 RabbitMQ。
- **Consumer（消费者）**：从 RabbitMQ 获取并处理消息。
- **Queue（队列）**：消息的存储位置。生产者发送消息到队列，消费者从队列接收消息。
- **Exchange（交换机）**：用于将消息路由到不同的队列。常见的交换机类型包括 `direct`、`topic`、`fanout` 和 `headers`。
- **Binding（绑定）**：绑定定义了交换机与队列之间的关系，即消息从交换机流向哪个队列。
- **Routing Key（路由键）**：消息的路由规则，决定消息如何从交换机路由到特定队列。
- **Connection（连接）**：客户端与 RabbitMQ 服务器之间的 TCP 连接。
- **Channel（通道）**：在连接内进行多路复用，节省资源。

### 3. **RabbitMQ 安装与配置**
#### 3.1. **前置条件**
- **Erlang 环境**：RabbitMQ 是基于 Erlang 语言开发的，因此需要先安装 Erlang。

#### 3.2. **安装 RabbitMQ**
1. **安装 Erlang**
   在 Ubuntu 上安装：
   ```bash
   sudo apt-get update
   sudo apt-get install erlang
   ```
   在 CentOS 上安装：
   ```bash
   sudo yum update
   sudo yum install erlang
   ```

2. **安装 RabbitMQ**
   在 Ubuntu 上安装：
   ```bash
   sudo apt-get install rabbitmq-server
   ```
   在 CentOS 上安装：
   ```bash
   sudo yum install rabbitmq-server
   ```

3. **启动 RabbitMQ 服务**
   ```bash
   sudo systemctl start rabbitmq-server
   sudo systemctl enable rabbitmq-server
   ```

4. **验证安装**
   检查 RabbitMQ 服务状态：
   ```bash
   sudo systemctl status rabbitmq-server
   ```

### 4. **RabbitMQ 基本操作**
#### 4.1. **启用管理插件**
RabbitMQ 提供了一个 Web 管理界面，可以通过浏览器管理和监控 RabbitMQ 实例。
```bash
sudo rabbitmq-plugins enable rabbitmq_management
```
启用后，通过浏览器访问 `http://localhost:15672/`，默认用户名和密码都是 `guest`。

#### 4.2. **创建用户与权限管理**
创建一个新的用户并赋予其权限：
```bash
sudo rabbitmqctl add_user myuser mypassword
sudo rabbitmqctl set_user_tags myuser administrator
sudo rabbitmqctl set_permissions -p / myuser ".*" ".*" ".*"
```

#### 4.3. **声明队列**
使用 RabbitMQ 客户端（如 Python 的 `pika`）声明队列：
```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 声明一个队列
channel.queue_declare(queue='hello')
```

#### 4.4. **发送消息**
```python
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Hello RabbitMQ!')
print(" [x] Sent 'Hello RabbitMQ!'")
```

#### 4.5. **接收消息**
```python
def callback(ch, method, properties, body):
    print(f" [x] Received {body}")

channel.basic_consume(queue='hello',
                      on_message_callback=callback,
                      auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
```

### 5. **高级使用**
#### 5.1. **交换机与绑定**
RabbitMQ 通过交换机和绑定将消息路由到正确的队列。下面是几种常见的交换机类型：
- **Direct Exchange**：消息会被路由到绑定了相同路由键的队列。
- **Topic Exchange**：基于通配符的路由键，消息可以路由到匹配的多个队列。
- **Fanout Exchange**：将消息广播到所有绑定的队列。
- **Headers Exchange**：基于消息的 headers 属性来路由。

#### 5.2. **消息确认与重试**
- **消息确认**：RabbitMQ 支持消息确认机制，确保消息被正确处理。如果消息未被确认，RabbitMQ 会重新投递消息。
- **重试机制**：通过 `Dead Letter Exchange (DLX)` 和 `TTL`（Time to Live）可以实现消息的重试逻辑。

#### 5.3. **消息持久化**
队列和消息可以配置为持久化，以防止 RabbitMQ 服务器重启或崩溃时消息丢失。持久化队列：
```python
channel.queue_declare(queue='durable_queue', durable=True)
```
持久化消息：
```python
channel.basic_publish(exchange='',
                      routing_key='durable_queue',
                      body='Persistent message',
                      properties=pika.BasicProperties(
                          delivery_mode=2,  # make message persistent
                      ))
```

### 6. **监控与管理**
RabbitMQ 提供了多种监控和管理工具，如：
- **rabbitmqctl**：命令行工具，用于管理 RabbitMQ 节点。
- **Web 管理界面**：用于查看队列状态、交换机、绑定以及消息流量。
- **Prometheus 与 Grafana**：可以结合使用监控 RabbitMQ 的各项指标。

### 7. **集群与高可用性**
RabbitMQ 支持集群部署，通过多个节点提高系统的可用性和性能。常见的集群模式有：
- **普通集群**：节点之间共享消息队列元数据，但不共享消息内容。
- **高可用队列（HA）**：通过 `mirrored queues` 机制，将队列镜像到集群中的其他节点，从而实现高可用。

### 8. **常见问题与故障排查**
- **无法连接 RabbitMQ**：检查防火墙设置，确保 5672 和 15672 端口开放。
- **消息堆积**：检查消费者是否正常工作，是否需要扩展消费者的数量或性能。
- **内存或磁盘告警**：RabbitMQ 对内存和磁盘使用有一定限制，确保节点有足够的资源，或者调整相关配置。

以上是 RabbitMQ 的详细使用教程，涵盖了从基础概念、安装配置到高级使用的各个方面，希望对你有所帮助！