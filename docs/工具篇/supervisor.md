# Supervisor 教程

## Supervisor 教程 详解

Supervisor 是一个用 Python 编写的进程管理工具，常用于类 UNIX 操作系统下管理和监控进程，特别适合管理守护进程、长期运行的服务（如 web 应用、后台任务等）。它能够自动重启异常退出的进程，并提供强大的日志管理和 Web UI 界面，方便开发者监控和控制进程。

本文将深入介绍如何使用 Supervisor 管理进程，覆盖从安装、配置到高级功能及实践的各个方面。

### 目录
1. Supervisor 简介
2. Supervisor 安装
3. Supervisor 配置文件详解
4. Supervisor 常用命令
5. Supervisor 实战案例
6. Supervisor 高级功能
7. Supervisor Web 界面
8. 日志管理
9. 常见问题与解决方案
10. 总结

---

### 1. Supervisor 简介

Supervisor 的主要功能是管理进程，使得进程能够在后台运行、重启，并具备监控功能。它特别适合以下场景：

- **守护进程**：例如 Web 服务器、后台任务队列。
- **服务监控**：自动重启由于崩溃或退出而停止的服务。
- **日志管理**：集中管理所有子进程的日志。

#### 核心功能：

- **自动重启进程**：当进程崩溃或非正常退出时，Supervisor 会自动重启该进程。
- **日志管理**：可以记录进程的标准输出和标准错误日志，方便调试和监控。
- **进程组管理**：支持将多个进程归类管理，方便批量操作。
- **Web 界面**：通过简单的 Web 界面监控和控制进程状态。

---

### 2. Supervisor 安装

#### 2.1 安装 Supervisor

在 CentOS、Ubuntu 或其他基于 Linux 的系统中，可以通过包管理器或 `pip` 安装 Supervisor。

##### 使用 `yum` 安装（CentOS）

```bash
sudo yum install epel-release
sudo yum install supervisor
```

##### 使用 `apt` 安装（Ubuntu）

```bash
sudo apt-get update
sudo apt-get install supervisor
```

##### 使用 `pip` 安装（适用于任何 Python 环境）

```bash
pip install supervisor
```

#### 2.2 启动 Supervisor

安装完成后，通过以下命令启动 Supervisor 守护进程：

```bash
sudo systemctl start supervisord
sudo systemctl enable supervisord
```

默认配置文件路径为 `/etc/supervisord.conf`，其中包含 Supervisor 的主要配置。

---

### 3. Supervisor 配置文件详解

Supervisor 的配置文件采用 `.ini` 格式，每个部分定义不同的功能和参数。

#### 3.1 生成默认配置文件

如果需要自定义配置文件，可以通过以下命令生成默认配置：

```bash
echo_supervisord_conf > /etc/supervisord.conf
```

#### 3.2 配置文件结构

配置文件主要由以下几个部分组成：

##### **[unix_http_server]**
用于定义 Supervisor 与 `supervisorctl` 或 Web 界面之间的通信方式，通常通过 Unix socket 连接。

```ini
[unix_http_server]
file=/var/run/supervisor.sock   ; 定义 socket 文件路径
chmod=0700                      ; 权限设置
```

##### **[inet_http_server]**
定义 HTTP 服务器配置，用于启用 Web 界面。

```ini
[inet_http_server]
port=127.0.0.1:9001             ; 定义监听的 IP 和端口
username=user                   ; Web 界面登录用户名
password=pass                   ; Web 界面登录密码
```

##### **[supervisord]**
用于定义 Supervisor 主进程的配置，如日志、pid 文件等。

```ini
[supervisord]
logfile=/var/log/supervisord.log ; 主日志文件路径
pidfile=/var/run/supervisord.pid ; 守护进程的 pid 文件
childlogdir=/var/log/supervisor/ ; 子进程的日志目录
```

##### **[program:x]**
每个需要管理的进程都通过 `[program:x]` 定义。可以管理多个子进程。

```ini
[program:myapp]
command=/path/to/your/program    ; 启动命令
autostart=true                   ; Supervisor 启动时是否自动启动
autorestart=true                 ; 如果进程崩溃是否自动重启
stderr_logfile=/var/log/myapp.err.log  ; 错误日志路径
stdout_logfile=/var/log/myapp.out.log  ; 输出日志路径
```

---

### 4. Supervisor 常用命令

Supervisor 提供了命令行工具 `supervisorctl`，用于管理和监控进程。常见命令如下：

#### 4.1 查看进程状态

查看所有由 Supervisor 管理的进程状态：

```bash
supervisorctl status
```

#### 4.2 启动进程

启动某个由 Supervisor 管理的进程：

```bash
supervisorctl start myapp
```

#### 4.3 停止进程

停止某个进程：

```bash
supervisorctl stop myapp
```

#### 4.4 重启进程

重启某个进程：

```bash
supervisorctl restart myapp
```

#### 4.5 重新读取配置文件

当修改了 Supervisor 的配置文件后，使用以下命令重新读取配置：

```bash
supervisorctl reread
supervisorctl update
```

---

### 5. Supervisor 实战案例

#### 案例：使用 Supervisor 管理 Gunicorn

假设我们有一个 Django 项目，并使用 Gunicorn 作为 WSGI 服务器，可以通过 Supervisor 管理 Gunicorn 进程。

#### 5.1 配置 Gunicorn 的 Supervisor 配置

在 `/etc/supervisord.d/gunicorn.conf` 文件中添加以下内容：

```ini
[program:gunicorn]
command=/path/to/venv/bin/gunicorn myproject.wsgi:application --bind 127.0.0.1:8000
directory=/path/to/myproject
autostart=true
autorestart=true
stderr_logfile=/var/log/gunicorn.err.log
stdout_logfile=/var/log/gunicorn.out.log
user=www-data
```

#### 5.2 启动 Supervisor 管理 Gunicorn

启动 Supervisor 并让其管理 Gunicorn：

```bash
supervisorctl reread
supervisorctl update
supervisorctl start gunicorn
```

Gunicorn 将由 Supervisor 启动和管理，任何崩溃或异常都会触发自动重启。

---

### 6. Supervisor 高级功能

#### 6.1 进程组管理

Supervisor 允许你将多个进程归类为一组，便于对整组进程进行统一操作。

在配置文件中定义一个进程组：

```ini
[group:mygroup]
programs=myapp1,myapp2
```

你可以使用以下命令来控制整个组：

```bash
supervisorctl start mygroup:*
supervisorctl stop mygroup:*
```

#### 6.2 环境变量

通过 `environment` 参数可以为某个进程定义自定义的环境变量。

```ini
[program:myapp]
command=/path/to/myapp
environment=LANG=en_US.UTF-8,LC_ALL=en_US.UTF-8
```

---

### 7. Supervisor Web 界面

Supervisor 提供了一个简洁的 Web 界面，便于通过浏览器监控和控制进程。

#### 7.1 启用 Web 界面

在配置文件中启用 Web 界面：

```ini
[inet_http_server]
port=127.0.0.1:9001   ; 定义监听的 IP 和端口
username=admin        ; 登录用户名
password=admin        ; 登录密码
```

重启 Supervisor：

```bash
sudo supervisorctl reread
sudo supervisorctl update
```

然后，访问 `http://127.0.0.1:9001`，输入配置的用户名和密码，即可查看和管理进程。

---

### 8. 日志管理

Supervisor 可以对进程的标准输出和标准错误进行日志记录，并提供日志轮转功能。

#### 8.1 设置日志路径

在每个进程的配置中，通过 `stdout_logfile` 和 `stderr_logfile` 来指定日志路径：

```ini
[program:myapp]
stdout_logfile=/var/log/myapp.out.log
stderr_logfile=/var/log/myapp.err.log
```

#### 8.2 日志轮转

Supervisor 支持日志轮转（log rotation），可以通过以下参数设置：

```ini
stdout_logfile_maxbytes=50MB  ; 单个日志文件的最大大小
stdout_logfile_backups=10     ; 保留的旧日志文件数量
```

---

### 9. 常见问题与解决方案

#### 9.1 权限问题

如果你的进程需要以非 root 用户身份运行，请确保在配置文件中设置正确的 `user` 参数，并保证该用户有权限访问相关文件和目录。

#### 9.2 配置文件修改后不生效

当你修改了 Supervisor 的配置文件后，需要使用以下命令重新加载配置：

```bash
supervisorctl reread
supervisorctl update
```

---

### 10. 总结

Supervisor 是一个功能强大的进程管理工具，尤其适合管理长期运行的 Python 应用程序。通过 Supervisor，开发者可以方便地管理后台服务、记录日志、重启进程并通过 Web 界面监控系统。