# nginx 基础

Nginx 是一个高性能的 HTTP 服务器、反向代理服务器和邮件代理服务器。它被广泛用于负载均衡、静态内容提供、反向代理等应用场景。以下是 Nginx 的详细教程，涵盖从基本安装到配置和常见应用。

### 一、Nginx 简介
Nginx 由 Igor Sysoev 开发，最初用于俄罗斯的 Rambler.ru 网站。它以高性能、低资源消耗和模块化设计著称，能够处理大量的并发连接，是当前最受欢迎的 Web 服务器之一。

### 二、Nginx 安装

#### 2.1 在 Ubuntu 上安装
```bash
sudo apt update
sudo apt install nginx
```

#### 2.2 在 CentOS 上安装
```bash
sudo yum update
sudo yum install nginx
```

#### 2.3 启动和管理 Nginx
安装完成后，可以通过以下命令来启动、停止和重启 Nginx：
```bash
sudo systemctl start nginx    # 启动 Nginx
sudo systemctl stop nginx     # 停止 Nginx
sudo systemctl restart nginx  # 重启 Nginx
```
你可以通过访问 `http://your_server_ip/` 来检查 Nginx 是否运行成功。

### 三、Nginx 配置文件结构

Nginx 的主配置文件通常位于 `/etc/nginx/nginx.conf`。Nginx 的配置文件由指令和块组成，常见的块包括 `events`、`http` 和 `server` 块。

#### 3.1 配置文件示例
```nginx
user  nginx;
worker_processes  1;

error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;

events {
    worker_connections  1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile        on;
    keepalive_timeout  65;

    include /etc/nginx/conf.d/*.conf;
}
```
- `user`：指定运行 Nginx 的用户。
- `worker_processes`：指定工作进程的数量。
- `error_log`：定义错误日志的路径。
- `events`：设置工作进程的事件处理模型。
- `http`：包含处理 HTTP 请求的配置。

### 四、Nginx 基本配置

#### 4.1 创建服务器块（Server Block）
服务器块用于定义不同的网站配置，类似于 Apache 的虚拟主机。你可以为每个站点创建一个单独的服务器块。
```nginx
server {
    listen       80;
    server_name  example.com www.example.com;

    location / {
        root   /var/www/example.com;
        index  index.html index.htm;
    }

    error_page  404 /404.html;
    location = /40x.html {
    }

    error_page   500 502 503 504  /50x.html;
    location = /50x.html {
    }
}
```
- `listen`：指定 Nginx 监听的端口（通常为 80 或 443）。
- `server_name`：定义服务器名称，可以是域名或 IP 地址。
- `location`：定义请求 URI 匹配的处理方式。
- `root`：指定站点的根目录。
- `index`：定义默认的索引文件。

#### 4.2 设置默认站点
默认站点通常配置在 `/etc/nginx/sites-available/default` 中，可以通过修改这个文件来设置默认站点。

### 五、Nginx 反向代理

#### 5.1 基本反向代理配置
Nginx 可以作为反向代理服务器，将请求转发到后端服务器。
```nginx
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
- `proxy_pass`：定义请求转发的后端服务器。
- `proxy_set_header`：设置传递到后端服务器的头信息。

### 六、Nginx 负载均衡

#### 6.1 轮询负载均衡
Nginx 支持多种负载均衡算法，轮询是最基本的一种。
```nginx
upstream backend {
    server backend1.example.com;
    server backend2.example.com;
}

server {
    location / {
        proxy_pass http://backend;
    }
}
```

#### 6.2 加权轮询负载均衡
可以为不同的服务器分配不同的权重，以便某些服务器处理更多的请求。
```nginx
upstream backend {
    server backend1.example.com weight=3;
    server backend2.example.com weight=1;
}

server {
    location / {
        proxy_pass http://backend;
    }
}
```

### 七、Nginx 安全配置

#### 7.1 设置 HTTPS
Nginx 支持 SSL/TLS，可以通过配置 HTTPS 来保护网站安全。
```nginx
server {
    listen 443 ssl;
    server_name example.com;

    ssl_certificate /etc/ssl/certs/example.com.crt;
    ssl_certificate_key /etc/ssl/private/example.com.key;

    location / {
        root   /var/www/example.com;
        index  index.html index.htm;
    }
}
```
- `ssl_certificate`：指定 SSL 证书文件路径。
- `ssl_certificate_key`：指定 SSL 证书密钥文件路径。

#### 7.2 禁用不安全的协议和加密套件
确保仅使用强加密协议和套件来增强安全性。
```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'HIGH:!aNULL:!MD5';
```

### 八、Nginx 性能优化

#### 8.1 缓存静态内容
通过缓存静态内容，可以减少服务器负载并加快页面加载速度。
```nginx
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
    expires 30d;
    access_log off;
}
```
- `expires`：设置缓存过期时间。

#### 8.2 启用 Gzip 压缩
Gzip 压缩可以减少传输的数据量，从而提高响应速度。
```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
gzip_min_length 1000;
```

### 九、Nginx 常见问题排查

#### 9.1 检查配置文件语法
在应用配置更改之前，可以通过以下命令检查 Nginx 配置文件的语法：
```bash
sudo nginx -t
```

#### 9.2 查看日志文件
Nginx 的日志文件通常存储在 `/var/log/nginx/` 目录下，可以通过查看 `error.log` 和 `access.log` 来排查问题。

### 十、Nginx 模块

Nginx 提供了丰富的模块来扩展其功能，包括第三方模块。常见模块有：
- **HTTP 核心模块**：用于处理 HTTP 请求。
- **Gzip 模块**：用于内容压缩。
- **SSL 模块**：用于处理 SSL/TLS。

可以在编译 Nginx 时添加模块，或者使用一些常用的预编译包。

### 结语
Nginx 是一个功能强大且灵活的服务器软件，适用于各种 Web 应用场景。通过本教程的学习，你应该能够掌握 Nginx 的基本使用，并能根据需求配置反向代理、负载均衡、HTTPS 等功能。