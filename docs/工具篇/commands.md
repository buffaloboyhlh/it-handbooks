# Linux 常用命令

在服务器管理中，Linux 系统常用命令的熟练掌握是至关重要的。为了全面涵盖各个方面的操作，以下内容将以更加详细的方式介绍 Linux 服务器中的常用命令，包括文件操作、系统监控、进程管理、网络配置、安全管理等。

---

## 1. 系统信息与性能监控

### 1.1 `uname` - 查看操作系统信息
```bash
uname [选项]
```
**常用选项**：
- `uname -a`：显示所有系统信息（包括内核版本、主机名、硬件架构等）。
- `uname -r`：显示当前内核版本。

### 1.2 `uptime` - 查看系统运行时间和负载
```bash
uptime
```
显示系统的当前时间、运行时间、当前登录的用户数，以及最近 1、5、15 分钟的系统平均负载。

### 1.3 `top` - 实时查看系统资源使用情况
```bash
top
```
**主要信息**：

- `PID`：进程 ID。
- `USER`：运行进程的用户。
- `PR`：优先级。
- `%CPU`：CPU 使用率。
- `%MEM`：内存使用率。

按 `q` 退出，`h` 查看帮助，`k` 杀掉某个进程。

### 1.4 `htop` - 更友好的实时监控工具（需安装）
```bash
htop
```
`htop` 比 `top` 更直观，支持上下滚动和更丰富的进程信息展示。它提供图形化 CPU 和内存利用率。

### 1.5 `free` - 查看内存使用情况
```bash
free [选项]
```
- `free -h`：以人类可读的格式显示内存使用情况，显示已用、可用、缓存内存等信息。

### 1.6 `df` - 查看磁盘空间使用情况
```bash
df [选项]
```
- `df -h`：以人类可读的格式显示每个分区的使用情况（单位为 GB/MB）。
- `df -T`：显示每个分区的文件系统类型。

### 1.7 `du` - 查看目录或文件的大小
```bash
du [选项] [路径]
```
- `du -sh /var/log`：显示 `/var/log` 目录的总大小。
- `du -h --max-depth=1 /`：显示根目录下每个一级子目录的大小。

### 1.8 `iostat` - 显示磁盘 I/O 统计（需安装 `sysstat`）
```bash
iostat [选项]
```
- `iostat`：显示每个设备的 CPU 使用情况和 I/O 统计。
- `iostat -x`：显示详细的设备 I/O 统计信息。

### 1.9 `vmstat` - 系统性能监控
```bash
vmstat [时间间隔] [次数]
```
- `vmstat 1 5`：每秒刷新一次，共输出 5 次系统的内存、CPU、I/O 等情况。

### 1.10 `sar` - 系统性能历史记录（需安装 `sysstat`）
```bash
sar [选项] [时间间隔] [次数]
```
- `sar -u 1 3`：每秒显示一次 CPU 使用情况，显示 3 次。
- `sar -r 1 3`：每秒显示一次内存使用情况，显示 3 次。

---

## 2. 文件与目录操作

### 2.1 `ls` - 列出目录内容
```bash
ls [选项] [目录]
```
**常用选项**：
- `ls`：列出当前目录的内容。
- `ls -l`：以详细列表模式显示文件信息（包括权限、所有者、大小、修改时间等）。
- `ls -a`：显示所有文件，包括隐藏文件（以 `.` 开头）。
- `ls -h`：以人类可读的方式显示文件大小。

### 2.2 `cd` - 切换目录
```bash
cd [目录]
```
- `cd /path/to/directory`：切换到指定目录。
- `cd ~`：切换到用户的主目录。
- `cd -`：切换到上一次所在的目录。

### 2.3 `pwd` - 显示当前所在目录
```bash
pwd
```
显示当前工作目录的绝对路径。

### 2.4 `cp` - 复制文件或目录
```bash
cp [选项] 源文件 目标路径
```
**常用选项**：
- `cp file1 /path/to/destination/`：复制文件 `file1` 到目标路径。
- `cp -r dir1 /path/to/destination/`：递归复制目录 `dir1` 及其内容。

### 2.5 `mv` - 移动或重命名文件
```bash
mv [选项] 源文件 目标路径
```
**常用操作**：
- `mv file1 /path/to/destination/`：将文件 `file1` 移动到目标路径。
- `mv file1 file2`：将 `file1` 重命名为 `file2`。

### 2.6 `rm` - 删除文件或目录
```bash
rm [选项] 文件名
```
**常用选项**：
- `rm file1`：删除文件 `file1`。
- `rm -r dir1`：递归删除目录 `dir1` 及其内容。
- `rm -i`：删除前提示确认。

### 2.7 `mkdir` - 创建新目录
```bash
mkdir [选项] 目录名
```
- `mkdir newdir`：创建目录 `newdir`。
- `mkdir -p /path/to/newdir`：递归创建目录，包括必要的父目录。

### 2.8 `touch` - 创建空文件或更新文件时间戳
```bash
touch 文件名
```
- `touch newfile`：创建一个新的空文件 `newfile`。
- `touch -t 202309251200 file`：将 `file` 的时间戳设置为 2023年09月25日12时00分。

### 2.9 `find` - 查找文件或目录
```bash
find [路径] [选项] [搜索条件]
```
**常用示例**：
- `find /home -name "*.log"`：在 `/home` 目录下查找所有 `.log` 文件。
- `find / -type d -name "test"`：查找名为 `test` 的目录。
- `find /var/log -mtime -7`：查找过去 7 天内修改过的文件。

### 2.10 `locate` - 快速查找文件（需安装）
```bash
locate 文件名
```
- `locate filename`：快速查找文件 `filename`，其基于预先生成的数据库。

---

## 3. 权限管理

### 3.1 `chmod` - 修改文件权限
```bash
chmod [权限] 文件名
```
- `chmod 755 script.sh`：为文件 `script.sh` 设置 `rwxr-xr-x` 权限。
- `chmod +x script.sh`：为文件 `script.sh` 添加执行权限。

### 3.2 `chown` - 修改文件所有者
```bash
chown [选项] 所有者:用户组 文件名
```
- `chown user1:group1 file1`：将 `file1` 的所有者改为 `user1`，所属组改为 `group1`。
- `chown -R user1:group1 /path/to/dir`：递归修改目录及其文件的所有者和所属组。

### 3.3 `umask` - 设置默认权限掩码
```bash
umask [权限掩码]
```
- `umask 022`：设置文件默认权限掩码，使新文件默认权限为 `644`，目录为 `755`。

---

## 4. 进程管理

### 4.1 `ps` - 查看当前进程
```bash
ps [选项]
```
- `ps aux`：显示所有进程及其详细信息，包括进程号、CPU 使用率、内存使用率等。
- `ps -ef`：显示所有进程的完整格式。

### 4.2 `kill` - 终止进程
```bash
kill [选项] 进程号
```
- `kill 1234`：终止进程号为 `1234` 的进程。
- `kill -9 1234`：强制终止进程号为 `1234` 的进程。

### 4.3 `pkill` - 根据进程名终止进程
```bash
pkill [选项] 进程名
```
- `pkill nginx`：终止所有名为 `nginx` 的进程。

### 4.4 `systemctl` - 管理系统服务
```bash
systemctl [动作] [服务名]
```
**常用命令**：
- `systemctl start nginx`：启动 `nginx` 服务。
- `systemctl stop nginx`：停止 `nginx` 服务。
- `systemctl restart nginx`：重启 `nginx` 服务。
- `systemctl status nginx`：查看 `nginx` 服务状态。
- `systemctl enable nginx`：设置 `nginx` 服务开机自动启动。
- `systemctl disable nginx`：取消 `nginx` 服务开机启动。

### 4.5 `service` - 管理服务（旧版命令）
```bash
service [服务名] [动作]
```
**常用命令**：
- `service nginx start`：启动 `nginx` 服务。
- `service nginx stop`：停止 `nginx` 服务。

---

## 5. 网络管理

### 5.1 `ip` - 查看和配置网络接口
```bash
ip [选项] 命令
```
**常用命令**：
- `ip addr`：查看网络接口的 IP 地址。
- `ip link`：显示网络接口的状态。
- `ip route`：查看路由信息。

### 5.2 `ping` - 测试网络连通性
```bash
ping [选项] 目标
```
- `ping google.com`：测试到 `google.com` 的网络连通性。

### 5.3 `netstat` - 查看网络连接（`ss` 是其替代品）
```bash
netstat [选项]
```
**常用选项**：
- `netstat -tuln`：显示正在监听的 TCP/UDP 端口。
- `netstat -anp`：显示所有网络连接及其关联进程。

### 5.4 `ss` - 替代 `netstat` 的现代工具
```bash
ss [选项]
```
- `ss -tuln`：显示正在监听的 TCP/UDP 端口。
- `ss -anp`：显示所有网络连接及其关联进程。

### 5.5 `curl` - 命令行网络请求工具
```bash
curl [选项] URL
```
- `curl http://example.com`：请求 `example.com` 并返回响应内容。
- `curl -I http://example.com`：仅返回响应头信息。

### 5.6 `wget` - 下载文件
```bash
wget [选项] URL
```
- `wget http://example.com/file.zip`：下载文件 `file.zip`。

---

## 6. 软件包管理

### 6.1 Debian/Ubuntu 系统

- `apt update`：更新软件包索引。
- `apt upgrade`：升级所有已安装的软件包。
- `apt install package_name`：安装软件包。
- `apt remove package_name`：卸载软件包。

### 6.2 CentOS/RHEL 系统

- `yum update`：更新软件包和系统。
- `yum install package_name`：安装指定软件包。
- `yum remove package_name`：卸载指定软件包。

---

以上是 Linux 服务器管理中常用命令的详细介绍，涵盖了系统监控、文件管理、权限控制、进程管理、网络配置和软件管理等方面。这些命令的熟练运用将帮助你高效管理和维护服务器。