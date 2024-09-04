# os模块

### Python `os` 模块详解

`os` 模块提供了非常多的与操作系统进行交互的函数。它使得程序可以执行诸如文件处理、目录操作、路径管理等任务，并且可以与系统的环境变量进行交互。

#### 1. 导入模块

```python
import os
```

#### 2. 目录操作

- **获取当前工作目录**

```python
current_directory = os.getcwd()
print(current_directory)
```

- **改变当前工作目录**

```python
os.chdir('/path/to/directory')
```

- **创建目录**

```python
os.mkdir('new_directory')  # 创建单级目录
os.makedirs('new_directory/sub_directory')  # 创建多级目录
```

- **删除目录**

```python
os.rmdir('directory')  # 删除单级空目录
os.removedirs('directory/sub_directory')  # 删除多级空目录
```

- **列出目录内容**

```python
files = os.listdir('.')
print(files)
```

#### 3. 文件操作

- **创建文件**

```python
with open('file.txt', 'w') as f:
    f.write('Hello, World!')
```

- **删除文件**

```python
os.remove('file.txt')
```

- **重命名文件或目录**

```python
os.rename('old_name.txt', 'new_name.txt')
```

- **获取文件或目录的状态**

```python
stat_info = os.stat('file.txt')
print(stat_info)
```

#### 4. 路径操作

- **获取文件路径的绝对路径**

```python
absolute_path = os.path.abspath('file.txt')
print(absolute_path)
```

- **检查路径是否存在**

```python
exists = os.path.exists('file.txt')
print(exists)
```

- **检查是否是文件**

```python
is_file = os.path.isfile('file.txt')
print(is_file)
```

- **检查是否是目录**

```python
is_dir = os.path.isdir('directory')
print(is_dir)
```

- **路径拼接**

```python
full_path = os.path.join('/path', 'to', 'directory')
print(full_path)
```

#### 5. 环境变量

- **获取环境变量**

```python
path = os.getenv('PATH')
print(path)
```

- **设置环境变量**

```python
os.environ['MY_VARIABLE'] = 'my_value'
```

#### 6. 进程管理

- **执行系统命令**

```python
os.system('ls -l')
```

- **获取当前进程 ID**

```python
pid = os.getpid()
print(pid)
```

- **获取父进程 ID**

```python
ppid = os.getppid()
print(ppid)
```

- **创建子进程**

```python
pid = os.fork()
if pid == 0:
    print('This is the child process.')
else:
    print('This is the parent process.')
```

#### 7. 异常处理

`os` 模块的很多操作可能会引发异常，特别是文件和目录操作，因此在使用时应进行适当的异常处理。

```python
try:
    os.remove('non_existent_file.txt')
except FileNotFoundError:
    print("File not found.")
```

#### 8. 常用的其他功能

- **获取操作系统类型**

```python
os_type = os.name
print(os_type)  # 输出 'posix', 'nt', 'os2', 'ce', 'java', 'riscos' 之一
```

- **获取系统的详细信息**

```python
system_info = os.uname()  # 仅适用于 Unix 系统
print(system_info)
```

- **列出环境变量**

```python
env_vars = os.environ
print(env_vars)
```

### 总结

`os` 模块是一个非常强大的模块，可以帮助开发者进行各种与操作系统交互的任务。在使用时，了解各个函数的作用和适用场景将大大提高开发效率。