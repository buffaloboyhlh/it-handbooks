# pathlib 模块

### Python `pathlib` 模块详解

`pathlib` 是 Python 3.4 引入的一个标准库模块，它提供了面向对象的文件系统路径操作方法。相比于传统的 `os.path` 模块，`pathlib` 更加直观和强大，适用于文件和目录的创建、读取、修改、删除等操作。

#### 1. 基本概念

`pathlib` 模块引入了 `Path` 类，该类表示文件系统路径。根据不同的操作系统，`pathlib` 会自动处理路径分隔符问题，使代码更具跨平台性。

```python
from pathlib import Path

# 创建一个 Path 对象
p = Path('/usr/bin')
```

#### 2. 创建路径对象

- **绝对路径**: 直接传入绝对路径。

  ```python
  p = Path('/usr/bin')
  ```

- **相对路径**: 传入相对路径，路径将相对于当前工作目录。

  ```python
  p = Path('my_folder/my_file.txt')
  ```

- **当前目录**: 使用 `Path.cwd()` 获取当前工作目录。

  ```python
  current_dir = Path.cwd()
  ```

- **家目录**: 使用 `Path.home()` 获取当前用户的家目录。

  ```python
  home_dir = Path.home()
  ```

#### 3. 路径操作

- **获取路径的各部分**:

  ```python
  p = Path('/usr/bin/python3')

  print(p.name)        # 文件名: python3
  print(p.parent)      # 父目录: /usr/bin
  print(p.stem)        # 文件名不带扩展名: python3
  print(p.suffix)      # 文件扩展名: .3
  print(p.parts)       # 路径的所有部分: ('/', 'usr', 'bin', 'python3')
  ```

- **组合路径**: 使用 `/` 操作符组合路径。

  ```python
  p = Path('/usr') / 'bin' / 'python3'
  print(p)  # 输出: /usr/bin/python3
  ```

- **检查路径**: 使用各种方法检查路径的状态。

  ```python
  p = Path('/usr/bin/python3')

  print(p.exists())      # 检查路径是否存在
  print(p.is_file())     # 检查是否为文件
  print(p.is_dir())      # 检查是否为目录
  print(p.is_absolute()) # 检查是否为绝对路径
  ```

#### 4. 文件操作

- **读取文件内容**: 使用 `read_text()` 和 `read_bytes()` 读取文件内容。

  ```python
  p = Path('example.txt')

  content = p.read_text(encoding='utf-8')  # 读取文本文件
  binary_data = p.read_bytes()             # 读取二进制文件
  ```

- **写入文件内容**: 使用 `write_text()` 和 `write_bytes()` 写入文件。

  ```python
  p = Path('example.txt')

  p.write_text('Hello, World!', encoding='utf-8')  # 写入文本文件
  p.write_bytes(b'binary data')                    # 写入二进制文件
  ```

- **创建文件和目录**: 使用 `touch()` 创建文件，`mkdir()` 创建目录。

  ```python
  p = Path('new_file.txt')
  p.touch()  # 创建一个空文件

  p = Path('new_folder')
  p.mkdir(parents=True, exist_ok=True)  # 创建目录，自动创建父目录
  ```

- **删除文件和目录**: 使用 `unlink()` 删除文件，`rmdir()` 删除空目录。

  ```python
  p = Path('new_file.txt')
  p.unlink()  # 删除文件

  p = Path('new_folder')
  p.rmdir()   # 删除空目录
  ```

#### 5. 遍历目录

- **`iterdir()`**: 用于遍历目录中的内容。

  ```python
  p = Path('.')
  for entry in p.iterdir():
      print(entry.name)  # 打印当前目录下的所有文件和目录
  ```

- **`glob()` 和 `rglob()`**: 支持通配符匹配，`rglob()` 可以递归匹配。

  ```python
  p = Path('.')
  for txt_file in p.glob('*.txt'):
      print(txt_file.name)  # 匹配当前目录下的所有 .txt 文件

  for py_file in p.rglob('*.py'):
      print(py_file.name)  # 递归匹配当前目录及子目录下的所有 .py 文件
  ```

#### 6. 路径比较与相对路径

- **比较路径**: 可以直接比较 `Path` 对象。

  ```python
  p1 = Path('/usr/bin/python3')
  p2 = Path('/usr/bin/python3')

  print(p1 == p2)  # 输出: True
  ```

- **获取相对路径**: 使用 `relative_to()` 方法。

  ```python
  p1 = Path('/usr/bin/python3')
  p2 = Path('/usr')

  print(p1.relative_to(p2))  # 输出: bin/python3
  ```

- **获取路径之间的相对路径**:

  ```python
  p1 = Path('/usr/bin/python3')
  p2 = Path('/usr/local/bin')

  print(p1.relative_to(p2))  # 输出: ../local/bin
  ```

### 总结

`pathlib` 模块为文件路径操作提供了更加直观和强大的工具。通过面向对象的方法，它使得路径操作更加清晰和易于理解。对于涉及文件和目录的任务，`pathlib` 是一个非常推荐的模块。