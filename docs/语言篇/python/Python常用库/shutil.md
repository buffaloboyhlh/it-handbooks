# shutil 模块 

### Python `shutil` 模块详解

`shutil` 是 Python 的标准库模块，提供了高层次的文件操作功能。`shutil` 模块支持文件复制、移动、删除、压缩和解压缩等操作，这使得它非常适合于处理文件和目录的常见任务。

#### 1. 复制文件和目录

- **`shutil.copy()`**: 复制文件内容和权限（但不包括元数据，如创建时间等）。

  ```python
  import shutil

  shutil.copy("source_file.txt", "destination_file.txt")
  ```

- **`shutil.copy2()`**: 复制文件内容、权限和元数据。

  ```python
  shutil.copy2("source_file.txt", "destination_file.txt")
  ```

- **`shutil.copytree()`**: 递归地复制整个目录树。

  ```python
  shutil.copytree("source_directory", "destination_directory")
  ```

#### 2. 移动文件和目录

- **`shutil.move()`**: 移动文件或目录。可以在不同的文件系统之间移动文件。

  ```python
  shutil.move("source_file_or_directory", "destination_file_or_directory")
  ```

#### 3. 删除文件和目录

- **`shutil.rmtree()`**: 递归删除目录树及其内容。

  ```python
  shutil.rmtree("directory_to_delete")
  ```

- **`shutil.unlink()`**: 删除单个文件，实际上是 `os.unlink()` 的别名。

  ```python
  import os
  os.unlink("file_to_delete.txt")
  ```

- **`shutil.remove()`**: 删除单个文件，可以删除文件的符号链接。

  ```python
  os.remove("file_to_delete.txt")
  ```

#### 4. 压缩和解压缩

- **`shutil.make_archive()`**: 压缩文件或目录为一个归档文件。支持格式包括 `zip`、`tar`、`gztar`、`bztar` 等。

  ```python
  shutil.make_archive("archive_name", "zip", "directory_to_compress")
  ```

- **`shutil.unpack_archive()`**: 解压缩归档文件。

  ```python
  shutil.unpack_archive("archive_name.zip", "output_directory")
  ```

#### 5. 计算文件夹的总大小

- **`shutil.disk_usage()`**: 获取磁盘使用情况。

  ```python
  usage = shutil.disk_usage("/")
  print(f"Total: {usage.total}, Used: {usage.used}, Free: {usage.free}")
  ```

- **计算目录大小**: 你可以使用 `shutil` 和 `os` 模块来计算整个目录的大小。

  ```python
  def get_directory_size(directory):
      total_size = 0
      for dirpath, dirnames, filenames in os.walk(directory):
          for f in filenames:
              fp = os.path.join(dirpath, f)
              total_size += os.path.getsize(fp)
      return total_size

  print(get_directory_size("directory_to_calculate"))
  ```

#### 6. 复制文件对象

- **`shutil.copyfileobj()`**: 将文件对象的内容复制到另一个文件对象。适用于需要对文件内容进行部分读取或写入的情况。

  ```python
  with open("source_file.txt", "rb") as src, open("destination_file.txt", "wb") as dst:
      shutil.copyfileobj(src, dst)
  ```

#### 7. 文件权限和属性

- **`shutil.chown()`**: 改变文件的所有者和组。

  ```python
  shutil.chown("file.txt", user="username", group="groupname")
  ```

- **`shutil.copymode()`**: 仅复制文件权限。

  ```python
  shutil.copymode("source_file.txt", "destination_file.txt")
  ```

- **`shutil.copystat()`**: 复制文件的状态信息（权限、最后访问时间、修改时间等）。

  ```python
  shutil.copystat("source_file.txt", "destination_file.txt")
  ```

### 总结

`shutil` 模块是处理文件和目录操作的强大工具，提供了高层次的接口，能够简化日常的文件操作任务。它特别适用于需要处理大量文件操作的脚本或应用程序，例如备份、文件管理工具等。