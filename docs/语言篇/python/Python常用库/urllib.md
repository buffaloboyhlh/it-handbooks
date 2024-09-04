# urllib 模块

`urllib` 模块是 Python 标准库中用于处理 URL 和 HTTP 请求的模块。它提供了多个子模块，包括 `urllib.request`、`urllib.parse`、`urllib.error` 和 `urllib.robotparser`，用于处理不同的 URL 和网络请求操作。

### 1. `urllib.request` 模块

`urllib.request` 模块用于打开和读取 URL 的内容，可以发送 HTTP 请求并处理响应。

#### 1.1 `urlopen()`
- **功能**：打开一个 URL，返回一个类似文件的对象。
- **参数**：
  - `url`：要打开的 URL。
  - `data`：可选，用于 POST 请求的数据。
  - `timeout`：可选，请求的超时时间（以秒为单位）。
- **示例**：
  ```python
  from urllib import request
  
  response = request.urlopen('http://www.example.com')
  html = response.read().decode('utf-8')
  print(html)
  ```

#### 1.2 `Request`
- **功能**：用于创建一个请求对象，可以附加 headers、data 等信息。
- **示例**：
  ```python
  from urllib import request
  
  url = 'http://www.example.com'
  req = request.Request(url)
  req.add_header('User-Agent', 'Mozilla/5.0')
  response = request.urlopen(req)
  print(response.read().decode('utf-8'))
  ```

#### 1.3 `urlretrieve()`
- **功能**：将 URL 指向的资源下载到本地文件。
- **示例**：
  ```python
  from urllib import request
  
  url = 'http://www.example.com/somefile.zip'
  request.urlretrieve(url, 'localfile.zip')
  ```

### 2. `urllib.parse` 模块

`urllib.parse` 模块用于解析 URL 字符串和处理 URL 的各个部分。

#### 2.1 `urlparse()`
- **功能**：解析 URL 并返回 `ParseResult` 对象，包含 URL 的各个部分（scheme, netloc, path, params, query, fragment）。
- **示例**：
  ```python
  from urllib import parse
  
  url = 'http://www.example.com/path;params?query=arg#fragment'
  parsed_url = parse.urlparse(url)
  print(parsed_url.scheme)   # 输出: 'http'
  print(parsed_url.netloc)   # 输出: 'www.example.com'
  print(parsed_url.path)     # 输出: '/path'
  ```

#### 2.2 `urlunparse()`
- **功能**：将 URL 的各部分组合成一个完整的 URL 字符串。
- **示例**：
  ```python
  from urllib import parse
  
  url_components = ('http', 'www.example.com', '/path', 'params', 'query=arg', 'fragment')
  url = parse.urlunparse(url_components)
  print(url)  # 输出: 'http://www.example.com/path;params?query=arg#fragment'
  ```

#### 2.3 `urlencode()`
- **功能**：将字典或元组列表编码为 URL 查询参数字符串。
- **示例**：
  ```python
  from urllib import parse
  
  params = {'name': 'Alice', 'age': 25}
  query_string = parse.urlencode(params)
  print(query_string)  # 输出: 'name=Alice&age=25'
  ```

### 3. `urllib.error` 模块

`urllib.error` 模块包含处理 HTTP 请求错误的异常类。

#### 3.1 `HTTPError`
- **功能**：当 HTTP 请求失败并返回错误代码（如404, 403等）时引发此异常。
- **示例**：
  ```python
  from urllib import request, error
  
  try:
      response = request.urlopen('http://www.example.com/404')
  except error.HTTPError as e:
      print(f'HTTP Error: {e.code}')
  ```

#### 3.2 `URLError`
- **功能**：当无法处理请求（如无法连接到服务器）时引发此异常。
- **示例**：
  ```python
  from urllib import request, error
  
  try:
      response = request.urlopen('http://nonexistent.example.com')
  except error.URLError as e:
      print(f'URL Error: {e.reason}')
  ```

### 4. `urllib.robotparser` 模块

`urllib.robotparser` 模块用于解析 robots.txt 文件，该文件告诉搜索引擎爬虫哪些页面可以或不能抓取。

#### 4.1 `RobotFileParser`
- **功能**：用于解析和处理 robots.txt 文件。
- **示例**：
  ```python
  from urllib import robotparser
  
  rp = robotparser.RobotFileParser()
  rp.set_url('http://www.example.com/robots.txt')
  rp.read()
  print(rp.can_fetch('*', 'http://www.example.com/path'))
  ```

### 5. 示例应用

#### 5.1 发送 GET 请求并处理响应
```python
from urllib import request

url = 'http://www.example.com'
response = request.urlopen(url)
content = response.read().decode('utf-8')
print(content)
```

#### 5.2 发送 POST 请求
```python
from urllib import request, parse

url = 'http://www.example.com/login'
data = parse.urlencode({'username': 'user', 'password': 'pass'}).encode()
req = request.Request(url, data=data)
response = request.urlopen(req)
print(response.read().decode('utf-8'))
```

#### 5.3 处理 HTTP 错误
```python
from urllib import request, error

try:
    response = request.urlopen('http://www.example.com/404')
except error.HTTPError as e:
    print(f'HTTP Error: {e.code}')
except error.URLError as e:
    print(f'URL Error: {e.reason}')
else:
    print(response.read().decode('utf-8'))
```

### 6. 注意事项

- `urllib` 模块适用于简单的 HTTP 请求操作。如果需要处理更复杂的 HTTP 请求（如需要处理Cookies、会话、文件上传等），建议使用 `requests` 库，它提供了更高级别的 API。

- 在处理敏感信息（如用户名和密码）时，要注意数据的加密和传输安全，使用 HTTPS 代替 HTTP。

`urllib` 模块为 Python 提供了强大的 URL 和 HTTP 请求处理功能，适用于各种网络编程场景。