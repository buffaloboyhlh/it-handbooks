# requests 模块

### Python `requests` 模块详解

`requests` 是一个功能强大的 Python 库，用于发送 HTTP 请求。它简化了与 Web 服务的交互，可以轻松处理 HTTP 请求和响应，是 Web 开发中非常常用的库之一。

#### 1. 安装

在使用 `requests` 模块之前，需要确保已安装它：

```bash
pip install requests
```

#### 2. 发送 HTTP 请求

`requests` 模块支持 GET、POST、PUT、DELETE 等常用的 HTTP 方法。

- **GET 请求**: 用于从服务器获取数据。

  ```python
  import requests

  response = requests.get("https://api.example.com/data")
  print(response.status_code)  # 输出响应状态码
  print(response.text)         # 输出响应内容
  ```

- **POST 请求**: 用于向服务器发送数据。

  ```python
  data = {"username": "user", "password": "pass"}
  response = requests.post("https://api.example.com/login", data=data)
  print(response.json())  # 解析并输出 JSON 响应
  ```

- **PUT 请求**: 用于更新服务器上的资源。

  ```python
  data = {"username": "new_user"}
  response = requests.put("https://api.example.com/user/1", data=data)
  ```

- **DELETE 请求**: 用于删除服务器上的资源。

  ```python
  response = requests.delete("https://api.example.com/user/1")
  ```

#### 3. 常见参数

- **`params`**: 用于传递 URL 查询参数。

  ```python
  params = {"search": "python", "page": 2}
  response = requests.get("https://api.example.com/search", params=params)
  ```

- **`headers`**: 用于设置自定义 HTTP 头。

  ```python
  headers = {"Authorization": "Bearer token"}
  response = requests.get("https://api.example.com/protected", headers=headers)
  ```

- **`data`**: 用于发送表单数据。

  ```python
  data = {"field1": "value1", "field2": "value2"}
  response = requests.post("https://api.example.com/form", data=data)
  ```

- **`json`**: 用于发送 JSON 数据。

  ```python
  json_data = {"key": "value"}
  response = requests.post("https://api.example.com/json", json=json_data)
  ```

- **`files`**: 用于上传文件。

  ```python
  files = {"file": open("example.txt", "rb")}
  response = requests.post("https://api.example.com/upload", files=files)
  ```

- **`auth`**: 用于传递身份验证信息。

  ```python
  from requests.auth import HTTPBasicAuth

  response = requests.get("https://api.example.com/basic-auth", auth=HTTPBasicAuth('user', 'pass'))
  ```

#### 4. 处理响应

- **`response.status_code`**: 获取响应状态码（如 200, 404）。

  ```python
  if response.status_code == 200:
      print("请求成功")
  ```

- **`response.text`**: 获取响应的文本内容。

  ```python
  print(response.text)
  ```

- **`response.json()`**: 将响应内容解析为 JSON。

  ```python
  data = response.json()
  ```

- **`response.content`**: 获取二进制内容，通常用于下载文件。

  ```python
  with open("image.png", "wb") as f:
      f.write(response.content)
  ```

- **`response.headers`**: 获取响应头信息。

  ```python
  print(response.headers['Content-Type'])
  ```

#### 5. 处理超时和重试

- **设置超时**: 防止请求无限期地等待响应。

  ```python
  response = requests.get("https://api.example.com/data", timeout=5)  # 5 秒超时
  ```

- **实现重试**: 使用 `requests` 的 `Session` 对象与 `HTTPAdapter` 配合来实现请求重试。

  ```python
  from requests.adapters import HTTPAdapter
  from requests.packages.urllib3.util.retry import Retry

  session = requests.Session()
  retry = Retry(connect=3, backoff_factor=0.5)
  adapter = HTTPAdapter(max_retries=retry)
  session.mount('http://', adapter)
  session.mount('https://', adapter)

  response = session.get("https://api.example.com/data")
  ```

#### 6. 处理 Cookie

- **发送请求时携带 Cookie**:

  ```python
  cookies = {'session_id': '12345'}
  response = requests.get("https://api.example.com/profile", cookies=cookies)
  ```

- **获取响应的 Cookie**:

  ```python
  print(response.cookies['session_id'])
  ```

- **使用 `Session` 对象自动管理 Cookie**:

  ```python
  session = requests.Session()
  session.get("https://api.example.com/login")
  response = session.get("https://api.example.com/profile")
  ```

#### 7. 请求重定向和历史记录

- **检查是否发生了重定向**:

  ```python
  response = requests.get("https://api.example.com/redirect", allow_redirects=False)
  print(response.status_code)  # 如果是 301 或 302 表示重定向
  ```

- **查看请求的重定向历史**:

  ```python
  response = requests.get("https://api.example.com/redirect")
  for resp in response.history:
      print(resp.status_code, resp.url)
  ```

#### 8. SSL 证书验证

- **禁用 SSL 证书验证**: 在请求时忽略 SSL 证书错误（不推荐在生产环境中使用）。

  ```python
  response = requests.get("https://self-signed.badssl.com/", verify=False)
  ```

- **使用自定义的 SSL 证书**:

  ```python
  response = requests.get("https://api.example.com", verify='/path/to/cert.pem')
  ```

### 总结

`requests` 模块提供了一个简单而强大的接口来处理 HTTP 请求，适用于从简单的 Web 请求到复杂的 API 交互。在 Web 开发、数据抓取和 API 集成中，它是一个不可或缺的工具。