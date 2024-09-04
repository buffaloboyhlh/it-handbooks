# re 模块

### Python `re` 模块详解

`re` 模块提供了正则表达式的支持，允许你使用正则表达式来搜索、匹配和操作字符串。正则表达式是一种用于描述字符串模式的强大工具，特别适用于复杂的字符串处理任务。

#### 1. 导入模块

```python
import re
```

#### 2. 基本的正则表达式操作

- **匹配模式 `re.match()`**

`re.match()` 从字符串的起始位置匹配一个模式。

```python
pattern = r'\d+'  # 匹配一个或多个数字
string = "123abc"
match = re.match(pattern, string)

if match:
    print("匹配成功:", match.group())  # 输出 123
else:
    print("匹配失败")
```

- **搜索模式 `re.search()`**

`re.search()` 在整个字符串中搜索第一个匹配的模式。

```python
pattern = r'\d+'  # 匹配一个或多个数字
string = "abc123def"
search = re.search(pattern, string)

if search:
    print("搜索成功:", search.group())  # 输出 123
else:
    print("搜索失败")
```

- **查找所有匹配 `re.findall()`**

`re.findall()` 返回所有非重叠的匹配项列表。

```python
pattern = r'\d+'  # 匹配一个或多个数字
string = "abc123def456"
matches = re.findall(pattern, string)
print(matches)  # 输出 ['123', '456']
```

- **替换匹配项 `re.sub()`**

`re.sub()` 替换所有匹配的模式。

```python
pattern = r'\d+'  # 匹配一个或多个数字
string = "abc123def456"
result = re.sub(pattern, '#', string)
print(result)  # 输出 abc#def#
```

- **拆分字符串 `re.split()`**

`re.split()` 根据匹配模式拆分字符串。

```python
pattern = r'\d+'  # 匹配一个或多个数字
string = "abc123def456ghi"
split_result = re.split(pattern, string)
print(split_result)  # 输出 ['abc', 'def', 'ghi']
```

#### 3. 正则表达式语法

- **普通字符**

普通字符在模式中表示它们字面的含义，例如 `"abc"` 将匹配字符串 `"abc"`。

- **特殊字符**

  - `.`: 匹配任意一个字符（不包括换行符）。
  - `^`: 匹配字符串的开始。
  - `$`: 匹配字符串的结束。
  - `*`: 匹配前面的元素0次或多次。
  - `+`: 匹配前面的元素1次或多次。
  - `?`: 匹配前面的元素0次或1次。
  - `[]`: 匹配方括号内的任意字符，如 `[a-z]` 匹配小写字母。
  - `{m,n}`: 匹配前面的元素至少 m 次，至多 n 次。
  - `|`: 表示或运算。
  - `()`：捕获组，用于提取匹配的子串。

```python
pattern = r'(\d+)\s(\w+)'  # 捕获一个或多个数字，跟着一个空格，然后是一个或多个字母数字字符
string = "123 abc"
match = re.match(pattern, string)

if match:
    print(match.group(1))  # 输出 123
    print(match.group(2))  # 输出 abc
```

#### 4. 编译正则表达式

- **`re.compile()`**

`re.compile()` 用于编译正则表达式模式，提高多次使用同一模式时的效率。

```python
pattern = re.compile(r'\d+')  # 编译正则表达式
match = pattern.match("123abc")

if match:
    print("匹配成功:", match.group())  # 输出 123
```

#### 5. 正则表达式标志（Flags）

在正则表达式中可以使用标志来改变匹配的行为，如忽略大小写、多行匹配等。

- **`re.IGNORECASE` (`re.I`)**: 忽略大小写。

```python
pattern = re.compile(r'abc', re.IGNORECASE)
print(pattern.match("ABC"))  # 匹配成功
```

- **`re.MULTILINE` (`re.M`)**: 多行匹配，影响 `^` 和 `$`。

```python
pattern = re.compile(r'^abc', re.MULTILINE)
print(pattern.findall("abc\ndefabc\nabc"))  # 输出 ['abc', 'abc']
```

- **`re.DOTALL` (`re.S`)**: 使 `.` 匹配包括换行符在内的所有字符。

```python
pattern = re.compile(r'a.b', re.DOTALL)
print(pattern.match("a\nb"))  # 匹配成功
```

#### 6. 高级匹配操作

- **分组和命名组**

使用圆括号 `()` 进行分组，使用 `(?P<name>...)` 进行命名分组。

```python
pattern = r'(?P<area>\d{3})-(?P<number>\d{4})'
match = re.match(pattern, '123-4567')

if match:
    print(match.group('area'))  # 输出 123
    print(match.group('number'))  # 输出 4567
```

- **非捕获组**

使用 `(?:...)` 定义非捕获组。

```python
pattern = r'(?:abc){2}'
match = re.match(pattern, 'abcabc')

if match:
    print(match.group())  # 输出 abcabc
```

#### 7. 使用正则表达式进行字符串验证

正则表达式常用于验证字符串是否符合特定的格式，如电子邮件、电话号码等。

```python
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None

print(is_valid_email("test@example.com"))  # 输出 True
print(is_valid_email("invalid-email"))  # 输出 False
```

### 总结

`re` 模块是 Python 中处理字符串的强大工具，适用于复杂的模式匹配、查找和替换操作。通过掌握正则表达式的语法和 `re` 模块的使用方法，你可以高效地处理各种字符串操作任务。