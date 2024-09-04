# pytest 模块

### Python `pytest` 模块详解

`pytest` 是一个强大的 Python 测试框架，广泛用于编写简单的单元测试到复杂的功能测试。它具有易于使用、支持参数化测试、强大的插件生态系统等特点。

#### 1. 安装

首先，使用 `pip` 安装 `pytest`：

```bash
pip install pytest
```

安装完成后，可以通过在项目根目录运行 `pytest` 命令来执行测试。

#### 2. 编写测试

`pytest` 自动识别以 `test_` 开头或结尾的函数和文件。

- **基本测试函数**:

  ```python
  # test_example.py
  def test_addition():
      assert 1 + 1 == 2

  def test_subtraction():
      assert 2 - 1 == 1
  ```

- **执行测试**:

  ```bash
  pytest
  ```

  这将自动发现并运行所有符合命名规范的测试函数。

#### 3. 使用 `assert` 语句

`pytest` 使用 Python 的 `assert` 语句进行测试，当断言失败时，`pytest` 会输出详细的错误信息。

```python
def test_multiplication():
    result = 2 * 3
    assert result == 6
```

#### 4. 参数化测试

参数化测试允许你为相同的测试函数提供不同的数据集，从而在不同的输入下测试相同的代码逻辑。

```python
import pytest

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (2, 3, 5),
    (4, 5, 9),
])
def test_addition(a, b, expected):
    assert a + b == expected
```

#### 5. 跳过测试和预期失败

- **跳过测试**:

  ```python
  import pytest

  @pytest.mark.skip(reason="暂时跳过")
  def test_skipped():
      assert 1 + 1 == 2
  ```

- **标记预期失败**:

  ```python
  @pytest.mark.xfail(reason="此功能尚未实现")
  def test_expected_failure():
      assert 1 + 1 == 3
  ```

#### 6. 组织测试

`pytest` 通过目录和模块自动发现测试。可以通过以下方式组织测试：

- **测试文件**: 测试文件通常以 `test_` 开头或 `_test` 结尾。
- **测试类**: 可以在测试文件中定义测试类，类名以 `Test` 开头。类中的方法必须以 `test_` 开头。

  ```python
  class TestMathOperations:
      def test_addition(self):
          assert 1 + 1 == 2

      def test_subtraction(self):
          assert 2 - 1 == 1
  ```

#### 7. Fixtures（固定装置）

Fixtures 是用于在测试函数执行之前或之后提供特定状态的代码。`pytest` 使用 `@pytest.fixture` 装饰器定义 fixtures。

- **简单 Fixture**:

  ```python
  import pytest

  @pytest.fixture
  def sample_data():
      return {"key": "value"}

  def test_sample_data(sample_data):
      assert sample_data["key"] == "value"
  ```

- **Fixture 作用范围**:

  Fixture 可以通过 `scope` 参数指定作用范围，如 `function`（默认）、`module`、`class` 或 `session`。

  ```python
  @pytest.fixture(scope="module")
  def setup_module():
      # 模块级别的设置代码
      return {"setup": "module"}
  ```

#### 8. 断言重写和调试

`pytest` 提供了强大的断言重写功能，使得断言失败时输出详细的信息，帮助快速定位问题。

- **捕获异常**:

  ```python
  import pytest

  def test_zero_division():
      with pytest.raises(ZeroDivisionError):
          1 / 0
  ```

- **启用调试**:

  可以在测试运行时启用调试器，例如 `pdb`：

  ```bash
  pytest --pdb
  ```

#### 9. 使用命令行选项

`pytest` 提供了多种命令行选项来控制测试运行：

- **指定测试文件**:

  ```bash
  pytest test_example.py
  ```

- **运行特定测试**:

  ```bash
  pytest test_example.py::test_addition
  ```

- **显示详细信息**:

  ```bash
  pytest -v
  ```

- **生成 HTML 报告**（需要安装 `pytest-html` 插件）:

  ```bash
  pytest --html=report.html
  ```

#### 10. 扩展和插件

`pytest` 拥有丰富的插件生态系统，可以通过插件扩展其功能。例如：

- **`pytest-cov`**: 用于生成测试覆盖率报告。
- **`pytest-xdist`**: 支持并行测试执行。
- **`pytest-mock`**: 用于简化测试中的 mocking 操作。

安装插件：

```bash
pip install pytest-cov pytest-xdist pytest-mock
```

### 总结

`pytest` 是一个功能强大且灵活的测试框架，适用于从简单到复杂的测试场景。它的易用性、丰富的插件生态系统以及强大的功能，使得它成为 Python 开发者最受欢迎的测试工具之一。