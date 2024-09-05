# C++ 基础教程 

C++ 是一种功能强大的编程语言，广泛用于系统软件、游戏开发、嵌入式系统以及高性能应用程序的开发。以下是一个详细的 C++ 基础教程：

### 1. C++ 简介
C++ 是在 C 语言的基础上发展起来的，它不仅继承了 C 语言的所有优点，还增加了面向对象编程的特性。C++ 既可以进行过程化编程，也可以进行面向对象编程，具有很强的灵活性。

### 2. 环境配置
在学习 C++ 之前，首先需要搭建一个编译和运行 C++ 程序的环境。以下是几个常用的编译器和 IDE：
- **GCC（GNU Compiler Collection）**：适用于 Linux 和 MacOS。
- **MinGW**：适用于 Windows，可以与 GCC 一起使用。
- **Visual Studio**：微软提供的 IDE，支持 Windows 和 MacOS。
- **CLion**：JetBrains 提供的跨平台 IDE。

### 3. C++ 程序结构
一个简单的 C++ 程序如下：

```cpp
#include <iostream> // 包含输入输出流库

int main() {
    std::cout << "Hello, World!" << std::endl; // 输出 Hello, World!
    return 0; // 返回 0 表示程序成功结束
}
```

#### 3.1 头文件
`#include <iostream>` 是一个预处理指令，用于包含标准输入输出流库，`<iostream>` 是头文件的名称。

#### 3.2 main() 函数
`int main()` 是程序的入口函数，程序从这里开始执行。

#### 3.3 输出语句
`std::cout` 是标准输出流对象，用于将数据输出到控制台。`<<` 是输出运算符，`std::endl` 用于插入一个换行符。

#### 3.4 返回值
`return 0;` 表示程序成功结束。

### 4. 数据类型
C++ 提供了多种基本数据类型，包括：
- **整数类型**：`int`、`short`、`long`、`long long`
- **浮点类型**：`float`、`double`、`long double`
- **字符类型**：`char`
- **布尔类型**：`bool`

### 5. 变量和常量
#### 5.1 变量
变量用于存储数据，必须先声明后使用。例如：

```cpp
int a = 10; // 声明并初始化一个整数变量 a
```

#### 5.2 常量
常量的值在程序执行期间不能改变，使用 `const` 关键字声明。例如：

```cpp
const int b = 20; // 声明并初始化一个常量 b
```

### 6. 运算符
C++ 提供了多种运算符，包括：
- **算术运算符**：`+`、`-`、`*`、`/`、`%`
- **关系运算符**：`==`、`!=`、`>`、`<`、`>=`、`<=`
- **逻辑运算符**：`&&`、`||`、`!`
- **赋值运算符**：`=`、`+=`、`-=`、`*=`、`/=`、`%=`
- **自增和自减运算符**：`++`、`--`

### 7. 控制语句
C++ 提供了丰富的控制语句，用于控制程序的执行流程：
- **条件语句**：`if`、`else if`、`else`
- **循环语句**：`for`、`while`、`do...while`
- **跳转语句**：`break`、`continue`、`return`

#### 7.1 条件语句示例
```cpp
int x = 10;
if (x > 0) {
    std::cout << "x 是正数" << std::endl;
} else {
    std::cout << "x 不是正数" << std::endl;
}
```

#### 7.2 循环语句示例
```cpp
for (int i = 0; i < 5; i++) {
    std::cout << "i = " << i << std::endl;
}
```

### 8. 函数
函数是将一组相关的代码封装在一起，便于重复使用和维护的工具。

#### 8.1 函数定义
```cpp
int add(int x, int y) {
    return x + y;
}
```

#### 8.2 函数调用
```cpp
int result = add(3, 4); // 调用 add 函数
```

### 9. 数组和字符串
#### 9.1 数组
数组用于存储相同类型的数据，例如：
```cpp
int arr[5] = {1, 2, 3, 4, 5};
```

#### 9.2 字符串
C++ 中的字符串可以使用字符数组或 `std::string` 对象。例如：
```cpp
char str[] = "Hello";
std::string str2 = "World";
```

### 10. 指针
指针是存储另一个变量的内存地址的变量。例如：
```cpp
int a = 10;
int *p = &a; // p 是指向 a 的指针
```

### 11. 类和对象
C++ 是面向对象编程语言，类和对象是其核心概念。

#### 11.1 类的定义
```cpp
class Dog {
public:
    void bark() {
        std::cout << "Woof!" << std::endl;
    }
};
```

#### 11.2 创建对象
```cpp
Dog myDog;
myDog.bark(); // 调用对象的成员函数
```

### 12. 继承
继承允许一个类从另一个类继承属性和方法。例如：
```cpp
class Animal {
public:
    void eat() {
        std::cout << "Eating..." << std::endl;
    }
};

class Dog : public Animal {
public:
    void bark() {
        std::cout << "Woof!" << std::endl;
    }
};
```

### 13. 多态
多态允许通过基类指针或引用来调用派生类的函数。例如：
```cpp
class Animal {
public:
    virtual void sound() {
        std::cout << "Some sound" << std::endl;
    }
};

class Dog : public Animal {
public:
    void sound() override {
        std::cout << "Woof!" << std::endl;
    }
};
```

### 14. 模板
模板使得函数和类可以处理不同的数据类型。例如：
```cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}
```

### 15. 异常处理
C++ 使用 `try...catch` 块来处理异常。例如：
```cpp
try {
    int result = 10 / 0; // 可能会抛出异常
} catch (const std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
}
```

以下是 C++ 进阶教程中的更多内容，涵盖了更复杂的主题，如动态内存管理、标准模板库（STL）、高级类设计、智能指针、多线程编程和文件操作等。

### 17. 动态内存管理
在 C++ 中，可以使用动态内存管理在程序运行时分配内存。这在处理无法预先确定大小的数据结构时非常有用。

#### 17.1 `new` 和 `delete`
- **`new`** 运算符用于在堆上动态分配内存。例如：
  ```cpp
  int* p = new int; // 分配一个整数大小的内存
  *p = 10;
  ```
- **`delete`** 运算符用于释放使用 `new` 分配的内存。例如：
  ```cpp
  delete p; // 释放内存
  ```

#### 17.2 动态数组
使用 `new` 可以创建动态数组。例如：
```cpp
int* arr = new int[5]; // 分配一个大小为5的整数数组
arr[0] = 1;
delete[] arr; // 释放动态数组
```

### 18. 标准模板库（STL）
STL 是 C++ 中的一个强大工具包，包含了多种数据结构和算法。

#### 18.1 容器
STL 提供了多种容器类，用于存储和管理数据。常见的容器有：
- **`vector`**：动态数组，支持随机访问。
  ```cpp
  std::vector<int> vec = {1, 2, 3, 4, 5};
  vec.push_back(6);
  ```
- **`list`**：双向链表，支持快速插入和删除。
  ```cpp
  std::list<int> lst = {1, 2, 3};
  lst.push_back(4);
  ```
- **`map`**：关联容器，存储键值对。
  ```cpp
  std::map<std::string, int> m;
  m["apple"] = 5;
  ```

#### 18.2 迭代器
迭代器是遍历容器中元素的对象。例如：
```cpp
std::vector<int> vec = {1, 2, 3, 4};
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";
}
```

#### 18.3 算法
STL 提供了一系列的算法，如排序、查找、修改等。例如：
```cpp
std::sort(vec.begin(), vec.end()); // 对 vec 进行排序
```

### 19. 高级类设计
C++ 提供了多种高级类设计方法，增强代码的可维护性和可重用性。

#### 19.1 构造函数和析构函数
- **构造函数**：在对象创建时自动调用，用于初始化对象。例如：
  ```cpp
  class MyClass {
  public:
      MyClass() { std::cout << "Constructor called" << std::endl; }
  };
  ```
- **析构函数**：在对象销毁时自动调用，用于清理资源。例如：
  ```cpp
  class MyClass {
  public:
      ~MyClass() { std::cout << "Destructor called" << std::endl; }
  };
  ```

#### 19.2 拷贝构造函数和赋值运算符
用于处理对象的拷贝和赋值操作。例如：
```cpp
class MyClass {
public:
    MyClass(const MyClass &other) { /* 自定义拷贝逻辑 */ }
    MyClass& operator=(const MyClass &other) { /* 自定义赋值逻辑 */ return *this; }
};
```

#### 19.3 多重继承
C++ 支持一个类从多个基类继承。例如：
```cpp
class Base1 { /* ... */ };
class Base2 { /* ... */ };
class Derived : public Base1, public Base2 { /* ... */ };
```

#### 19.4 虚继承
虚继承解决了多重继承中的二义性问题。例如：
```cpp
class Base { /* ... */ };
class Derived1 : virtual public Base { /* ... */ };
class Derived2 : virtual public Base { /* ... */ };
class MostDerived : public Derived1, public Derived2 { /* ... */ };
```

### 20. 智能指针
智能指针是 C++ 提供的一种 RAII（资源获取即初始化）工具，用于自动管理动态内存，避免内存泄漏。

#### 20.1 `std::unique_ptr`
- **`std::unique_ptr`** 是独占所有权的智能指针，不允许复制。
  ```cpp
  std::unique_ptr<int> p1(new int(10));
  ```

#### 20.2 `std::shared_ptr`
- **`std::shared_ptr`** 是共享所有权的智能指针，多个指针可以共享同一块内存，自动管理内存释放。
  ```cpp
  std::shared_ptr<int> p2 = std::make_shared<int>(20);
  ```

#### 20.3 `std::weak_ptr`
- **`std::weak_ptr`** 是一种弱引用智能指针，用于避免循环引用问题。
  ```cpp
  std::weak_ptr<int> wp = p2;
  ```

### 21. 多线程编程
C++ 提供了多线程支持，可以并发执行代码，提高程序的性能。

#### 21.1 创建线程
使用 `std::thread` 类可以创建并管理线程。例如：
```cpp
void print() {
    std::cout << "Thread is running" << std::endl;
}

std::thread t1(print);
t1.join(); // 等待线程结束
```

#### 21.2 线程同步
线程间的同步可以使用互斥锁 `std::mutex`，避免数据竞争问题。例如：
```cpp
std::mutex mtx;

void thread_safe_function() {
    std::lock_guard<std::mutex> lock(mtx);
    // 线程安全的代码
}
```

### 22. 文件操作
C++ 提供了标准的文件输入输出库，支持对文件进行读写操作。

#### 22.1 读取文件
```cpp
std::ifstream infile("example.txt");
std::string line;
while (std::getline(infile, line)) {
    std::cout << line << std::endl;
}
infile.close();
```

#### 22.2 写入文件
```cpp
std::ofstream outfile("example.txt");
outfile << "Hello, file!" << std::endl;
outfile.close();
```

### 23. C++11/14/17/20 新特性
C++ 标准不断演进，引入了许多新特性，提升了开发效率和代码质量。

#### 23.1 自动类型推导（C++11）
使用 `auto` 关键字自动推导变量类型。
```cpp
auto i = 10; // i 被推导为 int 类型
```

#### 23.2 范围循环（C++11）
简化了对容器的遍历。
```cpp
std::vector<int> vec = {1, 2, 3, 4};
for (int n : vec) {
    std::cout << n << std::endl;
}
```

#### 23.3 Lambda 表达式（C++11）
匿名函数的简洁写法。
```cpp
auto add = [](int a, int b) { return a + b; };
std::cout << add(2, 3) << std::endl;
```

#### 23.4 结构化绑定（C++17）
解构元组或结构体。
```cpp
std::tuple<int, double> t(1, 2.0);
auto [a, b] = t;
```

#### 23.5 占位符语法（C++20）
简化模板函数的定义。
```cpp
auto f(auto a, auto b) { return a + b; }
```

### 24. 综合实例
最后，通过一个综合实例将上述内容整合在一起。例如，实现一个简单的银行账户管理系统，包括账户创建、存取款操作、账户信息显示等功能。

```cpp
#include <iostream>
#include <vector>
#include <memory>

class Account {
public:
    Account(std::string owner, double balance) 
        : owner_(owner), balance_(balance) {}

    void deposit(double amount) {
        balance_ += amount;
    }

    void withdraw(double amount) {
        if (amount <= balance_) {
            balance_ -= amount;
        } else {
            std::cout << "Insufficient funds" << std::endl;
        }
    }

    void display() const {
        std::cout << "Owner: " << owner_ << ", Balance: " << balance_ << std::endl;
    }

private:
    std::string owner_;
    double balance_;
};

int main() {
    std::vector<std::unique_ptr<Account>> accounts;
    accounts.push_back(std::make_unique<Account>("Alice", 1000.0));
    accounts.push_back(std::make_unique<Account>("Bob", 500.0));

    accounts[0]->deposit(500.0);
    accounts[1]->withdraw(100.0);

    for (const auto& account : accounts) {
        account->display();
    }

    return 0;
}
```

