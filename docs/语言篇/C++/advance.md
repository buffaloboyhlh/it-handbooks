# C++ 进阶教程

以下是 C++ 进阶教程的详细讲解，涵盖了更复杂的主题和高级特性。这个教程将帮助你深入理解 C++ 的高级特性及其应用。

### 1. 动态内存管理

#### 1.1 `new` 和 `delete`
- **`new`** 用于在堆上分配内存，返回分配内存的指针。
  ```cpp
  int* p = new int; // 分配一个整数大小的内存
  *p = 10; // 设置内存中的值
  ```
- **`delete`** 用于释放使用 `new` 分配的内存，避免内存泄漏。
  ```cpp
  delete p; // 释放内存
  ```

#### 1.2 动态数组
- **`new[]`** 用于分配动态数组。
  ```cpp
  int* arr = new int[5]; // 分配一个包含5个整数的数组
  arr[0] = 1; // 访问数组元素
  delete[] arr; // 释放动态数组
  ```

#### 1.3 自定义内存管理
- 可以通过重载 `new` 和 `delete` 操作符自定义内存分配策略。
  ```cpp
  void* operator new(size_t size) {
      std::cout << "Custom new: " << size << " bytes" << std::endl;
      return malloc(size);
  }

  void operator delete(void* pointer) {
      std::cout << "Custom delete" << std::endl;
      free(pointer);
  }
  ```

### 2. 标准模板库（STL）

#### 2.1 容器
- **`std::vector`**：动态数组，支持快速随机访问和动态扩展。
  ```cpp
  std::vector<int> vec = {1, 2, 3, 4};
  vec.push_back(5); // 添加元素
  vec.pop_back(); // 删除最后一个元素
  ```

- **`std::list`**：双向链表，适用于频繁的插入和删除操作。
  ```cpp
  std::list<int> lst = {1, 2, 3};
  lst.push_back(4); // 添加元素
  lst.pop_front(); // 删除第一个元素
  ```

- **`std::map`**：基于红黑树的关联容器，存储键值对。
  ```cpp
  std::map<std::string, int> m;
  m["apple"] = 5;
  m["banana"] = 2;
  ```

#### 2.2 迭代器
- 迭代器用于遍历容器中的元素。
  ```cpp
  std::vector<int> vec = {1, 2, 3, 4};
  for (auto it = vec.begin(); it != vec.end(); ++it) {
      std::cout << *it << " ";
  }
  ```

#### 2.3 算法
- STL 提供了多种算法，如排序、查找、修改等。
  ```cpp
  std::vector<int> vec = {4, 2, 3, 1};
  std::sort(vec.begin(), vec.end()); // 排序
  ```

### 3. 高级类设计

#### 3.1 构造函数和析构函数
- **构造函数**：用于初始化对象。
  ```cpp
  class MyClass {
  public:
      MyClass() { std::cout << "Constructor" << std::endl; }
  };
  ```

- **析构函数**：用于清理资源。
  ```cpp
  class MyClass {
  public:
      ~MyClass() { std::cout << "Destructor" << std::endl; }
  };
  ```

#### 3.2 拷贝构造函数和赋值运算符
- 用于处理对象的拷贝和赋值。
  ```cpp
  class MyClass {
  public:
      MyClass(const MyClass& other) { /* 拷贝构造逻辑 */ }
      MyClass& operator=(const MyClass& other) { /* 赋值逻辑 */ return *this; }
  };
  ```

#### 3.3 多重继承
- 一个类可以继承多个基类。
  ```cpp
  class Base1 { /* ... */ };
  class Base2 { /* ... */ };
  class Derived : public Base1, public Base2 { /* ... */ };
  ```

#### 3.4 虚继承
- 用于解决多重继承中的二义性问题。
  ```cpp
  class Base { /* ... */ };
  class Derived1 : virtual public Base { /* ... */ };
  class Derived2 : virtual public Base { /* ... */ };
  class MostDerived : public Derived1, public Derived2 { /* ... */ };
  ```

### 4. 智能指针

#### 4.1 `std::unique_ptr`
- 独占所有权的智能指针，不能复制，但可以转移所有权。
  ```cpp
  std::unique_ptr<int> p1(new int(10));
  ```

#### 4.2 `std::shared_ptr`
- 共享所有权的智能指针，多个指针可以共享同一块内存，自动管理内存释放。
  ```cpp
  std::shared_ptr<int> p2 = std::make_shared<int>(20);
  ```

#### 4.3 `std::weak_ptr`
- 弱引用智能指针，用于避免循环引用问题。
  ```cpp
  std::weak_ptr<int> wp = p2;
  ```

### 5. 多线程编程

#### 5.1 创建线程
- 使用 `std::thread` 类创建并管理线程。
  ```cpp
  void print() {
      std::cout << "Thread is running" << std::endl;
  }

  std::thread t1(print);
  t1.join(); // 等待线程结束
  ```

#### 5.2 线程同步
- 使用互斥锁 `std::mutex` 避免数据竞争。
  ```cpp
  std::mutex mtx;

  void thread_safe_function() {
      std::lock_guard<std::mutex> lock(mtx);
      // 线程安全的代码
  }
  ```

### 6. 文件操作

#### 6.1 读取文件
- 使用 `std::ifstream` 读取文件内容。
  ```cpp
  std::ifstream infile("example.txt");
  std::string line;
  while (std::getline(infile, line)) {
      std::cout << line << std::endl;
  }
  infile.close();
  ```

#### 6.2 写入文件
- 使用 `std::ofstream` 写入文件内容。
  ```cpp
  std::ofstream outfile("example.txt");
  outfile << "Hello, file!" << std::endl;
  outfile.close();
  ```

### 7. C++11/14/17/20 新特性

#### 7.1 自动类型推导（C++11）
- 使用 `auto` 关键字自动推导变量类型。
  ```cpp
  auto i = 10; // i 被推导为 int 类型
  ```

#### 7.2 范围循环（C++11）
- 简化对容器的遍历。
  ```cpp
  std::vector<int> vec = {1, 2, 3, 4};
  for (int n : vec) {
      std::cout << n << std::endl;
  }
  ```

#### 7.3 Lambda 表达式（C++11）
- 匿名函数的简洁写法。
  ```cpp
  auto add = [](int a, int b) { return a + b; };
  std::cout << add(2, 3) << std::endl;
  ```

#### 7.4 结构化绑定（C++17）
- 解构元组或结构体。
  ```cpp
  std::tuple<int, double> t(1, 2.0);
  auto [a, b] = t;
  ```

#### 7.5 占位符语法（C++20）
- 简化模板函数的定义。
  ```cpp
  auto f(auto a, auto b) { return a + b; }
  ```

### 8. 综合实例

以下是一个更复杂的示例，展示了如何将上述知识应用于一个简单的银行账户管理系统。

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <string>

class Account {
public:
    Account(std::string owner, double balance) 
        : owner_(std::move(owner)), balance_(balance) {}

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

当然，我们可以进一步扩展内容，涉及到一些 C++ 进阶主题，如设计模式、元编程、高级模板编程、异常处理、动态绑定和虚函数等。

### 10. 设计模式

设计模式是一种通用的解决方案，用于解决软件设计中的常见问题。以下是几种常见的设计模式。

#### 10.1 单例模式
- 确保一个类只有一个实例，并提供一个全局访问点。
  ```cpp
  class Singleton {
  public:
      static Singleton& getInstance() {
          static Singleton instance; // 局部静态变量
          return instance;
      }
      // 删除拷贝构造函数和赋值运算符
      Singleton(const Singleton&) = delete;
      Singleton& operator=(const Singleton&) = delete;
  private:
      Singleton() {} // 私有构造函数
  };
  ```

#### 10.2 工厂模式
- 提供一个创建对象的接口，而无需指定具体的类。
  ```cpp
  class Product {
  public:
      virtual void use() = 0;
  };

  class ConcreteProductA : public Product {
  public:
      void use() override { std::cout << "Product A" << std::endl; }
  };

  class ConcreteProductB : public Product {
  public:
      void use() override { std::cout << "Product B" << std::endl; }
  };

  class Factory {
  public:
      static Product* createProduct(int type) {
          if (type == 1) return new ConcreteProductA();
          if (type == 2) return new ConcreteProductB();
          return nullptr;
      }
  };
  ```

#### 10.3 观察者模式
- 定义一种一对多的依赖关系，使得当一个对象的状态发生改变时，所有依赖于它的对象都会收到通知。
  ```cpp
  class Observer {
  public:
      virtual void update() = 0;
  };

  class Subject {
  public:
      void attach(Observer* obs) {
          observers.push_back(obs);
      }
      void notify() {
          for (auto& obs : observers) {
              obs->update();
          }
      }
  private:
      std::vector<Observer*> observers;
  };
  ```

### 11. 模板编程

#### 11.1 函数模板
- 允许编写泛型函数，可以处理不同类型的数据。
  ```cpp
  template <typename T>
  T max(T a, T b) {
      return (a > b) ? a : b;
  }
  ```

#### 11.2 类模板
- 允许编写泛型类，可以处理不同类型的数据。
  ```cpp
  template <typename T>
  class Stack {
  public:
      void push(const T& value) {
          data.push_back(value);
      }
      T pop() {
          T value = data.back();
          data.pop_back();
          return value;
      }
  private:
      std::vector<T> data;
  };
  ```

#### 11.3 模板特化
- 允许为特定类型提供不同的实现。
  ```cpp
  template <typename T>
  class Calculator {
  public:
      T add(T a, T b) { return a + b; }
  };

  // 特化版本
  template <>
  class Calculator<std::string> {
  public:
      std::string add(const std::string& a, const std::string& b) { return a + b; }
  };
  ```

### 12. 元编程

元编程是编写程序以生成其他程序的技术。C++ 中的模板元编程可以在编译时进行计算和决策。

#### 12.1 编译时常量
- 使用 `constexpr` 进行编译时计算。
  ```cpp
  constexpr int factorial(int n) {
      return (n <= 1) ? 1 : n * factorial(n - 1);
  }
  ```

#### 12.2 类型特征
- 使用 `std::is_same` 判断类型是否相同。
  ```cpp
  #include <type_traits>

  static_assert(std::is_same<int, int>::value, "Types are not the same");
  ```

#### 12.3 类型萃取
- 提取类型信息，例如获取类型的大小或是否为整数。
  ```cpp
  template <typename T>
  struct TypeTraits {
      static const bool isInteger = false;
  };

  template <>
  struct TypeTraits<int> {
      static const bool isInteger = true;
  };
  ```

### 13. 异常处理

异常处理是 C++ 处理运行时错误的一种机制。

#### 13.1 基本异常处理
- 使用 `try`、`catch` 和 `throw` 进行异常处理。
  ```cpp
  try {
      throw std::runtime_error("An error occurred");
  } catch (const std::exception& e) {
      std::cout << "Caught exception: " << e.what() << std::endl;
  }
  ```

#### 13.2 自定义异常类
- 自定义异常类以处理特定错误。
  ```cpp
  class MyException : public std::exception {
  public:
      const char* what() const noexcept override {
          return "MyException occurred";
      }
  };

  throw MyException();
  ```

#### 13.3 异常安全
- 确保资源在异常发生时不会泄漏。
  ```cpp
  class Resource {
  public:
      Resource() { /* allocate resource */ }
      ~Resource() { /* release resource */ }
  private:
      // 禁止拷贝
      Resource(const Resource&) = delete;
      Resource& operator=(const Resource&) = delete;
  };
  ```

### 14. 动态绑定和虚函数

#### 14.1 虚函数
- 使用虚函数实现多态性。
  ```cpp
  class Base {
  public:
      virtual void show() { std::cout << "Base class" << std::endl; }
  };

  class Derived : public Base {
  public:
      void show() override { std::cout << "Derived class" << std::endl; }
  };

  Base* b = new Derived();
  b->show(); // 输出 "Derived class"
  ```

#### 14.2 虚析构函数
- 确保在删除基类指针时调用派生类的析构函数。
  ```cpp
  class Base {
  public:
      virtual ~Base() { /* cleanup */ }
  };

  class Derived : public Base {
  public:
      ~Derived() override { /* cleanup */ }
  };
  ```

### 15. 内存管理优化

#### 15.1 内存池
- 通过内存池优化内存分配。
  ```cpp
  class MemoryPool {
  public:
      void* allocate(size_t size) { /* allocation logic */ }
      void deallocate(void* ptr) { /* deallocation logic */ }
  private:
      // 内存池管理代码
  };
  ```

#### 15.2 对象池
- 使用对象池管理固定数量的对象实例。
  ```cpp
  class ObjectPool {
  public:
      ObjectPool(size_t size) : poolSize(size) { /* initialize pool */ }
      MyObject* acquire() { /* acquire object */ }
      void release(MyObject* obj) { /* release object */ }
  private:
      size_t poolSize;
      // 对象池管理代码
  };
  ```

### 16. 高级模板编程

#### 16.1 模板元编程
- 使用模板实现编译时计算。
  ```cpp
  template <int N>
  struct Factorial {
      static const int value = N * Factorial<N - 1>::value;
  };

  template <>
  struct Factorial<0> {
      static const int value = 1;
  };
  ```

#### 16.2 SFINAE（Substitution Failure Is Not An Error）
- 使用 SFINAE 进行条件编译。
  ```cpp
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, void>::type
  print(T value) {
      std::cout << "Integral: " << value << std::endl;
  }
  ```

### 17. 总结与下一步

掌握 C++ 的进阶特性需要深入学习和实践。可以参考以下资源：
- **书籍**：《C++ Primer》、《Effective C++》、《Modern C++ Design》
- **在线资源**：C++ 标准库文档、C++ 参考手册
- **实践**：参与开源项目、编写复杂的 C++ 程序、解决实际问题

通过这些内容的学习和实践，你将能够更高效地使用 C++ 进行复杂的编程任务，并提升编程能力。