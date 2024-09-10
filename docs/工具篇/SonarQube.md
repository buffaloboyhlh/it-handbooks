# SonarQube 基本教程

SonarQube 是一个开源的代码质量管理工具，主要用于检测代码中的漏洞、代码异味、重复代码、单元测试覆盖率等问题，并提供详细的质量报告。它适用于多种编程语言（如 Java、Python、JavaScript 等），可以帮助开发团队维护代码质量并持续改进。

以下是 SonarQube 的详细教程，涵盖从安装到配置和使用的每个步骤。

---

### 一、SonarQube 安装

#### 1. 系统要求

- **操作系统**：SonarQube 支持运行在各种操作系统上，如 Linux、Windows、macOS。
- **数据库**：需要一个数据库来存储代码分析的结果。支持的数据库包括 PostgreSQL、MySQL、Oracle 等。
- **Java 环境**：SonarQube 需要 JDK 11 或更高版本的支持。

#### 2. 安装步骤

1. **下载 SonarQube**：
   - 从官方网站下载 SonarQube 的最新版本：[SonarQube 下载链接](https://www.sonarqube.org/downloads/)
   
2. **解压并安装**：
   下载完成后，解压 SonarQube 安装包。例如：

   ```bash
   unzip sonarqube-<version>.zip
   ```

3. **配置数据库**：
   编辑 SonarQube 的配置文件 `sonar.properties`，配置数据库连接信息：

   - 位置：`/conf/sonar.properties`
   
   - 配置示例（以 PostgreSQL 为例）：

     ```ini
     sonar.jdbc.username=sonar
     sonar.jdbc.password=sonar
     sonar.jdbc.url=jdbc:postgresql://localhost/sonar
     ```

4. **启动 SonarQube**：
   进入 SonarQube 安装目录，运行以下命令启动服务：

   - 在 Linux/macOS 上：

     ```bash
     ./bin/linux-x86-64/sonar.sh start
     ```

   - 在 Windows 上：

     ```bash
     ./bin/windows-x86-64/StartSonar.bat
     ```

5. **访问 SonarQube**：
   启动成功后，打开浏览器，访问 `http://localhost:9000`。使用默认管理员账户登录：

   - **用户名**：`admin`
   - **密码**：`admin`

   登录后，建议立即修改默认密码以提高安全性。

### 二、SonarQube 基本配置

#### 1. 创建项目

1. 登录 SonarQube Web 界面。
2. 点击页面顶部的 **"Create Project"**，填写项目名称和项目密钥。
3. 生成一个 **Token**，用于项目分析时的身份验证。

#### 2. 配置分析工具（SonarQube Scanner）

SonarQube Scanner 是一个命令行工具，用于将项目代码上传到 SonarQube 服务器进行分析。

1. **下载 SonarQube Scanner**：[下载链接](https://docs.sonarqube.org/latest/analysis/scan/sonarscanner/)
2. **配置 SonarQube Scanner**：
   编辑配置文件 `sonar-scanner.properties`，设置 SonarQube 服务器的地址和认证信息，例如：

   ```ini
   sonar.host.url=http://localhost:9000
   sonar.login=your_token
   ```

3. **添加 `sonar-project.properties` 文件**：
   在项目根目录下创建 `sonar-project.properties` 文件，定义项目的 SonarQube 分析配置：

   ```ini
   # 项目唯一标识符
   sonar.projectKey=my_project_key
   # 项目名称
   sonar.projectName=My Project
   # 项目版本
   sonar.projectVersion=1.0
   # 源代码目录
   sonar.sources=src
   # 文件编码
   sonar.sourceEncoding=UTF-8
   ```

4. **运行 SonarQube Scanner**：
   打开命令行，进入项目目录，运行以下命令进行代码分析：

   ```bash
   sonar-scanner
   ```

   SonarQube Scanner 会扫描项目代码并将分析结果上传到 SonarQube 服务器。

### 三、SonarQube 分析结果

#### 1. 查看分析报告

完成代码分析后，可以在 SonarQube 的 Web 界面上查看项目的分析报告。报告包括以下几个主要指标：

- **漏洞（Vulnerabilities）**：代码中的安全问题。
- **错误（Bugs）**：代码中的潜在错误。
- **代码异味（Code Smells）**：不影响代码功能但可能影响可维护性的问题。
- **重复率（Duplications）**：代码中的重复段落。
- **单元测试覆盖率（Coverage）**：单元测试覆盖的代码比例。
- **技术债务（Technical Debt）**：需要花费时间来修复的代码问题。

#### 2. 质量门（Quality Gates）

SonarQube 提供了 **质量门（Quality Gates）** 功能，允许你设置代码质量的标准。如果项目不符合这些标准，则项目的质量门状态会被标记为不合格。

你可以在 SonarQube 管理界面配置自定义的质量门，例如：
- 允许的代码重复率不得超过 5%。
- 漏洞数量必须为 0。
- 单元测试覆盖率必须至少达到 80%。

### 四、在 CI/CD 中集成 SonarQube

SonarQube 常用于持续集成/持续交付（CI/CD）中，帮助开发团队自动化代码质量分析。以下是如何将 SonarQube 集成到 CI/CD 流水线中的示例。

#### 1. 在 Jenkins 中集成 SonarQube

1. **安装 SonarQube 插件**：
   在 Jenkins 中安装 `SonarQube Scanner` 插件。

2. **配置 SonarQube 服务器**：
   在 Jenkins 中的 `Manage Jenkins` -> `Configure System` 中，添加 SonarQube 服务器的连接信息。

3. **添加 Pipeline 脚本**：

   在 Jenkins Pipeline 中添加 SonarQube 分析步骤：

   ```groovy
   pipeline {
       agent any
       stages {
           stage('Build') {
               steps {
                   echo 'Building...'
               }
           }
           stage('SonarQube analysis') {
               steps {
                   withSonarQubeEnv('SonarQube') {
                       sh 'sonar-scanner'
                   }
               }
           }
       }
   }
   ```

#### 2. 在 GitLab CI 中集成 SonarQube

1. **配置 `.gitlab-ci.yml` 文件**：

   在 GitLab CI 中，你可以将 SonarQube 扫描添加到 CI 流水线中。添加以下内容到 `.gitlab-ci.yml` 文件中：

   ```yaml
   stages:
     - build
     - sonarqube

   sonarqube-check:
     stage: sonarqube
     image: sonarsource/sonar-scanner-cli:latest
     script:
       - sonar-scanner
     allow_failure: true
     only:
       - master
   ```

### 五、SonarQube 高级配置

#### 1. 多模块项目分析

对于多模块项目，你可以通过在 `sonar-project.properties` 文件中添加模块的配置信息来对项目进行更详细的分析。例如：

```ini
sonar.modules=module1,module2
module1.sonar.projectName=Module 1
module1.sonar.sources=module1/src
module2.sonar.projectName=Module 2
module2.sonar.sources=module2/src
```

#### 2. 自定义质量规则

SonarQube 提供了多种编程语言的内置质量规则，你也可以根据团队需求创建自定义的规则。例如，可以自定义规则来检查特定的代码风格、命名约定或安全问题。

#### 3. 插件扩展

SonarQube 支持安装多种插件来扩展功能，例如额外的编程语言支持、安全分析插件、性能优化工具等。你可以在 SonarQube 管理界面中浏览和安装所需插件。

### 六、总结

SonarQube 是一个强大的代码质量管理工具，适用于个人开发者和大型团队。它提供了全面的代码分析功能，通过静态分析来帮助开发人员及早发现问题。通过将 SonarQube 集成到 CI/CD 流水线中，你可以实现自动化代码质量检测，提升项目的整体代码质量。

关键步骤包括：
1. 安装和配置 SonarQube 服务器。
2. 配置 SonarQube Scanner 或使用 SonarLint 进行代码分析。
3. 在 SonarQube 中查看分析报告，持续监控和优化代码质量。
4. 将 SonarQube 集成到 Jenkins、GitLab 等 CI/CD 流水线中，实现持续自动化的代码质量检查。