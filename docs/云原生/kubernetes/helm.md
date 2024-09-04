# kubernetes 包管理工具 helm

Helm 是 Kubernetes 的包管理工具，类似于操作系统中的 `apt` 或 `yum`，用于简化 Kubernetes 应用的部署和管理。Helm 通过称为“Charts”的包来定义、安装和管理 Kubernetes 应用。接下来我们详细讲解 Helm 的使用，包括其基本概念、安装步骤、主要命令以及如何创建和管理自己的 Helm Chart。

## 1. Helm 基本概念

### 1.1 Helm 和 Chart
- **Helm**：Kubernetes 的包管理工具，用于管理 Kubernetes 应用程序的定义、安装和升级。
- **Chart**：Helm 包的单位，是描述一个 Kubernetes 应用的一组文件，包括 Kubernetes 资源的定义文件、配置文件等。
- **Release**：通过 Helm 部署的一个 Chart 实例。在同一个集群中，Chart 可以被多次安装，每次安装称为一个 Release，每个 Release 都有一个唯一的名称。
- **Repository**：存放 Charts 的集合。Helm 可以从多个 Repository 中获取 Charts。

### 1.2 Helm 的工作原理
Helm 的工作分为客户端（Helm CLI）和服务端（Tiller）。在 Helm 3 中，Tiller 已经被移除，Helm CLI 直接与 Kubernetes API 交互。

- **客户端**：用于开发、版本管理、配置和 Chart 的发布。
- **服务端**：接收 Helm 客户端请求，并在 Kubernetes 集群中执行相应的操作（Helm 2 及以前使用 Tiller，Helm 3 移除了 Tiller）。

## 2. Helm 的安装

### 2.1 安装 Helm

Helm 的安装方式有多种，以下介绍使用脚本和包管理工具的方式：

#### 2.1.1 使用脚本安装

在 Linux 或 macOS 系统中，使用以下命令快速安装 Helm：

```bash
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

#### 2.1.2 使用包管理工具安装

在 macOS 系统中，你可以通过 Homebrew 安装 Helm：

```bash
brew install helm
```

在 Linux 系统中，如果使用 `apt`：

```bash
sudo apt-get update
sudo apt-get install -y helm
```

安装完成后，可以使用 `helm version` 命令检查是否安装成功：

```bash
helm version
```

## 3. Helm 的基本使用

### 3.1 添加 Chart 仓库

Helm 的 Chart 通常存放在仓库中。官方的默认仓库是 `stable` 仓库，可以通过以下命令添加其他仓库：

```bash
helm repo add stable https://charts.helm.sh/stable
helm repo add bitnami https://charts.bitnami.com/bitnami
```

通过 `helm repo list` 可以查看已添加的仓库：

```bash
helm repo list
```

### 3.2 搜索和查找 Charts

通过 `helm search` 命令可以在仓库中搜索 Charts：

- 搜索仓库中的 Chart：
  ```bash
  helm search repo <chart_name>
  ```

- 示例：查找 MySQL Chart：
  ```bash
  helm search repo mysql
  ```

### 3.3 安装 Chart

使用 `helm install` 命令安装 Chart：

```bash
helm install <release_name> <chart_name>
```

例如，安装 MySQL Chart：

```bash
helm install my-mysql stable/mysql
```

### 3.4 查看 Release

安装完成后，可以使用 `helm list` 命令查看当前集群中所有的 Release：

```bash
helm list
```

### 3.5 升级 Release

当 Chart 版本或配置文件更新时，可以使用 `helm upgrade` 命令升级 Release：

```bash
helm upgrade <release_name> <chart_name>
```

例如，升级 MySQL Release：

```bash
helm upgrade my-mysql stable/mysql
```

### 3.6 回滚 Release

使用 `helm rollback` 可以将 Release 回滚到指定的版本：

```bash
helm rollback <release_name> <revision_number>
```

查看 Release 的版本历史：

```bash
helm history <release_name>
```

### 3.7 卸载 Release

使用 `helm uninstall` 命令可以卸载 Release：

```bash
helm uninstall <release_name>
```

### 3.8 查看 Release 状态

使用 `helm status` 可以查看 Release 的详细状态信息：

```bash
helm status <release_name>
```

## 4. 创建和管理 Helm Chart

### 4.1 创建一个新的 Chart

使用 `helm create` 命令创建一个新的 Helm Chart：

```bash
helm create mychart
```

这将在当前目录下生成一个名为 `mychart` 的目录结构，其中包含一些默认的模板和配置文件。

### 4.2 Chart 目录结构

一个典型的 Chart 目录结构如下：

```
mychart/
  Chart.yaml          # Chart 的元数据，如名称、版本等
  values.yaml         # 默认的配置值
  templates/          # Kubernetes 资源的模板文件
    deployment.yaml
    service.yaml
    ...
  charts/             # 子 Charts
  .helmignore         # 忽略文件列表
```

### 4.3 自定义模板

`templates/` 目录下的文件是 Kubernetes 资源的模板文件，可以使用 Go 模板语法来定义动态的配置。

#### 示例：自定义 `deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-{{ .Values.app.name }}
spec:
  replicas: {{ .Values.replicas }}
  selector:
    matchLabels:
      app: {{ .Values.app.name }}
  template:
    metadata:
      labels:
        app: {{ .Values.app.name }}
    spec:
      containers:
      - name: {{ .Values.app.name }}
        image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
        ports:
        - containerPort: {{ .Values.service.port }}
```

### 4.4 使用 Values 文件进行配置

`values.yaml` 文件包含了 Chart 的默认配置值。你可以在安装 Chart 时使用 `-f` 选项指定自定义的 `values.yaml` 文件：

```bash
helm install myapp ./mychart -f custom-values.yaml
```

### 4.5 打包和发布 Chart

使用 `helm package` 命令可以将 Chart 打包成 `.tgz` 文件：

```bash
helm package mychart
```

生成的 `.tgz` 文件可以上传到一个 Chart 仓库，供其他用户使用。

### 4.6 验证和调试 Chart

使用 `helm lint` 命令可以对 Chart 进行语法和结构的检查：

```bash
helm lint mychart
```

使用 `helm template` 命令可以在不实际部署的情况下，生成 Kubernetes 资源的 YAML 文件，用于调试：

```bash
helm template mychart
```

## 5. Helm 的进阶使用

### 5.1 Helm Hooks

Helm Hooks 是一组特殊的模板，可以在 Chart 安装、升级、删除的过程中触发执行，用于自定义操作，如数据迁移、备份等。

### 5.2 子 Chart 和依赖管理

`requirements.yaml` 文件定义了 Chart 的依赖关系，Helm 会自动下载并管理这些依赖：

```yaml
dependencies:
  - name: redis
    version: ">=10.0.0"
    repository: "https://charts.bitnami.com/bitnami"
```

安装 Chart 时，Helm 会自动下载并安装依赖的子 Chart。

### 5.3 Helm 安全性

Helm 3 移除了 Tiller，安全性大大提高。对于敏感配置（如密码），可以使用 Kubernetes Secret 来管理。

## 6. 实战示例：使用 Helm 部署 WordPress

使用 Helm 从 Bitnami 仓库中安装 WordPress：

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-wordpress bitnami/wordpress
```

安装完成后，使用 `helm list` 查看 Release，使用 `helm status` 查看 WordPress 的运行状态。

通过 NodePort 或 Ingress 访问 WordPress 应用，进一步配置和使用。

---

以上是 Helm 的详细教程，包括基本概念、安装、主要命令以及如何创建和管理自己的 Helm Chart。通过 Helm，你可以简化 Kubernetes 应用的管理，提高效率和可维护性。如果有任何疑问或需要更深入的解释，请随时提出。