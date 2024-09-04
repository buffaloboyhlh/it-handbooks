# kubernetes 教程

## 1. Kubernetes 基础概念

### 1.1 什么是 Kubernetes？

Kubernetes（简称 K8s）是一个开源的容器编排平台，用于自动化部署、扩展和管理容器化应用程序。它由 Google 开发，后来成为 CNCF（Cloud Native Computing Foundation）的一部分。Kubernetes 旨在简化容器的管理，使得部署和管理大规模容器化应用变得更加高效和可靠。

### 1.2 Kubernetes 架构

Kubernetes 的架构主要包括两个部分：**控制平面** 和 **节点**。

- **控制平面（Control Plane）**：负责管理整个 Kubernetes 集群，调度 Pods 和管理集群状态。控制平面通常由以下组件组成：
  - **API Server**：Kubernetes 的 API 入口点，所有对集群的操作都是通过 API Server 进行的。
  - **Controller Manager**：监控集群的状态，并负责处理控制循环，如确保 Pods 数量符合预期。
  - **Scheduler**：负责将 Pods 调度到合适的节点上。
  - **etcd**：分布式键值存储，用于保存集群状态和配置数据。

- **节点（Nodes）**：集群中的工作机器，每个节点运行一个或多个 Pods。节点主要包括：
  - **Kubelet**：管理容器的生命周期，确保容器按预期运行。
  - **Kube Proxy**：维护网络规则，确保服务的网络通信。
  - **Container Runtime**：运行容器的工具，如 Docker、containerd 等。

### 1.3 核心概念

- **Pod**：Kubernetes 中的基本运行单元，一个 Pod 可以包含一个或多个容器，它们共享网络、存储等资源。
- **Service**：定义访问 Pods 的方法，提供负载均衡和服务发现功能。
- **Deployment**：管理 Pods 和 ReplicaSets，实现应用的部署和滚动更新。
- **ReplicaSet**：确保指定数量的 Pods 副本在集群中运行。
- **Namespace**：用于组织和隔离集群资源，支持多租户环境。
- **ConfigMap**：管理非敏感配置数据。
- **Secret**：管理敏感数据，如密码和密钥。
- **Volume**：用于持久化数据存储。

---

## 2. Kubernetes 核心对象

### 2.1 Pod

**Pod** 是 Kubernetes 中最小的调度单元，包含一个或多个容器。这些容器共享网络、存储等资源，通常一起部署和管理。

**Pod 的配置示例**：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  labels:
    app: my-app
spec:
  containers:
  - name: my-container
    image: nginx:latest
    ports:
    - containerPort: 80
  restartPolicy: Always
```

- **metadata**：Pod 的元数据，包括名称和标签。
- **spec**：Pod 的规范，包括容器配置和重启策略。
- **containers**：Pod 中的容器列表，每个容器包括名称、镜像、暴露的端口等信息。

**操作命令**：

```bash
kubectl apply -f pod.yaml
kubectl get pods
kubectl describe pod my-pod
kubectl delete pod my-pod
```

### 2.2 Service

**Service** 提供稳定的访问点来访问一组 Pods，支持负载均衡和服务发现。

**Service 的配置示例**：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: ClusterIP
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
```

- **type**：指定 Service 的类型（ClusterIP、NodePort、LoadBalancer、ExternalName）。
- **selector**：选择与 Service 匹配的 Pods。
- **ports**：指定 Service 的端口和目标端口。

**操作命令**：

```bash
kubectl apply -f service.yaml
kubectl get services
kubectl describe service my-service
kubectl delete service my-service
```

### 2.3 Deployment

**Deployment** 用于声明式地管理 Pods 和 ReplicaSets，实现应用的部署和更新。

**Deployment 的配置示例**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-container
        image: nginx:latest
        ports:
        - containerPort: 80
```

- **replicas**：指定 Pod 副本的数量。
- **selector**：选择与 Deployment 匹配的 Pods。
- **template**：Pod 模板，包括 Pod 的规范。

**操作命令**：

```bash
kubectl apply -f deployment.yaml
kubectl get deployments
kubectl describe deployment my-deployment
kubectl rollout status deployment/my-deployment
kubectl set image deployment/my-deployment my-container=nginx:latest
kubectl rollout undo deployment/my-deployment
kubectl delete deployment my-deployment
```

### 2.4 ReplicaSet

**ReplicaSet** 确保指定数量的 Pod 副本在集群中运行。通常不直接创建 ReplicaSet，而是通过 Deployment 管理。

**ReplicaSet 的配置示例**：

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: my-replicaset
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-container
        image: nginx:latest
        ports:
        - containerPort: 80
```

**操作命令**：

```bash
kubectl apply -f replicaset.yaml
kubectl get replicasets
kubectl describe replicaset my-replicaset
kubectl delete replicaset my-replicaset
```

### 2.5 Namespace

**Namespace** 用于将集群中的资源分隔到逻辑上不同的区域中。

**Namespace 的配置示例**：

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: my-namespace
```

**操作命令**：

```bash
kubectl apply -f namespace.yaml
kubectl get namespaces
kubectl describe namespace my-namespace
kubectl delete namespace my-namespace
```

### 2.6 ConfigMap 和 Secret

**ConfigMap** 和 **Secret** 用于管理配置数据和敏感数据。

**ConfigMap 的配置示例**：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
data:
  key1: value1
  key2: value2
```

**Secret 的配置示例**：

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-secret
type: Opaque
data:
  password: cGFzc3dvcmQ=  # base64 编码的密码
```

**操作命令**：

```bash
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl get configmaps
kubectl get secrets
kubectl describe configmap my-config
kubectl describe secret my-secret
kubectl delete configmap my-config
kubectl delete secret my-secret
```

### 2.7 Volume

**Volume** 用于存储数据并在 Pod 之间共享数据。

**PersistentVolume 和 PersistentVolumeClaim 的配置示例**：

**PersistentVolume**：

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /mnt/data
```

**PersistentVolumeClaim**：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

**在 Pod 中使用 PVC**：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: my-container
    image: nginx:latest
    volumeMounts:
    - mountPath: /usr/share/nginx/html
      name: my-storage
  volumes:
  - name: my-storage
    persistentVolumeClaim:
      claimName: my-pvc
```

**操作命令**：

```bash
kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml
kubectl apply -f pod.yaml
kubectl get persistentvolumes
kubectl get persistentvolumeclaims
```

---

## 3. 高级配置

### 3.1 网络配置

**NetworkPolicy** 用于控制 Pods 之间的网络通信。

**NetworkPolicy 的配置示例**：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: my-network-policy
spec:
  podSelector:
    matchLabels:
      app: my-app
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: my-namespace
```

**操作命令**：

```bash
kubectl apply -f networkpolicy.yaml
kubectl get networkpolicies
kubectl describe networkpolicy my-network-policy
```

### 3.2 Ingress

**Ingress** 提供 HTTP 和 HTTPS 访问到集群内部的服务，并支持 URL 路径和主机名的路由。

**Ingress 的配置示例**：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  rules:
  - host: my-app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: 80
```

**操作命令**：

```bash
kubectl apply -f ingress.yaml
kubectl get ingress
kubectl describe ingress my-ingress
```

### 3.3 Horizontal Pod Autoscaler

**Horizontal Pod Autoscaler** 根据 CPU 使用率或其他指标自动扩展 Pods。

**Horizontal Pod Autoscaler 的配置示例**：

```yaml
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: my-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-deployment
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

**操作命令**：

```bash
kubectl apply -f hpa.yaml
kubectl get hpa
kubectl describe hpa my-hpa
```

### 3.4 Cluster Autoscaler

**Cluster Autoscaler** 自动调整集群中节点的数量以适应负载需求。

**部署 Cluster Autoscaler 的配置示例**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      containers:
      - name: cluster-autoscaler
        image: k8s.gcr.io/cluster-autoscaler:v1.21.0
        command:
        - ./cluster-autoscaler
        - --v=4
        - --cloud-provider=aws
        - --cluster-name=my-cluster
        - --namespace=kube-system
```

**操作命令**：

```bash
kubectl apply -f cluster-autoscaler.yaml
kubectl get deployments -n kube-system
kubectl describe deployment cluster-autoscaler -n kube-system
```

### 3.5 监控和日志

- **Prometheus 和 Grafana** 用于集群和应用程序的监控和可视化。

**部署 Prometheus**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
        - "--config.file=/etc/prometheus/prometheus.yml"
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
```

**部署 Grafana**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
```

**部署 ELK Stack**（Elasticsearch, Logstash, Kibana）用于日志管理和分析。

**部署 Elasticsearch**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elasticsearch
  namespace: logging
spec:
  replicas: 1
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:7.10.1
        env:
        - name: discovery.type
          value: single-node
        ports:
        - containerPort: 9200
```

好的，我们继续深入探讨 Kubernetes 的更多高级主题，包括集群管理、调度策略、网络插件以及一些实际操作示例。

---

## 4. Kubernetes 高级主题

### 4.1 Kubernetes 调度策略

Kubernetes 提供了多种调度策略来管理 Pod 的调度和分配。理解这些策略可以帮助你优化集群的资源利用和性能。

#### 4.1.1 Pod 的调度

调度是 Kubernetes 将 Pod 分配到集群中适当节点上的过程。默认情况下，Kubernetes 会选择资源足够且负载最少的节点来运行 Pod。可以通过以下几种方式影响调度策略：

- **节点选择器（Node Selector）**：通过标签选择特定的节点来运行 Pod。
- **节点亲和性（Node Affinity）**：通过更复杂的规则指定 Pod 应该或必须调度到的节点。
- **Pod 亲和性与反亲和性（Pod Affinity & Anti-Affinity）**：控制 Pod 与其他 Pods 的调度关系。

#### 4.1.2 节点选择器

节点选择器是一种简单的调度策略，允许你指定 Pod 只能调度到具有特定标签的节点上。

**示例**：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: my-container
    image: nginx:latest
  nodeSelector:
    disktype: ssd
```

这个示例中的 Pod 只会调度到带有 `disktype=ssd` 标签的节点上。

#### 4.1.3 节点亲和性

节点亲和性提供了比节点选择器更灵活的调度策略。它允许你指定软性和硬性要求。

**示例**：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: my-container
    image: nginx:latest
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: disktype
            operator: In
            values:
            - ssd
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 1
        preference:
          matchExpressions:
          - key: region
            operator: In
            values:
            - us-east-1
```

在这个示例中，Pod 必须调度到 `disktype=ssd` 的节点上，并且优先选择 `region=us-east-1` 的节点。

#### 4.1.4 Pod 亲和性与反亲和性

Pod 亲和性用于将一个 Pod 调度到与某些指定 Pod 位于同一节点或不同节点上。

**示例**：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: my-container
    image: nginx:latest
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - my-app
        topologyKey: "kubernetes.io/hostname"
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - another-app
        topologyKey: "kubernetes.io/hostname"
```

这个示例表示 Pod 需要与 `app=my-app` 的 Pod 位于同一节点，但不与 `app=another-app` 的 Pod 位于同一节点。

### 4.2 Kubernetes 网络插件

Kubernetes 网络模型允许每个 Pod 拥有一个独立的 IP 地址，并且不同 Pod 之间可以相互通信。网络插件（CNI 插件）用于实现这种网络模型。以下是一些常见的网络插件：

- **Flannel**：一个简单的网络插件，适合小型集群，提供基本的网络功能。
- **Calico**：支持网络策略、BGP 路由等高级功能，适合大规模集群。
- **Weave**：支持跨数据中心的网络通信，适合分布式集群。
- **Cilium**：基于 eBPF 的网络插件，提供高性能的网络策略和安全性功能。

#### 4.2.1 Flannel

Flannel 是一个简单的覆盖网络插件，适合小型集群。它通过 VXLAN 技术实现跨节点的容器网络。

**安装 Flannel**：

```bash
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

#### 4.2.2 Calico

Calico 提供了强大的网络策略和安全功能，适合大规模集群。

**安装 Calico**：

```bash
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
```

### 4.3 Kubernetes 存储

Kubernetes 支持多种存储方案，满足不同应用的数据持久化需求。以下是一些常见的存储类型：

- **PersistentVolume（PV）和 PersistentVolumeClaim（PVC）**：PV 是集群管理员配置的存储资源，PVC 是用户申请使用这些资源的请求。PV 与 PVC 的绑定是持久存储的基础。
- **StorageClass**：用于定义动态创建 PV 的策略，支持不同的存储后端，如 NFS、Ceph、AWS EBS 等。

#### 4.3.1 PersistentVolume 和 PersistentVolumeClaim

PV 是集群中的实际存储资源，PVC 是用户对存储的请求。以下是一个创建 PV 和 PVC 的示例：

**PersistentVolume 示例**：

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  nfs:
    path: /mnt/data
    server: nfs-server.example.com
```

**PersistentVolumeClaim 示例**：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

#### 4.3.2 StorageClass

StorageClass 定义了动态创建 PV 的存储策略。

**StorageClass 示例**：

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
  fsType: ext4
```

### 4.4 Kubernetes 安全

安全性是 Kubernetes 中的一个重要方面，涉及到认证、授权、网络策略等多个层面。

#### 4.4.1 认证与授权

- **认证（Authentication）**：Kubernetes 支持多种认证方式，包括 X.509 客户端证书、Bearer Token、OpenID Connect 等。
- **授权（Authorization）**：Kubernetes 使用 RBAC（基于角色的访问控制）来管理对资源的访问权限。

**RBAC 示例**：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
```

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: default
subjects:
- kind: User
  name: "jane"
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

#### 4.4.2 网络策略

**NetworkPolicy** 用于控制 Pods 之间的网络流量。

**NetworkPolicy 示例**：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-http
spec:
  podSelector:
    matchLabels:
      app: my-app
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 80
```

### 4.5 Kubernetes 日志与监控

监控和日志管理是 Kubernetes 运营的重要部分。

#### 4.5.1 Prometheus 与 Grafana

**Prometheus** 是一个用于监控和报警的系统，**Grafana** 是一个用于数据可视化的工具。二者结合可实现强大的监控和可视化功能。

**Prometheus 部署示例**：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
          - role: endpoints
```

**Grafana 部署示例**
：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
```

#### 4.5.2 日志管理

Kubernetes 日志管理通常使用 ELK Stack（Elasticsearch, Logstash, Kibana）。它可以实现集中式日志管理和分析。

**Elasticsearch 部署示例**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elasticsearch
  namespace: logging
spec:
  replicas: 1
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:7.10.1
        env:
        - name: discovery.type
          value: single-node
        ports:
        - containerPort: 9200
```