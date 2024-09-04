# scikit-learn

### Scikit-learn 教程详解

`Scikit-learn` 是一个广泛使用的 Python 库，提供了各种机器学习算法的实现。它包括用于分类、回归、聚类、降维、模型选择和预处理的工具。`Scikit-learn` 简单易用，非常适合机器学习的快速原型开发和探索性数据分析。

#### 1. 安装 Scikit-learn

在开始使用 `Scikit-learn` 之前，你需要安装它：

```bash
pip install scikit-learn
```

#### 2. Scikit-learn 基本概念

`Scikit-learn` 主要围绕以下几个核心概念展开：

- **Estimator（估计器）**: 机器学习模型，所有算法在 `Scikit-learn` 中都实现为一个估计器。
- **Predictor（预测器）**: 具有 `predict` 方法的估计器，用于预测数据的输出。
- **Transformer（转换器）**: 具有 `transform` 方法的估计器，用于转换数据。
- **Pipeline（管道）**: 用于将多个估计器串联在一起，便于组合模型的构建和测试。

#### 3. 数据集加载与预处理

`Scikit-learn` 提供了几个内置的数据集，可以直接使用，也可以从外部加载数据。

```python
from sklearn.datasets import load_iris
import pandas as pd

# 加载 Iris 数据集
data = load_iris()

# 将数据集转换为 DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
```

##### 3.1 数据预处理

数据预处理通常是机器学习工作流的重要一步。`Scikit-learn` 提供了许多预处理工具。

###### 3.1.1 标准化数据

使用 `StandardScaler` 进行数据标准化：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 拟合并转换数据
scaled_data = scaler.fit_transform(df.drop('target', axis=1))

print(scaled_data[:5])
```

###### 3.1.2 独热编码

使用 `OneHotEncoder` 进行分类变量的独热编码：

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)

# 将目标变量进行独热编码
encoded_target = encoder.fit_transform(df[['target']])

print(encoded_target[:5])
```

#### 4. 分类与回归模型

`Scikit-learn` 提供了多种分类和回归模型，以下是常用的几个模型。

##### 4.1 逻辑回归

逻辑回归是二元分类的常用方法。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(scaled_data, df['target'], test_size=0.3, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

##### 4.2 支持向量机

支持向量机（SVM）是另一种常见的分类方法。

```python
from sklearn.svm import SVC

# 创建支持向量机模型
svc_model = SVC()

# 训练模型
svc_model.fit(X_train, y_train)

# 预测
y_pred_svc = svc_model.predict(X_test)

# 评估模型
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f'SVM Accuracy: {accuracy_svc}')
```

##### 4.3 随机森林

随机森林是一种集成学习方法，通过多个决策树的组合来提高模型的准确性。

```python
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测
y_pred_rf = rf_model.predict(X_test)

# 评估模型
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')
```

#### 5. 聚类模型

`Scikit-learn` 提供了几种聚类算法，用于在无监督学习中发现数据的自然分组。

##### 5.1 K-means 聚类

K-means 是最常用的聚类算法之一。

```python
from sklearn.cluster import KMeans

# 创建 K-means 模型
kmeans = KMeans(n_clusters=3, random_state=42)

# 拟合模型
kmeans.fit(scaled_data)

# 获取聚类结果
clusters = kmeans.labels_
print(clusters)
```

#### 6. 模型评估

`Scikit-learn` 提供了多种模型评估方法，包括交叉验证、混淆矩阵、ROC 曲线等。

##### 6.1 交叉验证

交叉验证用于评估模型的泛化能力。

```python
from sklearn.model_selection import cross_val_score

# 交叉验证逻辑回归模型
cv_scores = cross_val_score(model, scaled_data, df['target'], cv=5)

print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')
```

##### 6.2 混淆矩阵

混淆矩阵用于评估分类模型的性能。

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 显示混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

##### 6.3 ROC 曲线

ROC 曲线和 AUC 值是评估二元分类模型的重要指标。

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

#### 7. 管道与模型选择

`Pipeline` 是 `Scikit-learn` 中的一个重要工具，允许将多个处理步骤串联起来。

##### 7.1 创建管道

```python
from sklearn.pipeline import Pipeline

# 创建一个包含标准化和逻辑回归的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

# 训练管道
pipeline.fit(X_train, y_train)

# 预测
y_pred_pipeline = pipeline.predict(X_test)
print(f'Pipeline Accuracy: {accuracy_score(y_test, y_pred_pipeline)}')
```

##### 7.2 网格搜索

`GridSearchCV` 用于在参数网格上进行搜索，以找到最佳的模型超参数。

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'logreg__C': [0.1, 1.0, 10.0],
    'logreg__solver': ['liblinear', 'lbfgs']
}

# 创建网格搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# 运行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f'Best Parameters: {grid_search.best_params_}')
```

#### 8. 总结

`Scikit-learn` 是一个功能强大的机器学习库，提供了从数据预处理、模型训练到模型评估的全流程工具。通过掌握 `Scikit-learn`，你可以有效地构建、评估和优化机器学习模型，解决各种实际问题。