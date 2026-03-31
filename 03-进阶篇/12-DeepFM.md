# 第十二章：DeepFM

> 在 Wide&Deep 之后，再往前一步理解“自动特征交叉”为什么重要

---

## 12.1 从 Wide&Deep 到 DeepFM

### 12.1.1 Wide&Deep 的局限性

**Wide&Deep 虽然很强，但也有下面这些问题：**

| 问题 | 说明 | 影响 |
|------|------|------|
| **依赖人工特征** | Wide 部分需要手动设计交叉特征 | 特征工程成本高 |
| **特征组合有限** | 只能设计低阶交叉（2-3 阶） | 无法捕捉高阶交互 |
| **可扩展性差** | 新特征需要重新设计 | 维护成本高 |

### 12.1.2 DeepFM 的核心思想

> 用 FM（Factorization Machine）替代 Wide 部分，自动学习特征交叉

**DeepFM = FM（低阶交叉） + Deep（高阶交叉）**

```
Wide&Deep:  人工交叉特征 + Deep 神经网络
     ↓
DeepFM:   FM 自动交叉 + Deep 神经网络
```

---

## 12.2 FM（因子分解机）

### 12.2.1 FM 原理

**FM 的预测公式：**

```
y(x) = w0 + Σ(wi × xi) + ΣΣ(vi^T × vj) × xi × xj

其中：
- w0: 全局偏置
- wi × xi: 一阶项（线性）
- vi^T × vj × xi × xj: 二阶交叉项
- vi: 特征 i 的隐向量
```

**简化理解：**
```
FM = 线性部分 + 二阶交叉部分
```

### 12.2.2 FM 的优势

**相比传统线性模型：**

| 特性 | 线性模型 | FM |
|------|---------|-----|
| **特征交叉** | 无 | 自动学习 |
| **稀疏数据** | 效果差 | 效果好 |
| **参数数量** | O(n) | O(kn)，k 是隐向量维度 |

### 12.2.3 FM 的代码实现

```python
import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    """因子分解机（FM）"""
    
    def __init__(self, n_features, k=4):
        """
        Args:
            n_features: 特征数量
            k: 隐向量维度
        """
        super(FactorizationMachine, self).__init__()
        
        # 偏置
        self.w0 = nn.Parameter(torch.zeros(1))
        
        # 一阶权重
        self.w = nn.Parameter(torch.zeros(n_features))
        
        # 二阶隐向量
        self.V = nn.Parameter(torch.zeros(n_features, k))
        
        # 初始化
        nn.init.normal_(self.w0, mean=0, std=0.1)
        nn.init.normal_(self.w, mean=0, std=0.1)
        nn.init.normal_(self.V, mean=0, std=0.1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, n_features]
        
        Returns:
            预测值 [batch_size]
        """
        # 一阶部分
        linear = self.w0 + torch.sum(x * self.w, dim=1)
        
        # 二阶部分（高效计算）
        # 公式：ΣΣ(vi^T × vj) × xi × xj = 0.5 × [(Σvi×xi)^2 - Σ(vi×xi)^2]
        xv = torch.mm(x, self.V)  # [batch, k]
        x2v = torch.pow(xv, 2)    # [batch, k]
        
        sum_xv_sq = torch.pow(torch.sum(xv, dim=1), 2)  # (Σvi×xi)^2
        sum_xv2 = torch.sum(x2v, dim=1)                  # Σ(vi×xi)^2
        
        second_order = 0.5 * (sum_xv_sq - sum_xv2)
        
        # 最终输出
        output = linear + second_order
        
        return output


# 使用示例
fm = FactorizationMachine(n_features=100, k=4)
x = torch.randn(32, 100)  # batch_size=32
output = fm(x)
print(f"FM 输出形状：{output.shape}")
```

---

## 12.3 DeepFM 架构

### 12.3.1 整体架构

```
┌─────────────────────────────────────────────┐
│                  输出层                      │
│              (sigmoid)                       │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐       ┌───────▼───────┐
│     FM 部分    │       │   Deep 部分    │
│  (二阶交叉)    │       │ (高阶交叉)     │
└───────┬───────┘       └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    │
            ┌───────▼───────┐
            │   嵌入层       │
            │ (共享嵌入)     │
            └───────┬───────┘
                    │
            原始特征（稀疏）
```

### 12.3.2 关键特点

| 特点 | 说明 |
|------|------|
| **共享嵌入** | FM 和 Deep 共享相同的嵌入层 |
| **端到端** | 无需人工特征工程 |
| **低阶 + 高阶** | FM 学二阶，Deep 学高阶 |
| **联合训练** | 两部分同时训练 |

---

## 12.4 DeepFM 实现

### 12.4.1 完整模型

```python
import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """DeepFM 模型"""
    
    def __init__(
        self,
        field_dims,      # 每个 field 的维度（类别数）
        embedding_dim=16,
        deep_hidden_dims=[32, 16],
    ):
        """
        Args:
            field_dims: 每个特征的类别数列表
                       例如 [1000, 500, 10, 50] 表示 4 个 field
            embedding_dim: 嵌入维度
            deep_hidden_dims: Deep 部分隐藏层维度
        """
        super(DeepFM, self).__init__()
        
        self.n_fields = len(field_dims)
        self.embedding_dim = embedding_dim
        
        # ========== 嵌入层（FM 和 Deep 共享） ==========
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in field_dims
        ])
        
        # 初始化
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.1)
        
        # ========== FM 部分 ==========
        self.fm_w0 = nn.Parameter(torch.zeros(1))  # 偏置
        self.fm_w = nn.Parameter(torch.zeros(self.n_fields))  # 一阶权重
        
        # ========== Deep 部分 ==========
        layers = []
        input_dim = self.n_fields * embedding_dim  # 所有嵌入拼接
        
        for hidden_dim in deep_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        self.deep_network = nn.Sequential(*layers)
        
        # Deep 输出层
        self.deep_output = nn.Linear(input_dim, 1)
        
        # ========== 最终输出 ==========
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, n_fields]
               每个 field 是类别索引
        
        Returns:
            预测值 [batch_size]
        """
        batch_size = x.shape[0]
        
        # ========== 嵌入层 ==========
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb = emb_layer(x[:, i])  # [batch, emb_dim]
            emb_list.append(emb)
        
        # ========== FM 部分 ==========
        # 一阶
        fm_linear = self.fm_w0 + torch.sum(self.fm_w.unsqueeze(0) * x.float(), dim=1)
        
        # 二阶（使用嵌入向量）
        stacked_emb = torch.stack(emb_list, dim=1)  # [batch, n_fields, emb_dim]
        
        # sum(xv)
        sum_xv = torch.sum(stacked_emb, dim=1)  # [batch, emb_dim]
        sum_xv_sq = torch.pow(sum_xv, 2)
        
        # sum(xv^2)
        sum_xv2 = torch.sum(torch.pow(stacked_emb, 2), dim=1)
        
        fm_second_order = 0.5 * torch.sum(sum_xv_sq - sum_xv2, dim=1)
        
        fm_output = fm_linear + fm_second_order
        
        # ========== Deep 部分 ==========
        # 拼接所有嵌入
        deep_input = torch.cat(emb_list, dim=1)  # [batch, n_fields*emb_dim]
        
        deep_hidden = self.deep_network(deep_input)
        deep_output = self.deep_output(deep_hidden).squeeze()
        
        # ========== 融合 ==========
        combined = fm_output + deep_output
        output = self.sigmoid(combined)
        
        return output


# 创建模型
field_dims = [1000, 500, 10, 50, 100]  # 5 个特征 field
model = DeepFM(field_dims=field_dims, embedding_dim=16, deep_hidden_dims=[32, 16])

print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
```

### 12.4.2 训练代码

```python
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DeepFMDataset(Dataset):
    """DeepFM 数据集"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: 特征矩阵 [n_samples, n_fields]
            labels: 标签 [n_samples]
        """
        self.features = torch.LongTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_deepfm(model, train_data, val_data, epochs=20, batch_size=256, lr=0.001):
    """训练 DeepFM 模型"""
    
    # 创建 DataLoader
    train_dataset = DeepFMDataset(train_data['features'], train_data['labels'])
    val_dataset = DeepFMDataset(val_data['features'], val_data['labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 训练循环
    best_val_auc = 0
    
    for epoch in range(epochs):
        # ----- 训练 -----
        model.train()
        train_loss = 0
        
        for features, labels in train_loader:
            # 前向传播
            preds = model(features)
            
            # 计算损失
            loss = criterion(preds, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # ----- 验证 -----
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                preds = model(features)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        # 计算 AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # 打印进度
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_deepfm.pth')
            print(f"  ✓ 保存最佳模型 (AUC: {val_auc:.4f})")
    
    return model


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    n_samples = 10000
    n_fields = 5
    field_dims = [1000, 500, 10, 50, 100]
    
    # 生成特征（每个 field 是类别索引）
    features = np.zeros((n_samples, n_fields), dtype=int)
    for i, dim in enumerate(field_dims):
        features[:, i] = np.random.randint(0, dim, n_samples)
    
    # 生成标签
    labels = np.random.randint(0, 2, n_samples).astype(float)
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    train_data = {'features': train_features, 'labels': train_labels}
    val_data = {'features': val_features, 'labels': val_labels}
    
    # 创建模型
    model = DeepFM(field_dims=field_dims, embedding_dim=16, deep_hidden_dims=[32, 16])
    
    # 训练
    model = train_deepfm(model, train_data, val_data, epochs=20, batch_size=256)
    
    print(f"\n最佳验证 AUC: {best_val_auc:.4f}")
```

---

## 12.5 与 Wide&Deep 对比

### 12.5.1 架构对比

| 维度 | Wide&Deep | DeepFM |
|------|-----------|--------|
| **Wide/FM 部分** | 线性 + 人工交叉 | FM 自动二阶交叉 |
| **特征工程** | 需要人工设计 | 无需人工 |
| **嵌入共享** | 否 | 是 |
| **高阶交叉** | Deep 部分 | Deep 部分 |

### 12.5.2 性能对比

**在相同数据集上的典型表现：**

| 指标 | Wide&Deep | DeepFM | 提升 |
|------|-----------|--------|------|
| **AUC** | 0.78 | 0.80 | +2.6% |
| **特征工程时间** | 2 周 | 2 天 | -86% |
| **模型参数量** | 较多 | 较少 | -30% |

---

## 12.6 实战：广告 CTR 预测

### 12.6.1 数据准备

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 加载数据
df = pd.read_csv('ad_clicks.csv')

# 特征字段
categorical_features = [
    'user_id',
    'ad_id',
    'user_age_group',
    'user_gender',
    'ad_category',
    'advertiser_id',
    'context_hour',
]

# 编码类别特征
encoders = {}
for feat in categorical_features:
    le = LabelEncoder()
    df[feat + '_enc'] = le.fit_transform(df[feat].fillna('NA').astype(str))
    encoders[feat] = le

# 准备特征和标签
features = df[[feat + '_enc' for feat in categorical_features]].values
labels = df['clicked'].values

# 计算每个 field 的维度
field_dims = [len(encoders[feat].classes_) for feat in categorical_features]

print(f"特征维度：{field_dims}")
print(f"样本数量：{len(features)}")
```

### 12.6.2 训练和评估

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

# 划分数据集
train_features, val_features, train_labels, val_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

train_data = {'features': train_features, 'labels': train_labels}
val_data = {'features': val_features, 'labels': val_labels}

# 创建模型
model = DeepFM(
    field_dims=field_dims,
    embedding_dim=16,
    deep_hidden_dims=[128, 64, 32]
)

# 训练
model = train_deepfm(model, train_data, val_data, epochs=20, batch_size=1024, lr=0.001)

# 测试集评估
test_features, test_labels = ...  # 加载测试集
test_dataset = DeepFMDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

model.eval()
test_preds = []
test_labels_list = []

with torch.no_grad():
    for features, labels in test_loader:
        preds = model(features)
        test_preds.extend(preds.cpu().numpy())
        test_labels_list.extend(labels.numpy())

# 计算指标
test_auc = roc_auc_score(test_labels_list, test_preds)
test_logloss = log_loss(test_labels_list, test_preds)

print(f"测试集 AUC: {test_auc:.4f}")
print(f"测试集 Log Loss: {test_logloss:.4f}")
```

---

## 12.7 调参建议

### 12.7.1 关键参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| **embedding_dim** | 8-64 | 特征多用小维度 |
| **deep_hidden_dims** | [128, 64, 32] | 3 层足够 |
| **learning_rate** | 0.0005-0.002 | Adam 默认 0.001 |
| **batch_size** | 256-2048 | 大数据集用大 batch |
| **dropout** | 0.1-0.3 | 防止过拟合 |

### 12.7.2 防止过拟合

```python
# 1. Dropout（已在模型中）
nn.Dropout(0.2)

# 2. L2 正则化
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 3. Batch Normalization（已在模型中）
nn.BatchNorm1d(hidden_dim)

# 4. Early Stopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# 训练时
scheduler.step(val_auc)
```

---

## 12.8 本章小结

### 核心知识点

1. **FM 原理** — 自动学习二阶特征交叉
2. **DeepFM 架构** — FM + Deep，共享嵌入
3. **端到端训练** — 无需人工特征工程
4. **CTR 预测** — 典型应用场景
5. **调参技巧** — 防止过拟合

### 思考题

1. DeepFM 相比 Wide&Deep 的主要改进是什么？
2. 为什么 FM 和 Deep 可以共享嵌入层？
3. 如何处理连续特征？

### 下章预告

下一章我们将学习**DIN/DIEN**，引入注意力机制建模用户兴趣演化。

---

## 参考资料

1. [DeepFM 论文](https://arxiv.org/abs/1703.04247)
2. [Factorization Machines 论文](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
3. [DeepFM 代码实现](https://github.com/ChenglongChen/tensorflow-DeepFM)

---

**更新日期：** 2026-03-12  
**作者：** xuan
