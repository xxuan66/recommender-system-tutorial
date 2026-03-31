# 第十章：NeuralCF

> 用神经网络增强协同过滤的非线性建模能力

---

## 10.1 传统 CF 的局限性

### 10.1.1 线性假设问题

**传统矩阵分解的预测公式：**
```
r̂ui = μ + bu + bi + pu^T × qi
```

**问题：** 用户和物品的交互是简单的内积（线性）

**实际情况：** 用户和物品的交互可能是复杂的非线性关系

### 10.1.2 示例

```
传统 CF：
用户向量：[0.8, 0.2, 0.6]
物品向量：[0.9, 0.1, 0.8]
预测评分：0.8×0.9 + 0.2×0.1 + 0.6×0.8 = 1.22（线性组合）

实际情况：
- 用户可能特别喜欢"动作 + 科幻"的组合（非线性）
- 某些特征组合会产生协同效应
- 简单的内积无法捕捉复杂模式
```

### 10.1.3 NeuralCF 的核心思想

> 用神经网络替代内积，学习用户和物品的非线性交互

---

## 10.2 NeuralCF 架构

### 10.2.1 整体架构

```
┌─────────────────────────────────────────┐
│              输出层                      │
│           (预测评分)                      │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│              隐藏层                      │
│        (非线性交互建模)                    │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐       ┌───────▼───────┐
│   用户嵌入     │       │   物品嵌入     │
│   (User Embed)│       │   (Item Embed)│
└───────┬───────┘       └───────┬───────┘
        │                       │
    用户 ID                  物品 ID
```

### 10.2.2 数学表示

```
1. 嵌入层：
   pu = EmbeddingUser(u)
   qi = EmbeddingItem(i)

2. 拼接：
   z = [pu, qi]  # 拼接用户和物品向量

3. 隐藏层（可多层）：
   h1 = ReLU(W1 × z + b1)
   h2 = ReLU(W2 × h1 + b2)
   ...

4. 输出层：
   r̂ui = sigmoid(Wout × h_last + bout)
```

---

## 10.3 PyTorch 实现

### 10.3.1 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCF(nn.Module):
    """NeuralCF 模型"""
    
    def __init__(self, n_users, n_items, embedding_dim=32, hidden_dims=[64, 32]):
        """
        Args:
            n_users: 用户数量
            n_items: 物品数量
            embedding_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(NeuralCF, self).__init__()
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 初始化嵌入
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.1)
        
        # 隐藏层
        layers = []
        input_dim = embedding_dim * 2  # 用户 + 物品
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, user_ids, item_ids):
        """
        前向传播
        
        Args:
            user_ids: 用户 ID 张量 [batch_size]
            item_ids: 物品 ID 张量 [batch_size]
        
        Returns:
            预测评分 [batch_size]
        """
        # 嵌入
        user_emb = self.user_embedding(user_ids)  # [batch, emb_dim]
        item_emb = self.item_embedding(item_ids)  # [batch, emb_dim]
        
        # 拼接
        concat = torch.cat([user_emb, item_emb], dim=1)  # [batch, 2*emb_dim]
        
        # 隐藏层
        hidden = self.hidden_layers(concat)
        
        # 输出
        output = self.output_layer(hidden)
        
        # sigmoid 激活（评分 0-1）
        rating = torch.sigmoid(output).squeeze()
        
        return rating


# 创建模型
n_users = 1000
n_items = 500
model = NeuralCF(
    n_users=n_users,
    n_items=n_items,
    embedding_dim=32,
    hidden_dims=[64, 32]
)

print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
```

### 10.3.2 训练代码

```python
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RatingDataset(Dataset):
    """评分数据集"""
    
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


def train_neuralcf(model, ratings_matrix, epochs=50, batch_size=256, lr=0.001):
    """
    训练 NeuralCF 模型
    """
    # 准备数据
    users, items, ratings = [], [], []
    n_users, n_items = ratings_matrix.shape
    
    for u in range(n_users):
        for i in range(n_items):
            if ratings_matrix[u, i] > 0:
                users.append(u)
                items.append(i)
                # 归一化评分到 0-1
                ratings.append((ratings_matrix[u, i] - 0.5) / 4.5)
    
    # 创建 DataLoader
    dataset = RatingDataset(users, items, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch_users, batch_items, batch_ratings in dataloader:
            # 前向传播
            predictions = model(batch_users, batch_items)
            
            # 计算损失
            loss = criterion(predictions, batch_ratings)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    ratings = np.random.randint(1, 6, (100, 50)).astype(float)
    ratings[np.random.rand(100, 50) > 0.3] = 0  # 制造稀疏性
    
    # 创建模型
    model = NeuralCF(n_users=100, n_items=50, embedding_dim=16, hidden_dims=[32, 16])
    
    # 训练
    model = train_neuralcf(model, ratings, epochs=50, batch_size=64, lr=0.001)
    
    print("\n训练完成！")
    
    # 预测
    model.eval()
    with torch.no_grad():
        user = torch.LongTensor([0])
        item = torch.LongTensor([5])
        pred = model(user, item).item()
        # 反归一化
        pred_rating = pred * 4.5 + 0.5
        print(f"用户 0 对物品 5 的预测评分：{pred_rating:.2f}")
```

---

## 10.4 变体架构

### 10.4.1 NeuMF（Neural Matrix Factorization）

**结合 MF 和 NeuralCF：**

```python
class NeuMF(nn.Module):
    """NeuMF 模型 - 结合 MF 和 NeuralCF"""
    
    def __init__(self, n_users, n_items, mf_dim=16, neural_dim=32, hidden_dims=[64, 32]):
        super(NeuMF, self).__init__()
        
        # MF 部分
        self.mf_user_emb = nn.Embedding(n_users, mf_dim)
        self.mf_item_emb = nn.Embedding(n_items, mf_dim)
        
        # NeuralCF 部分
        self.ncf_user_emb = nn.Embedding(n_users, neural_dim)
        self.ncf_item_emb = nn.Embedding(n_items, neural_dim)
        
        # 隐藏层
        layers = []
        input_dim = mf_dim + neural_dim * 2  # MF + NCF
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(input_dim, 1)
    
    def forward(self, user_ids, item_ids):
        # MF 部分（内积）
        mf_user_emb = self.mf_user_emb(user_ids)
        mf_item_emb = self.mf_item_emb(item_ids)
        mf_vector = mf_user_emb * mf_item_emb  # 元素级乘积
        
        # NeuralCF 部分（拼接）
        ncf_user_emb = self.ncf_user_emb(user_ids)
        ncf_item_emb = self.ncf_item_emb(item_ids)
        ncf_vector = torch.cat([ncf_user_emb, ncf_item_emb], dim=1)
        
        # 融合
        concat = torch.cat([mf_vector, ncf_vector], dim=1)
        
        # 隐藏层
        hidden = self.hidden_layers(concat)
        
        # 输出
        output = self.output_layer(hidden)
        
        return torch.sigmoid(output).squeeze()
```

### 10.4.2 不同激活函数对比

```python
# ReLU（最常用）
nn.ReLU()

# Leaky ReLU（解决神经元死亡）
nn.LeakyReLU(negative_slope=0.01)

# GELU（Transformer 常用）
nn.GELU()

# Tanh
nn.Tanh()
```

---

## 10.5 使用 TensorFlow 实现

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class NeuralCF_TF(keras.Model):
    """TensorFlow 版 NeuralCF"""
    
    def __init__(self, n_users, n_items, embedding_dim=32, hidden_dims=[64, 32]):
        super(NeuralCF_TF, self).__init__()
        
        # 嵌入层
        self.user_embedding = layers.Embedding(
            input_dim=n_users,
            output_dim=embedding_dim,
            embeddings_initializer='normal'
        )
        self.item_embedding = layers.Embedding(
            input_dim=n_items,
            output_dim=embedding_dim,
            embeddings_initializer='normal'
        )
        
        # 隐藏层
        self.hidden_layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(
                layers.Dense(hidden_dim, activation='relu')
            )
            self.hidden_layers.append(layers.Dropout(0.1))
        
        # 输出层
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        user_ids, item_ids = inputs
        
        # 嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接
        concat = layers.concatenate([user_emb, item_emb])
        
        # 隐藏层
        x = concat
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # 输出
        output = self.output_layer(x)
        
        return tf.squeeze(output, axis=1)


# 创建和训练模型
model = NeuralCF_TF(n_users=1000, n_items=500, embedding_dim=32)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 训练
# model.fit([user_train, item_train], rating_train, epochs=50, batch_size=256)
```

---

## 10.6 调参技巧

### 10.6.1 超参数选择

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| **embedding_dim** | 16-128 | 数据量大用大维度 |
| **hidden_dims** | [64, 32] 或 [128, 64, 32] | 2-3 层足够 |
| **learning_rate** | 0.0001-0.01 | Adam 默认 0.001 |
| **batch_size** | 64-512 | GPU 大一些 |
| **dropout** | 0.1-0.3 | 防止过拟合 |

### 10.6.2 防止过拟合

```python
# 1. Dropout
layers.append(nn.Dropout(0.2))

# 2. L2 正则化
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 3. Early Stopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# 训练时
for epoch in range(epochs):
    # ... 训练 ...
    
    # 验证
    val_loss = evaluate(model, val_data)
    scheduler.step(val_loss)
```

---

## 10.7 与传统 CF 对比

| 维度 | 传统 CF | NeuralCF |
|------|--------|---------|
| **交互建模** | 线性（内积） | 非线性（神经网络） |
| **表达能力** | 有限 | 强 |
| **数据需求** | 少 | 多 |
| **训练速度** | 快 | 慢 |
| **可解释性** | 较好 | 较差 |
| **适用场景** | 小到中等数据 | 大数据集 |

---

## 10.8 本章小结

### 核心知识点

1. **传统 CF 局限** — 线性假设
2. **NeuralCF 架构** — 嵌入 + 隐藏层
3. **NeuMF 变体** — MF + NeuralCF 融合
4. **PyTorch 实现** — 完整训练流程
5. **调参技巧** — 防止过拟合

### 思考题

1. NeuralCF 相比传统 CF 提升了什么？
2. 为什么需要结合 MF 和 NeuralCF（NeuMF）？
3. 如何选择合适的嵌入维度和隐藏层？

### 下章预告

下一章我们将学习**Wide&Deep**模型，结合记忆和泛化能力。

---

## 参考资料

1. [Neural Collaborative Filtering 论文](https://arxiv.org/abs/1708.05031)
2. [NeuMF 论文](https://arxiv.org/abs/1708.05031)
3. [PyTorch 官方教程](https://pytorch.org/tutorials/)

---

**更新日期：** 2026-03-12  
**作者：** xuan
