# 第十一章：Wide&Deep

> 结合记忆能力和泛化能力的经典深度学习推荐模型

---

## 11.1 记忆与泛化

### 11.1.1 两种能力

推荐系统需要同时具备两种能力：

| 能力 | 说明 | 例子 |
|------|------|------|
| **记忆（Memorization）** | 学习历史共现模式 | "买了 A 的人常买 B" |
| **泛化（Generalization）** | 发现新的潜在模式 | "喜欢科幻的人也喜欢这个" |

### 11.1.2 各自的优缺点

**记忆能力（Wide 部分）：**
- ✅ 精确捕捉历史模式
- ✅ 可解释性强
- ❌ 无法处理未见过的组合
- ❌ 需要大量特征工程

**泛化能力（Deep 部分）：**
- ✅ 可以发现新关联
- ✅ 减少特征工程
- ❌ 可能过度泛化
- ❌ 可解释性差

### 11.1.3 Wide&Deep 的核心思想

> 将 Wide 模型（记忆）和 Deep 模型（泛化）结合，兼顾两者优势

---

## 11.2 Wide&Deep 架构

### 11.2.1 整体架构

```
┌─────────────────────────────────────────────┐
│                  输出层                      │
│              (CTR 预测)                       │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐       ┌───────▼───────┐
│   Wide 部分    │       │   Deep 部分    │
│  (线性模型)    │       │ (神经网络)     │
└───────┬───────┘       └───────┬───────┘
        │                       │
  ┌─────┴─────┐           ┌─────┴─────┐
  │           │           │           │
原始特征    交叉特征    稠密特征    嵌入特征
(用户/物品)  (人工设计)  (连续值)   (类别特征)
```

### 11.2.2 数学表示

```
Wide 部分：
y_wide = w^T × [x, φ(x)] + b

其中：
- x: 原始特征
- φ(x): 交叉特征（人工设计）
- w: 权重

Deep 部分：
a^(0) = [embeddings of categorical features] + [continuous features]
a^(l+1) = f(W^(l) × a^(l) + b^(l))  # ReLU 激活
y_deep = sigmoid(W^out × a^(last) + b^out)

最终输出：
P(CTR) = sigmoid(y_wide + y_deep)
```

---

## 11.3 PyTorch 实现

### 11.3.1 模型定义

```python
import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    """Wide&Deep 模型"""
    
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=16,
        wide_features=100,  # Wide 部分特征维度
        deep_hidden_dims=[32, 16],
    ):
        super(WideAndDeep, self).__init__()
        
        # ========== Wide 部分 ==========
        # 嵌入交叉特征
        self.user_age_emb = nn.Embedding(10, embedding_dim)  # 年龄段
        self.item_category_emb = nn.Embedding(50, embedding_dim)  # 物品类别
        
        # Wide 线性层
        self.wide_linear = nn.Linear(wide_features + embedding_dim * 2, 1)
        
        # ========== Deep 部分 ==========
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 深度网络
        layers = []
        input_dim = embedding_dim * 2  # user + item
        
        for hidden_dim in deep_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        self.deep_layers = nn.Sequential(*layers)
        
        # Deep 输出层
        self.deep_output = nn.Linear(input_dim, 1)
        
        # ========== 最终输出 ==========
        self.activation = nn.Sigmoid()
    
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 字典，包含：
                - user_ids: 用户 ID
                - item_ids: 物品 ID
                - user_age: 用户年龄（wide 特征）
                - item_category: 物品类别（wide 特征）
                - wide_features: 其他 wide 特征
        """
        # ----- Wide 部分 -----
        user_age_emb = self.user_age_emb(batch['user_age'])
        item_cat_emb = self.item_category_emb(batch['item_category'])
        
        wide_input = torch.cat([
            batch['wide_features'],
            user_age_emb,
            item_cat_emb
        ], dim=1)
        
        wide_output = self.wide_linear(wide_input)
        
        # ----- Deep 部分 -----
        user_emb = self.user_embedding(batch['user_ids'])
        item_emb = self.item_embedding(batch['item_ids'])
        
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        deep_hidden = self.deep_layers(deep_input)
        deep_output = self.deep_output(deep_hidden)
        
        # ----- 融合 -----
        combined = wide_output + deep_output
        output = self.activation(combined)
        
        return output.squeeze()
```

### 11.3.2 训练代码

```python
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WideDeepDataset(Dataset):
    """Wide&Deep 数据集"""
    
    def __init__(self, data):
        self.user_ids = torch.LongTensor(data['user_ids'])
        self.item_ids = torch.LongTensor(data['item_ids'])
        self.user_age = torch.LongTensor(data['user_age'])
        self.item_category = torch.LongTensor(data['item_category'])
        self.wide_features = torch.FloatTensor(data['wide_features'])
        self.labels = torch.FloatTensor(data['labels'])  # 0/1（点击/未点击）
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_ids': self.user_ids[idx],
            'item_ids': self.item_ids[idx],
            'user_age': self.user_age[idx],
            'item_category': self.item_category[idx],
            'wide_features': self.wide_features[idx],
            'labels': self.labels[idx],
        }


def train_wide_deep(model, train_data, val_data, epochs=20, batch_size=256, lr=0.001):
    """训练 Wide&Deep 模型"""
    
    # 创建 DataLoader
    train_dataset = WideDeepDataset(train_data)
    val_dataset = WideDeepDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    best_val_auc = 0
    
    for epoch in range(epochs):
        # ----- 训练 -----
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            # 前向传播
            preds = model(batch)
            
            # 计算损失
            loss = criterion(preds, batch['labels'])
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(preds.cpu().detach().numpy())
            train_labels.extend(batch['labels'].numpy())
        
        # ----- 验证 -----
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch)
                val_preds.extend(preds.cpu().detach().numpy())
                val_labels.extend(batch['labels'].numpy())
        
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
            torch.save(model.state_dict(), 'best_wide_deep.pth')
            print(f"  ✓ 保存最佳模型 (AUC: {val_auc:.4f})")
    
    return model


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    n_samples = 10000
    
    train_data = {
        'user_ids': np.random.randint(0, 1000, n_samples),
        'item_ids': np.random.randint(0, 500, n_samples),
        'user_age': np.random.randint(0, 10, n_samples),  # 年龄段 0-9
        'item_category': np.random.randint(0, 50, n_samples),  # 类别 0-49
        'wide_features': np.random.rand(n_samples, 100),  # 100 维 wide 特征
        'labels': np.random.randint(0, 2, n_samples).astype(float),  # 点击/未点击
    }
    
    val_data = {
        'user_ids': np.random.randint(0, 1000, 2000),
        'item_ids': np.random.randint(0, 500, 2000),
        'user_age': np.random.randint(0, 10, 2000),
        'item_category': np.random.randint(0, 50, 2000),
        'wide_features': np.random.rand(2000, 100),
        'labels': np.random.randint(0, 2, 2000).astype(float),
    }
    
    # 创建模型
    model = WideAndDeep(
        n_users=1000,
        n_items=500,
        embedding_dim=16,
        wide_features=100,
        deep_hidden_dims=[32, 16]
    )
    
    # 训练
    model = train_wide_deep(model, train_data, val_data, epochs=20, batch_size=256)
    
    print(f"\n最佳验证 AUC: {best_val_auc:.4f}")
```

---

## 11.4 特征工程

### 11.4.1 Wide 部分特征

**Wide 部分需要人工设计交叉特征：**

```python
def create_wide_features(user_info, item_info, interaction):
    """
    创建 Wide 部分特征
    """
    features = []
    
    # 1. 原始特征
    features.append(user_info['age'])
    features.append(user_info['gender'])
    features.append(item_info['price'])
    features.append(item_info['category'])
    
    # 2. 交叉特征（关键！）
    
    # 年龄段 × 物品类别
    age_category_cross = user_info['age'] * 100 + item_info['category']
    features.append(age_category_cross)
    
    # 性别 × 价格区间
    price_range = int(item_info['price'] / 100)
    gender_price_cross = user_info['gender'] * 10 + price_range
    features.append(gender_price_cross)
    
    # 用户历史行为 × 物品特征
    if 'user_history_categories' in user_info:
        history_match = item_info['category'] in user_info['user_history_categories']
        features.append(int(history_match))
    
    # 3. 统计特征
    features.append(item_info['avg_rating'])
    features.append(item_info['click_count_7d'])
    
    return np.array(features)
```

### 11.4.2 Deep 部分特征

**Deep 部分自动学习特征组合：**

```python
def create_deep_features(user_id, item_id, context):
    """
    Deep 部分只需要 ID 类特征（会自动嵌入）
    """
    return {
        'user_ids': [user_id],
        'item_ids': [item_id],
        # 其他类别特征也可以加入
        'user_city': [context['city']],
        'item_brand': [context['brand']],
    }
```

---

## 11.5 实战：点击率预测

### 11.5.1 完整流程

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 加载数据
df = pd.read_csv('user_item_interactions.csv')

# 2. 特征工程
# 编码类别特征
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user_id_enc'] = user_encoder.fit_transform(df['user_id'])
df['item_id_enc'] = item_encoder.fit_transform(df['item_id'])
df['age_group'] = (df['user_age'] // 10).clip(0, 9)  # 年龄段 0-9

# 3. 准备 Wide 特征
wide_feature_cols = ['price', 'avg_rating', 'click_count_7d']
df[wide_feature_cols] = df[wide_feature_cols].fillna(0)

# 4. 划分数据集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. 准备训练数据
train_data = {
    'user_ids': train_df['user_id_enc'].values,
    'item_ids': train_df['item_id_enc'].values,
    'user_age': train_df['age_group'].values,
    'item_category': train_df['item_category'].values,
    'wide_features': train_df[wide_feature_cols].values,
    'labels': train_df['clicked'].values,  # 0/1
}

val_data = {
    'user_ids': val_df['user_id_enc'].values,
    'item_ids': val_df['item_id_enc'].values,
    'user_age': val_df['age_group'].values,
    'item_category': val_df['item_category'].values,
    'wide_features': val_df[wide_feature_cols].values,
    'labels': val_df['clicked'].values,
}

# 6. 训练模型
model = WideAndDeep(
    n_users=len(user_encoder),
    n_items=len(item_encoder),
    embedding_dim=16,
    wide_features=len(wide_feature_cols),
    deep_hidden_dims=[32, 16]
)

model = train_wide_deep(model, train_data, val_data, epochs=20)
```

### 11.5.2 模型评估

```python
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

def evaluate_model(model, test_data):
    """评估模型"""
    model.eval()
    
    dataset = WideDeepDataset(test_data)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            preds = model(batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    
    # 转换为 0/1 预测
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, preds_binary)
    
    print(f"AUC: {auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return {
        'auc': auc,
        'logloss': logloss,
        'accuracy': accuracy,
    }

# 评估
metrics = evaluate_model(model, test_data)
```

---

## 11.6 与 NeuralCF 对比

| 维度 | NeuralCF | Wide&Deep |
|------|---------|-----------|
| **架构** | 纯 Deep | Wide + Deep |
| **特征** | 只需 ID | 需要人工设计 Wide 特征 |
| **记忆能力** | 弱 | 强（Wide 部分） |
| **泛化能力** | 强 | 强（Deep 部分） |
| **特征工程** | 少 | 多 |
| **适用场景** | 通用推荐 | CTR 预测、广告 |

---

## 11.7 本章小结

### 核心知识点

1. **记忆与泛化** — 推荐系统的两种能力
2. **Wide&Deep 架构** — Wide 线性 + Deep 神经
3. **特征工程** — Wide 部分需要交叉特征
4. **CTR 预测** — 典型应用场景
5. **模型评估** — AUC、Log Loss

### 思考题

1. Wide&Deep 相比 NeuralCF 有什么优势？
2. 如何设计有效的交叉特征？
3. Wide 部分和 Deep 部分的权重如何平衡？

### 下章预告

下一章我们将学习**DeepFM**，自动学习特征交叉，减少人工特征工程。

---

## 参考资料

1. [Wide & Deep Learning 论文](https://arxiv.org/abs/1606.07792)
2. [Google Wide & Deep 教程](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
3. [PyTorch 官方文档](https://pytorch.org/docs/)

---

**更新日期：** 2026-03-12  
**作者：** xuan
