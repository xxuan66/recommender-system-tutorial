# 第十三章：DIN/DIEN

> 用注意力机制建模用户兴趣的演化

---

## 13.1 从静态到动态

### 13.1.1 用户兴趣的特点

**传统模型的假设：**
```
用户兴趣是静态的
```

**实际情况：**
```
用户兴趣是动态演化的
- 短期兴趣：最近浏览的商品
- 长期兴趣：一贯的偏好
- 兴趣转移：从母婴用品到玩具
```

### 13.1.2 序列推荐的发展

| 模型 | 年份 | 核心思想 |
|------|------|---------|
| **FPMC** | 2010 | 矩阵分解 + 马尔可夫链 |
| **GRU4Rec** | 2015 | RNN 建模序列 |
| **DIN** | 2018 | 注意力机制 |
| **DIEN** | 2019 | 兴趣演化网络 |
| **BST** | 2019 | Transformer |

---

## 13.2 DIN（Deep Interest Network）

### 13.2.1 核心思想

> 用注意力机制捕捉用户兴趣与候选物品的相关性

**直观理解：**
```
用户历史行为：[手机，电脑，耳机，书，衣服]
候选物品：键盘

注意力权重：
- 手机：0.2
- 电脑：0.5  ← 相关度高
- 耳机：0.2
- 书：0.05
- 衣服：0.05

用户表示 = 0.2×手机 + 0.5×电脑 + 0.2×耳机 + ...
```

### 13.2.2 DIN 架构

```
┌─────────────────────────────────────────────┐
│                  输出层                      │
│              (CTR 预测)                       │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐       ┌───────▼───────┐
│  用户兴趣层    │       │   物品嵌入     │
│  (注意力)      │       │   (候选)       │
└───────┬───────┘       └───────┬───────┘
        │                       │
┌───────▼───────────────────────▼───────┐
│           注意力计算                    │
│  Attention(历史行为，候选物品)           │
└───────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐       ┌───────▼───────┐
│  历史行为序列  │       │   其他特征     │
│ [item1, ...]  │       │  (用户/上下文)  │
└───────────────┘       └───────────────┘
```

### 13.2.3 注意力计算

**DIN 的注意力公式：**

```
Attention(Q, K, V) = Σ(ai × vi)

其中：
ai = exp(f(Q, ki)) / Σ(exp(f(Q, kj)))

f(Q, ki) = ReLU(W1 × [Q, ki, Q-ki, Q×ki] + b)
```

**说明：**
- Q: 候选物品（Query）
- K: 历史物品（Key）
- V: 历史物品表示（Value）
- [Q, ki, Q-ki, Q×ki]: 局部激活单元（Local Activation Unit）

### 13.2.4 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINAttention(nn.Module):
    """DIN 注意力层"""
    
    def __init__(self, embedding_dim, hidden_dim=32):
        super(DINAttention, self).__init__()
        
        # 局部激活单元
        # 输入：[Q, K, Q-K, Q*K] → 4 * embedding_dim
        self.attention_network = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )
    
    def forward(self, query, keys, keys_length):
        """
        计算注意力
        
        Args:
            query: 候选物品 [batch_size, embedding_dim]
            keys: 历史物品 [batch_size, seq_len, embedding_dim]
            keys_length: 每个用户的历史行为长度 [batch_size]
        
        Returns:
            用户兴趣表示 [batch_size, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = keys.shape
        
        # 扩展 query 以匹配 keys 的维度
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, emb]
        
        # 计算注意力输入特征 [Q, K, Q-K, Q*K]
        features = torch.cat([
            query_expanded,                    # Q
            keys,                              # K
            query_expanded - keys,             # Q - K
            query_expanded * keys,             # Q * K
        ], dim=-1)  # [batch, seq_len, 4*emb]
        
        # 计算注意力分数
        attention_scores = self.attention_network(features).squeeze(-1)  # [batch, seq_len]
        
        # Mask 填充位置
        mask = torch.arange(seq_len, device=keys.device).unsqueeze(0) < keys_length.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # 加权求和
        user_interest = torch.sum(attention_weights * keys, dim=1)  # [batch, emb]
        
        return user_interest


class DIN(nn.Module):
    """DIN 完整模型"""
    
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=16,
        history_max_len=50,
    ):
        super(DIN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.history_max_len = history_max_len
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 注意力层
        self.attention = DINAttention(embedding_dim, hidden_dim=32)
        
        # Deep 部分
        self.deep_network = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),  # user_emb + item_emb + interest
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, batch):
        """
        Args:
            batch: 字典，包含：
                - user_ids: [batch_size]
                - item_ids: [batch_size]
                - history_items: [batch_size, seq_len]
                - history_length: [batch_size]
        """
        # 嵌入
        user_emb = self.user_embedding(batch['user_ids'])  # [batch, emb]
        item_emb = self.item_embedding(batch['item_ids'])  # [batch, emb]
        
        history_emb = self.item_embedding(batch['history_items'])  # [batch, seq_len, emb]
        
        # 注意力计算用户兴趣
        user_interest = self.attention(
            item_emb,
            history_emb,
            batch['history_length']
        )  # [batch, emb]
        
        # 拼接特征
        concat = torch.cat([user_emb, item_emb, user_interest], dim=1)  # [batch, 3*emb]
        
        # Deep 网络
        deep_output = self.deep_network(concat)
        
        # 输出
        output = self.sigmoid(deep_output).squeeze()
        
        return output
```

---

## 13.3 DIEN（Deep Interest Evolution Network）

### 13.3.1 从 DIN 到 DIEN

**DIN 的局限：**
- 只考虑兴趣的相关性
- 忽略兴趣的演化过程

**DIEN 的改进：**
- 用 GRU 建模兴趣演化
- 引入注意力更新门（AUGRU）

### 13.3.2 DIEN 架构

```
用户行为序列 → Interest Extractor (GRU) → Interest Evolving (AUGRU) → 预测
     ↓                                          ↓
  嵌入层                                    候选物品（注意力）
```

### 13.3.3 Interest Extractor Layer

**用 GRU 提取兴趣序列：**

```python
class InterestExtractor(nn.Module):
    """兴趣提取层（GRU）"""
    
    def __init__(self, embedding_dim, hidden_dim=32):
        super(InterestExtractor, self).__init__()
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
    
    def forward(self, item_emb, mask):
        """
        Args:
            item_emb: 物品序列 [batch, seq_len, emb]
            mask: 掩码 [batch, seq_len]
        
        Returns:
            隐藏状态 [batch, seq_len, hidden]
        """
        # GRU
        output, _ = self.gru(item_emb)  # [batch, seq_len, hidden]
        
        # Mask（处理填充）
        mask = mask.unsqueeze(-1).float()
        output = output * mask
        
        return output
```

### 13.3.4 Interest Evolving Layer（AUGRU）

**注意力更新门 GRU：**

```python
class AUGRUCell(nn.Module):
    """AUGRU 单元"""
    
    def __init__(self, input_dim, hidden_dim):
        super(AUGRUCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU 门
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.new_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # 注意力更新门
        self.attention_gate = nn.Linear(input_dim + hidden_dim, 1)
    
    def forward(self, item_emb, target_item, prev_hidden):
        """
        Args:
            item_emb: 当前物品 [batch, input_dim]
            target_item: 候选物品 [batch, input_dim]
            prev_hidden: 上一时刻隐藏状态 [batch, hidden_dim]
        """
        # 拼接
        concat = torch.cat([item_emb, prev_hidden], dim=1)
        
        # GRU 门
        update_gate = torch.sigmoid(self.update_gate(concat))
        reset_gate = torch.sigmoid(self.reset_gate(concat))
        new_gate = torch.tanh(self.new_gate(torch.cat([item_emb, prev_hidden * reset_gate], dim=1)))
        
        # 注意力分数
        attention_input = torch.cat([item_emb, prev_hidden], dim=1)
        attention_score = torch.sigmoid(self.attention_gate(attention_input)).squeeze(-1)
        
        # 注意力更新门
        update_gate = update_gate * attention_score.unsqueeze(-1)
        
        # 新隐藏状态
        new_hidden = (1 - update_gate) * prev_hidden + update_gate * new_gate
        
        return new_hidden


class InterestEvolving(nn.Module):
    """兴趣演化层"""
    
    def __init__(self, input_dim, hidden_dim=32):
        super(InterestEvolving, self).__init__()
        
        self.augru_cell = AUGRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, interest_seq, target_item, seq_length):
        """
        Args:
            interest_seq: 兴趣序列 [batch, seq_len, input_dim]
            target_item: 候选物品 [batch, input_dim]
            seq_length: 序列长度 [batch]
        
        Returns:
            最终隐藏状态 [batch, hidden_dim]
        """
        batch_size, seq_len, _ = interest_seq.shape
        
        # 初始化隐藏状态
        hidden = torch.zeros(batch_size, self.hidden_dim, device=interest_seq.device)
        
        # 按时间步迭代
        for t in range(seq_len):
            item_emb = interest_seq[:, t, :]
            hidden = self.augru_cell(item_emb, target_item, hidden)
        
        return hidden
```

### 13.3.5 DIEN 完整模型

```python
class DIEN(nn.Module):
    """DIEN 完整模型"""
    
    def __init__(self, n_users, n_items, embedding_dim=16, hidden_dim=32):
        super(DIEN, self).__init__()
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 兴趣提取
        self.interest_extractor = InterestExtractor(embedding_dim, hidden_dim)
        
        # 兴趣演化
        self.interest_evolving = InterestEvolving(hidden_dim, hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        # 嵌入
        user_emb = self.user_embedding(batch['user_ids'])
        target_item = self.item_embedding(batch['item_ids'])
        history_emb = self.item_embedding(batch['history_items'])
        
        # 兴趣提取
        mask = (batch['history_items'] > 0).float()
        interest_seq = self.interest_extractor(history_emb, mask)
        
        # 兴趣演化
        final_interest = self.interest_evolving(
            interest_seq,
            target_item,
            batch['history_length']
        )
        
        # 拼接
        concat = torch.cat([user_emb, target_item, final_interest], dim=1)
        
        # 输出
        output = self.output_layer(concat).squeeze()
        
        return output
```

---

## 13.4 训练与评估

### 13.4.1 数据准备

```python
import numpy as np
from torch.utils.data import Dataset

class DIN_Dataset(Dataset):
    """DIN/DIEN 数据集"""
    
    def __init__(self, data, history_max_len=50):
        self.user_ids = torch.LongTensor(data['user_ids'])
        self.item_ids = torch.LongTensor(data['item_ids'])
        self.labels = torch.FloatTensor(data['labels'])
        
        # 处理历史行为（padding）
        history_items = data['history_items']
        history_length = data['history_length']
        
        # Padding
        padded_history = np.zeros((len(history_items), history_max_len), dtype=int)
        for i, (hist, length) in enumerate(zip(history_items, history_length)):
            actual_len = min(length, history_max_len)
            padded_history[i, :actual_len] = hist[:actual_len]
        
        self.history_items = torch.LongTensor(padded_history)
        self.history_length = torch.LongTensor([min(l, history_max_len) for l in history_length])
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_ids': self.user_ids[idx],
            'item_ids': self.item_ids[idx],
            'history_items': self.history_items[idx],
            'history_length': self.history_length[idx],
            'labels': self.labels[idx],
        }
```

### 13.4.2 训练代码

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

def train_din(model, train_data, val_data, epochs=20, batch_size=256, lr=0.001):
    """训练 DIN/DIEN 模型"""
    
    train_dataset = DIN_Dataset(train_data)
    val_dataset = DIN_Dataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_auc = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # 将 batch 移到 GPU（如果使用）
            # batch = {k: v.cuda() for k, v in batch.items()}
            
            preds = model(batch)
            loss = criterion(preds, batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch['labels'].numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_dien.pth')
            print(f"  ✓ 保存最佳模型 (AUC: {val_auc:.4f})")
    
    return model
```

---

## 13.5 DIN vs DIEN 对比

| 维度 | DIN | DIEN |
|------|-----|------|
| **核心机制** | 注意力 | GRU + AUGRU |
| **兴趣建模** | 静态加权 | 动态演化 |
| **计算效率** | 高（并行） | 低（序列） |
| **适用场景** | 通用推荐 | 强序列依赖 |
| **参数量** | 较少 | 较多 |

---

## 13.6 本章小结

### 核心知识点

1. **DIN 注意力** — 局部激活单元
2. **DIEN 演化** — GRU + AUGRU
3. **序列建模** — 用户行为序列
4. **兴趣表示** — 动态加权
5. **CTR 预测** — 电商场景

### 思考题

1. DIN 的注意力机制与 Transformer 的注意力有什么区别？
2. 为什么 DIEN 要用 AUGRU 而不是普通 GRU？
3. 如何处理超长行为序列？

### 下章预告

下一章我们将学习**多任务学习**，同时优化多个目标（点击、购买、时长）。

---

## 参考资料

1. [DIN 论文](https://arxiv.org/abs/1706.06978)
2. [DIEN 论文](https://arxiv.org/abs/1809.03672)
3. [阿里巴巴推荐系统实践](https://github.com/alibaba/DeepRec)

---

**更新日期：** 2026-03-12  
**作者：** OpenClaw
