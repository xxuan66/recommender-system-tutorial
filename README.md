# 🎯 推荐系统从入门到实战

> 系统化的推荐系统学习教程，涵盖基础理论、经典算法、深度学习、实战项目全流程

[![Stars](https://img.shields.io/github/stars/xxuan66/recommender-system-tutorial)](https://github.com/xxuan66/recommender-system-tutorial/stargazers)
[![Issues](https://img.shields.io/github/issues/xxuan66/recommender-system-tutorial)](https://github.com/xxuan66/recommender-system-tutorial/issues)
[![License](https://img.shields.io/github/license/xxuan66/recommender-system-tutorial)](https://github.com/xxuan66/recommender-system-tutorial/blob/main/LICENSE)

---

## 📚 学习路线

```
入门篇 → 基础篇 → 进阶篇 → 实战篇 → 高级篇
```

### 📖 第一阶段：入门篇（1-2 周）

**目标：** 建立推荐系统整体认知

| 章节 | 内容 | 预计时间 |
|------|------|---------|
| 01 | 什么是推荐系统 | 2h |
| 02 | 推荐系统应用场景 | 2h |
| 03 | 推荐系统评估指标 | 3h |
| 04 | 开发环境搭建 | 2h |

### 📖 第二阶段：基础篇（3-4 周）

**目标：** 掌握经典推荐算法

| 章节 | 内容 | 预计时间 |
|------|------|---------|
| 05 | 协同过滤算法 | 6h |
| 06 | 基于内容推荐 | 4h |
| 07 | 矩阵分解 | 6h |
| 08 | 隐语义模型 | 4h |
| 09 | 冷启动问题 | 3h |

### 📖 第三阶段：进阶篇（4-6 周）

**目标：** 深度学习推荐模型

| 章节 | 内容 | 预计时间 |
|------|------|---------|
| 10 | NeuralCF | 5h |
| 11 | Wide&Deep | 5h |
| 12 | DeepFM | 5h |
| 13 | DIN/DIEN | 6h |
| 14 | 多任务学习 | 5h |
| 15 | 序列推荐 | 5h |

### 📖 第四阶段：实战篇（4-6 周）

**目标：** 完整项目实战

| 项目 | 技术栈 | 难度 |
|------|--------|------|
| 电影推荐系统 | Surprise + Flask | ⭐⭐ |
| 电商商品推荐 | TensorFlow + Redis | ⭐⭐⭐ |
| 新闻推荐系统 | PyTorch + Kafka | ⭐⭐⭐ |
| 视频推荐系统 | DeepFM + Spark | ⭐⭐⭐⭐ |

### 📖 第五阶段：高级篇（持续学习）

**目标：** 前沿技术和优化

| 主题 | 内容 |
|------|------|
| 大规模推荐 | 分布式训练、在线学习 |
| 可解释推荐 | 注意力机制、知识图谱 |
| 强化学习推荐 | DQN、Policy Gradient |
| 联邦学习推荐 | 隐私保护推荐 |

---

## 📁 目录结构

```
recommender-system-tutorial/
├── README.md                      # 项目说明
├── 01-入门篇/
│   ├── 01-什么是推荐系统.md
│   ├── 02-应用场景.md
│   ├── 03-评估指标.md
│   └── 04-环境搭建.md
├── 02-基础篇/
│   ├── 05-协同过滤算法/
│   │   ├── UserCF.md
│   │   ├── ItemCF.md
│   │   └── code/
│   ├── 06-基于内容推荐/
│   ├── 07-矩阵分解/
│   └── 08-隐语义模型/
├── 03-进阶篇/
│   ├── 10-NeuralCF/
│   ├── 11-Wide&Deep/
│   ├── 12-DeepFM/
│   ├── 13-DIN-DIEN/
│   └── 14-多任务学习/
├── 04-实战篇/
│   ├── 电影推荐系统/
│   ├── 电商商品推荐/
│   ├── 新闻推荐系统/
│   └── 视频推荐系统/
├── 05-高级篇/
│   ├── 大规模推荐/
│   ├── 可解释推荐/
│   └── 强化学习推荐/
├── datasets/                      # 数据集
│   ├── MovieLens/
│   ├── Amazon/
│   └── Taobao/
├── code/                          # 通用代码
│   ├── utils/
│   ├── models/
│   └── eval/
└── resources/                     # 学习资源
    ├── papers/
    ├── books/
    └── courses/
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/xxuan66/recommender-system-tutorial.git
cd recommender-system-tutorial
```

### 2. 安装依赖

```bash
# 基础依赖
pip install numpy pandas matplotlib scikit-learn

# 推荐系统库
pip install surprise implicit lightfm

# 深度学习
pip install tensorflow torch torchvision

# 实战项目
pip install flask redis kafka-python
```

### 3. 开始学习

从 [01-入门篇](./01-入门篇/) 开始，按顺序学习。

---

## 📊 核心知识点

### 经典算法

| 算法 | 原理 | 适用场景 |
|------|------|---------|
| UserCF | 用户相似度 | 用户少、物品多 |
| ItemCF | 物品相似度 | 用户多、物品少 |
| Matrix Factorization | 矩阵分解 | 稀疏矩阵 |
| FM | 因子分解机 | 特征组合 |

### 深度学习模型

| 模型 | 特点 | 优势 |
|------|------|------|
| NeuralCF | 神经网络 +CF | 非线性建模 |
| Wide&Deep | 记忆 + 泛化 | 兼顾两者 |
| DeepFM | FM+Deep | 自动特征交叉 |
| DIN | 注意力机制 | 用户兴趣建模 |

### 评估指标

| 指标 | 说明 | 公式 |
|------|------|------|
| Precision | 查准率 | TP/(TP+FP) |
| Recall | 查全率 | TP/(TP+FN) |
| NDCG | 折损累积增益 | - |
| AUC | 曲线下面积 | - |
| MAE/RMSE | 预测误差 | - |

---

## 🎓 学习资源

### 书籍推荐

- 《推荐系统实践》- 项亮
- 《Recommender Systems Handbook》
- 《Deep Learning for Recommender Systems》

### 论文精选

- [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121)
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [DeepFM: A Factorization-Machine based Neural Network](https://arxiv.org/abs/1703.04247)
- [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

### 公开数据集

| 数据集 | 规模 | 类型 | 链接 |
|--------|------|------|------|
| MovieLens | 100K-25M | 电影评分 | [链接](https://grouplens.org/datasets/movielens/) |
| Amazon Review | 数百万 | 商品评论 | [链接](https://jmcauley.ucsd.edu/data/amazon/) |
| Taobao User Behavior | 千万级 | 用户行为 | [链接](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) |
| Netflix Prize | 1 亿 + | 电影评分 | [链接](https://www.kaggle.com/c/netflix-prize) |

### 工具框架

| 工具 | 用途 | 语言 |
|------|------|------|
| Surprise | 传统推荐算法 | Python |
| Implicit | 隐式反馈推荐 | Python |
| LightFM | 混合推荐 | Python |
| TensorFlow Recommenders | 深度学习推荐 | Python |
| RecBole | 统一推荐库 | Python |

---

## 💻 实战项目

### 项目 1：电影推荐系统（入门）

**技术栈：** Surprise + Flask + SQLite

**功能：**
- 用户注册登录
- 电影评分
- 个性化推荐
- 热门推荐

**输出：** Web 应用 + 源码 + 部署文档

### 项目 2：电商商品推荐（进阶）

**技术栈：** TensorFlow + Redis + MySQL

**功能：**
- 用户行为追踪
- 实时推荐
- 召回 + 排序
- A/B 测试

**输出：** 完整系统 + 性能优化方案

### 项目 3：新闻推荐系统（高级）

**技术栈：** PyTorch + Kafka + Elasticsearch

**功能：**
- 实时新闻流
- 用户兴趣建模
- 多样性推荐
- 冷启动处理

**输出：** 分布式系统 + 监控方案

---

## 📅 学习计划

### 3 个月速成计划

| 周数 | 内容 | 产出 |
|------|------|------|
| 1-2 | 入门篇 | 环境搭建、基础概念 |
| 3-6 | 基础篇 | 经典算法实现 |
| 7-10 | 进阶篇 | 深度学习模型 |
| 11-14 | 实战篇 | 1-2 个完整项目 |
| 15-16 | 复习优化 | 简历准备、面试 |

### 6 个月系统学习

| 阶段 | 时间 | 目标 |
|------|------|------|
| 基础 | 1-2 月 | 掌握经典算法 |
| 进阶 | 3-4 月 | 深度学习模型 |
| 实战 | 5 月 | 完整项目 |
| 提升 | 6 月 | 前沿技术、论文阅读 |

---

## 🤝 贡献指南

欢迎贡献内容！

### 贡献类型

- ✅ 修正错别字和错误
- ✅ 补充代码示例
- ✅ 添加新的算法教程
- ✅ 分享实战项目
- ✅ 翻译英文资料

### 提交流程

1. Fork 本仓库
2. 创建分支 `git checkout -b feature/your-feature`
3. 提交更改 `git commit -m 'Add some feature'`
4. 推送到分支 `git push origin feature/your-feature`
5. 提交 Pull Request

---

## 📝 更新日志

### 2026-03-12
- ✅ 创建仓库
- ✅ 完成整体框架设计
- ✅ 添加入门篇内容
- ✅ 添加基础篇内容

### TODO
- [ ] 完成入门篇详细教程
- [ ] 添加协同过滤代码实现
- [ ] 创建 MovieLens 数据集下载
- [ ] 搭建项目 1 框架

---

## 📧 联系方式

- **GitHub Issues:** [提问](https://github.com/xxuan66/recommender-system-tutorial/issues)
- **邮箱:** xxx@example.com
- **微信群:** 添加助手微信 xxx

---

## 📄 许可证

MIT License

---

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

**📢 欢迎分享给更多需要的朋友！**
