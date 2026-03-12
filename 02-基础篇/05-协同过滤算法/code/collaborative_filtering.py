# 协同过滤算法示例代码

"""
协同过滤算法完整实现
包括：UserCF、ItemCF、Surprise 库使用
"""

import numpy as np
from collections import defaultdict


class UserCF:
    """基于用户的协同过滤"""
    
    def __init__(self, k=10):
        """
        Args:
            k: 近邻数量
        """
        self.k = k
        self.user_sim = {}
        self.user_means = {}
        
    def fit(self, ratings):
        """训练模型"""
        self.ratings = ratings
        n_users, n_items = ratings.shape
        
        # 计算用户平均评分
        for u in range(n_users):
            user_ratings = ratings[u][ratings[u] > 0]
            self.user_means[u] = np.mean(user_ratings) if len(user_ratings) > 0 else 0
        
        # 计算用户相似度矩阵
        self.user_sim = np.zeros((n_users, n_users))
        for i in range(n_users):
            for j in range(i, n_users):
                sim = self._pearson(i, j)
                self.user_sim[i, j] = sim
                self.user_sim[j, i] = sim
        
        return self
    
    def _pearson(self, u, v):
        """皮尔逊相似度"""
        mask = (self.ratings[u] > 0) & (self.ratings[v] > 0)
        if np.sum(mask) == 0:
            return 0
        
        ru = self.ratings[u][mask]
        rv = self.ratings[v][mask]
        
        mean_u = np.mean(ru)
        mean_v = np.mean(rv)
        
        numerator = np.sum((ru - mean_u) * (rv - mean_v))
        denominator = np.sqrt(np.sum((ru - mean_u) ** 2)) * np.sqrt(np.sum((rv - mean_v) ** 2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def predict(self, u, i):
        """预测评分"""
        if self.ratings[u, i] > 0:
            return self.ratings[u, i]
        
        rated_users = np.where(self.ratings[:, i] > 0)[0]
        if len(rated_users) == 0:
            return self.user_means[u]
        
        similarities = [(v, self.user_sim[u, v]) for v in rated_users]
        similarities.sort(key=lambda x: x[1], reverse=True)
        neighbors = similarities[:self.k]
        
        numerator = 0
        denominator = 0
        for v, sim in neighbors:
            if sim > 0:
                numerator += sim * (self.ratings[v, i] - self.user_means[v])
                denominator += abs(sim)
        
        if denominator == 0:
            return self.user_means[u]
        
        return self.user_means[u] + numerator / denominator
    
    def recommend(self, u, n=10):
        """推荐 Top-N 物品"""
        n_items = self.ratings.shape[1]
        predictions = []
        
        for i in range(n_items):
            if self.ratings[u, i] == 0:
                score = self.predict(u, i)
                predictions.append((i, score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class ItemCF:
    """基于物品的协同过滤"""
    
    def __init__(self, k=10):
        self.k = k
        self.item_sim = {}
        
    def fit(self, ratings):
        """训练模型"""
        self.ratings = ratings
        n_users, n_items = ratings.shape
        
        # 计算物品相似度矩阵
        self.item_sim = np.zeros((n_items, n_items))
        for i in range(n_items):
            for j in range(i, n_items):
                sim = self._cosine(i, j)
                self.item_sim[i, j] = sim
                self.item_sim[j, i] = sim
        
        return self
    
    def _cosine(self, i, j):
        """余弦相似度"""
        mask = (self.ratings[:, i] > 0) & (self.ratings[:, j] > 0)
        if np.sum(mask) == 0:
            return 0
        
        ri = self.ratings[mask, i]
        rj = self.ratings[mask, j]
        
        dot_product = np.dot(ri, rj)
        norm_i = np.linalg.norm(ri)
        norm_j = np.linalg.norm(rj)
        
        if norm_i == 0 or norm_j == 0:
            return 0
        
        return dot_product / (norm_i * norm_j)
    
    def predict(self, u, i):
        """预测评分"""
        if self.ratings[u, i] > 0:
            return self.ratings[u, i]
        
        rated_items = np.where(self.ratings[u] > 0)[0]
        if len(rated_items) == 0:
            return 0
        
        similarities = [(j, self.item_sim[i, j]) for j in rated_items]
        similarities.sort(key=lambda x: x[1], reverse=True)
        neighbors = similarities[:self.k]
        
        numerator = 0
        denominator = 0
        for j, sim in neighbors:
            if sim > 0:
                numerator += sim * self.ratings[u, j]
                denominator += abs(sim)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def recommend(self, u, n=10):
        """推荐 Top-N 物品"""
        n_items = self.ratings.shape[1]
        predictions = []
        
        for i in range(n_items):
            if self.ratings[u, i] == 0:
                score = self.predict(u, i)
                predictions.append((i, score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


def test_usercf():
    """测试 UserCF"""
    print("=" * 50)
    print("测试 UserCF 算法")
    print("=" * 50)
    
    ratings = np.array([
        [5, 3, 0, 4, 0, 2],
        [4, 0, 3, 5, 2, 0],
        [0, 4, 5, 0, 3, 4],
        [3, 0, 4, 0, 5, 0],
        [0, 5, 0, 3, 4, 5],
    ])
    
    model = UserCF(k=3)
    model.fit(ratings)
    
    user_id = 0
    recommendations = model.recommend(user_id, n=3)
    
    print(f"\n为用户 {user_id} 推荐的物品：")
    for item_id, score in recommendations:
        print(f"  物品 {item_id}: 预测评分 {score:.2f}")


def test_itemcf():
    """测试 ItemCF"""
    print("\n" + "=" * 50)
    print("测试 ItemCF 算法")
    print("=" * 50)
    
    ratings = np.array([
        [5, 3, 0, 4, 0, 2],
        [4, 0, 3, 5, 2, 0],
        [0, 4, 5, 0, 3, 4],
        [3, 0, 4, 0, 5, 0],
        [0, 5, 0, 3, 4, 5],
    ])
    
    model = ItemCF(k=3)
    model.fit(ratings)
    
    user_id = 0
    recommendations = model.recommend(user_id, n=3)
    
    print(f"\n为用户 {user_id} 推荐的物品：")
    for item_id, score in recommendations:
        print(f"  物品 {item_id}: 预测评分 {score:.2f}")


def test_surprise():
    """测试 Surprise 库"""
    print("\n" + "=" * 50)
    print("测试 Surprise 库")
    print("=" * 50)
    
    try:
        from surprise import Dataset, KNNBasic, train_test_split, accuracy
        
        # 加载数据
        data = Dataset.load_builtin('ml-100k')
        
        # 划分数据集
        trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
        
        # UserCF
        print("\nUserCF 评估：")
        sim_options = {'name': 'pearson', 'user_based': True}
        model = KNNBasic(sim_options=sim_options, k=40, min_k=5)
        model.fit(trainset)
        predictions = model.test(testset)
        accuracy.rmse(predictions)
        
        # ItemCF
        print("\nItemCF 评估：")
        sim_options = {'name': 'cosine', 'user_based': False}
        model = KNNBasic(sim_options=sim_options, k=40, min_k=5)
        model.fit(trainset)
        predictions = model.test(testset)
        accuracy.rmse(predictions)
        
    except ImportError:
        print("请先安装 Surprise: pip install scikit-surprise")


if __name__ == "__main__":
    test_usercf()
    test_itemcf()
    test_surprise()
