import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class ContentEncoder:
    """内容编码器 - 将文本内容转换为向量表示"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def encode_text(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        return self.model.encode([text])[0]
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        return self.model.encode(texts)
    
    def encode_tfidf(self, texts: List[str]) -> np.ndarray:
        """使用TF-IDF编码文本"""
        return self.tfidf.fit_transform(texts).toarray()

class DeepMatchingModel(nn.Module):
    """深度学习匹配模型"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(DeepMatchingModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class RecommendationEngine:
    """推荐引擎 - 包含召回、粗排、精排等模块"""
    
    def __init__(self, model_path: str = None):
        self.content_encoder = ContentEncoder()
        self.recall_model = None
        self.ranking_model = None
        self.final_ranking_model = None
        
        # 初始化模型
        self._init_models()
        
        # 存储专辑和广告的向量表示
        self.album_embeddings = {}
        self.ad_embeddings = {}
        
    def _init_models(self):
        """初始化各个模型"""
        # 召回模型 - 使用FAISS进行向量检索
        # 384维内容向量 + 384维标签向量 = 768维
        self.recall_model = faiss.IndexFlatIP(768)  # 768维向量
        
        # 粗排模型 - 随机森林
        self.ranking_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 精排模型 - 深度学习模型
        # 专辑768维 + 广告768维 = 1536维
        self.final_ranking_model = DeepMatchingModel(input_dim=1536)  # 1536维特征
        
    def add_album(self, album_id: str, content: str, tags: List[str], metadata: Dict):
        """添加专辑到推荐系统"""
        # 编码专辑内容
        content_embedding = self.content_encoder.encode_text(content)
        tags_text = ' '.join(tags)
        tags_embedding = self.content_encoder.encode_text(tags_text)
        
        # 合并特征
        combined_embedding = np.concatenate([content_embedding, tags_embedding])
        
        self.album_embeddings[album_id] = {
            'content_embedding': content_embedding,
            'tags_embedding': tags_embedding,
            'combined_embedding': combined_embedding,
            'metadata': metadata
        }
        
        # 添加到FAISS索引
        self.recall_model.add(combined_embedding.reshape(1, -1))
        
    def add_ad(self, ad_id: str, ad_content: str, ad_tags: List[str], ad_metadata: Dict):
        """添加广告到推荐系统"""
        # 编码广告内容
        ad_embedding = self.content_encoder.encode_text(ad_content)
        ad_tags_text = ' '.join(ad_tags)
        ad_tags_embedding = self.content_encoder.encode_text(ad_tags_text)
        
        # 合并特征
        combined_embedding = np.concatenate([ad_embedding, ad_tags_embedding])
        
        self.ad_embeddings[ad_id] = {
            'content_embedding': ad_embedding,
            'tags_embedding': ad_tags_embedding,
            'combined_embedding': combined_embedding,
            'metadata': ad_metadata
        }
        
    def recall_candidates(self, album_id: str, top_k: int = 100) -> List[str]:
        """召回阶段 - 基于向量相似度快速筛选候选广告"""
        if album_id not in self.album_embeddings:
            return []
            
        album_embedding = self.album_embeddings[album_id]['combined_embedding']
        
        # 使用FAISS进行向量检索
        scores, indices = self.recall_model.search(
            album_embedding.reshape(1, -1), top_k
        )
        
        # 返回候选广告ID列表
        ad_ids = list(self.ad_embeddings.keys())
        candidates = [ad_ids[i] for i in indices[0] if i < len(ad_ids)]
        
        return candidates[:top_k]
    
    def coarse_ranking(self, album_id: str, candidate_ads: List[str]) -> List[Tuple[str, float]]:
        """粗排阶段 - 基于规则和简单特征进行排序"""
        if not candidate_ads:
            return []
            
        album_info = self.album_embeddings[album_id]
        album_metadata = album_info['metadata']
        
        scores = []
        for ad_id in candidate_ads:
            ad_info = self.ad_embeddings[ad_id]
            ad_metadata = ad_info['metadata']
            
            # 计算匹配分数
            score = self._calculate_coarse_score(album_info, ad_info, album_metadata, ad_metadata)
            scores.append((ad_id, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _calculate_coarse_score(self, album_info: Dict, ad_info: Dict, 
                               album_metadata: Dict, ad_metadata: Dict) -> float:
        """计算粗排分数"""
        score = 0.0
        
        # 1. 内容相似度分数 (40%)
        content_sim = cosine_similarity(
            [album_info['content_embedding']], 
            [ad_info['content_embedding']]
        )[0][0]
        score += content_sim * 0.4
        
        # 2. 标签匹配分数 (30%)
        album_tags = set(album_metadata.get('tags', []))
        ad_tags = set(ad_metadata.get('tags', []))
        tag_overlap = len(album_tags & ad_tags) / max(len(album_tags | ad_tags), 1)
        score += tag_overlap * 0.3
        
        # 3. 受众匹配分数 (20%)
        if album_metadata.get('target_age') == ad_metadata.get('target_age'):
            score += 0.2
        
        # 4. 行业匹配分数 (10%)
        if album_metadata.get('industry') == ad_metadata.get('industry'):
            score += 0.1
            
        return score
    
    def fine_ranking(self, album_id: str, ranked_candidates: List[Tuple[str, float]], 
                    top_k: int = 10) -> List[Tuple[str, float]]:
        """精排阶段 - 使用深度学习模型进行精确排序"""
        if not ranked_candidates:
            return []
            
        # 选择前top_k个候选进行精排
        candidates = ranked_candidates[:top_k]
        
        # 构建特征向量
        features = []
        for ad_id, _ in candidates:
            album_info = self.album_embeddings[album_id]
            ad_info = self.ad_embeddings[ad_id]
            
            # 拼接专辑和广告的特征
            feature = np.concatenate([
                album_info['combined_embedding'],
                ad_info['combined_embedding']
            ])
            features.append(feature)
        
        features = np.array(features)
        
        # 使用深度学习模型预测分数
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            scores = self.final_ranking_model(features_tensor).numpy().flatten()
        
        # 重新排序
        final_scores = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores
    
    def recommend(self, album_id: str, top_k: int = 10) -> List[Dict]:
        """完整的推荐流程"""
        # 1. 召回阶段
        candidates = self.recall_candidates(album_id, top_k=100)
        
        # 2. 粗排阶段
        ranked_candidates = self.coarse_ranking(album_id, candidates)
        
        # 3. 精排阶段
        final_rankings = self.fine_ranking(album_id, ranked_candidates, top_k=top_k)
        
        # 4. 构建推荐结果
        recommendations = []
        for ad_id, score in final_rankings[:top_k]:
            ad_info = self.ad_embeddings[ad_id]
            recommendations.append({
                'ad_id': ad_id,
                'score': score,
                'metadata': ad_info['metadata'],
                'content_embedding': ad_info['content_embedding'].tolist()
            })
        
        return recommendations
    
    def save_models(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.recall_model, os.path.join(save_path, 'recall_model.faiss'))
        
        # 保存随机森林模型
        with open(os.path.join(save_path, 'ranking_model.pkl'), 'wb') as f:
            pickle.dump(self.ranking_model, f)
        
        # 保存深度学习模型
        torch.save(self.final_ranking_model.state_dict(), 
                  os.path.join(save_path, 'final_ranking_model.pth'))
        
        # 保存向量表示
        with open(os.path.join(save_path, 'embeddings.pkl'), 'wb') as f:
            pickle.dump({
                'album_embeddings': self.album_embeddings,
                'ad_embeddings': self.ad_embeddings
            }, f)
    
    def load_models(self, load_path: str):
        """加载模型"""
        # 加载FAISS索引
        self.recall_model = faiss.read_index(os.path.join(load_path, 'recall_model.faiss'))
        
        # 加载随机森林模型
        with open(os.path.join(load_path, 'ranking_model.pkl'), 'rb') as f:
            self.ranking_model = pickle.load(f)
        
        # 加载深度学习模型
        self.final_ranking_model.load_state_dict(
            torch.load(os.path.join(load_path, 'final_ranking_model.pth'))
        )
        
        # 加载向量表示
        with open(os.path.join(load_path, 'embeddings.pkl'), 'rb') as f:
            embeddings = pickle.load(f)
            self.album_embeddings = embeddings['album_embeddings']
            self.ad_embeddings = embeddings['ad_embeddings']
