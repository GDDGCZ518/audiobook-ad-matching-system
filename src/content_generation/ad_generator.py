import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llm_api import request_claude_api
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class AdGenerator:
    """广告生成器 - 根据专辑内容和广告信息生成广告语句"""
    
    def __init__(self):
        self.sentence_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.quality_metrics = {
            'relevance': 0.4,      # 相关性权重
            'creativity': 0.2,     # 创意性权重
            'persuasiveness': 0.2, # 说服力权重
            'readability': 0.1,    # 可读性权重
            'brand_consistency': 0.1  # 品牌一致性权重
        }
    
    def generate_ad_copy(self, album_info: Dict, ad_info: Dict, 
                        emotion_analysis: Dict = None) -> Dict:
        """
        生成广告文案
        
        Args:
            album_info: 专辑信息
            ad_info: 广告信息
            emotion_analysis: 情感分析结果
            
        Returns:
            生成的广告文案和元数据
        """
        # 构建生成提示词
        prompt = self._build_generation_prompt(album_info, ad_info, emotion_analysis)
        
        try:
            # 调用LLM API生成广告文案
            response = request_claude_api(prompt)
            
            # 解析生成结果
            ad_copy = self._parse_generation_response(response)
            
            # 评估生成质量
            quality_score = self._evaluate_quality(ad_copy, album_info, ad_info)
            
            return {
                'ad_copy': ad_copy,
                'quality_score': quality_score,
                'generation_metadata': {
                    'album_id': album_info.get('album_id'),
                    'ad_id': ad_info.get('ad_id'),
                    'emotion_analysis': emotion_analysis,
                    'generation_timestamp': self._get_timestamp()
                }
            }
            
        except Exception as e:
            print(f"广告生成失败: {e}")
            return self._get_default_ad_copy(album_info, ad_info)
    
    def _build_generation_prompt(self, album_info: Dict, ad_info: Dict, 
                                emotion_analysis: Dict = None) -> str:
        """构建广告生成提示词"""
        
        # 专辑信息
        album_content = album_info.get('content', '')[:500]
        album_tags = ', '.join(album_info.get('tags', []))
        album_theme = album_info.get('theme_category', '')
        
        # 广告信息
        ad_product = ad_info.get('product_name', '')
        ad_brand = ad_info.get('brand', '')
        ad_industry = ad_info.get('industry', '')
        ad_target = ad_info.get('target_audience', '')
        
        # 情感信息
        emotion_text = ""
        if emotion_analysis:
            emotions = ', '.join(emotion_analysis.get('emotion_categories', []))
            intensity = emotion_analysis.get('emotion_intensity', 5)
            emotion_text = f"专辑情感倾向: {emotions}, 情感强度: {intensity}/10"
        
        prompt = f"""
请为以下有声书专辑生成一段广告文案，要求：

1. 广告产品：{ad_product} ({ad_brand})
2. 产品行业：{ad_industry}
3. 目标受众：{ad_target}

专辑信息：
- 内容：{album_content}...
- 标签：{album_tags}
- 主题：{album_theme}
- {emotion_text}

要求：
1. 广告文案要与专辑内容高度相关
2. 语言要自然流畅，符合有声书听众的阅读习惯
3. 要有创意性和说服力
4. 长度控制在50-100字之间
5. 要体现品牌特色

请生成3个不同风格的广告文案，并返回JSON格式：

{{
    "ad_copies": [
        {{
            "style": "风格描述",
            "content": "广告文案内容",
            "target_emotion": "目标情感"
        }},
        {{
            "style": "风格描述", 
            "content": "广告文案内容",
            "target_emotion": "目标情感"
        }},
        {{
            "style": "风格描述",
            "content": "广告文案内容", 
            "target_emotion": "目标情感"
        }}
    ]
}}
"""
        return prompt
    
    def _parse_generation_response(self, response: str) -> List[Dict]:
        """解析LLM返回的广告文案"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('ad_copies', [])
            else:
                # 如果无法解析JSON，返回默认结果
                return self._get_default_ad_copies()
        except json.JSONDecodeError:
            return self._get_default_ad_copies()
    
    def _get_default_ad_copies(self) -> List[Dict]:
        """返回默认的广告文案"""
        return [
            {
                "style": "直接推荐",
                "content": "在您收听这本精彩有声书的同时，不妨了解一下我们的产品，为您的生活增添更多精彩。",
                "target_emotion": "中性"
            },
            {
                "style": "情感共鸣",
                "content": "正如这本书带给您的启发，我们的产品也能为您的每一天带来新的可能。",
                "target_emotion": "积极"
            },
            {
                "style": "价值主张",
                "content": "专注品质，用心服务，让我们的产品成为您生活中的得力助手。",
                "target_emotion": "信任"
            }
        ]
    
    def _evaluate_quality(self, ad_copies: List[Dict], 
                         album_info: Dict, ad_info: Dict) -> Dict:
        """评估广告文案质量"""
        quality_scores = {}
        
        for i, ad_copy in enumerate(ad_copies):
            score = self._calculate_quality_score(ad_copy, album_info, ad_info)
            quality_scores[f'ad_copy_{i+1}'] = score
        
        # 计算总体质量分数
        overall_score = np.mean(list(quality_scores.values()))
        
        return {
            'overall_score': overall_score,
            'individual_scores': quality_scores,
            'recommendations': self._generate_quality_recommendations(overall_score)
        }
    
    def _calculate_quality_score(self, ad_copy: Dict, 
                                album_info: Dict, ad_info: Dict) -> float:
        """计算单个广告文案的质量分数"""
        total_score = 0.0
        
        # 1. 相关性评分 (40%)
        relevance_score = self._calculate_relevance_score(ad_copy, album_info, ad_info)
        total_score += relevance_score * self.quality_metrics['relevance']
        
        # 2. 创意性评分 (20%)
        creativity_score = self._calculate_creativity_score(ad_copy)
        total_score += creativity_score * self.quality_metrics['creativity']
        
        # 3. 说服力评分 (20%)
        persuasiveness_score = self._calculate_persuasiveness_score(ad_copy)
        total_score += persuasiveness_score * self.quality_metrics['persuasiveness']
        
        # 4. 可读性评分 (10%)
        readability_score = self._calculate_readability_score(ad_copy)
        total_score += readability_score * self.quality_metrics['readability']
        
        # 5. 品牌一致性评分 (10%)
        brand_score = self._calculate_brand_consistency_score(ad_copy, ad_info)
        total_score += brand_score * self.quality_metrics['brand_consistency']
        
        return total_score
    
    def _calculate_relevance_score(self, ad_copy: Dict, 
                                  album_info: Dict, ad_info: Dict) -> float:
        """计算相关性分数"""
        # 使用语义相似度计算相关性
        album_content = album_info.get('content', '')[:200]
        ad_content = ad_copy.get('content', '')
        
        if not album_content or not ad_content:
            return 0.5
        
        # 编码文本
        album_embedding = self.sentence_encoder.encode([album_content])
        ad_embedding = self.sentence_encoder.encode([ad_content])
        
        # 计算余弦相似度
        similarity = cosine_similarity(album_embedding, ad_embedding)[0][0]
        
        return float(similarity)
    
    def _calculate_creativity_score(self, ad_copy: Dict) -> float:
        """计算创意性分数"""
        content = ad_copy.get('content', '')
        
        # 简单的创意性评估规则
        score = 0.5  # 基础分数
        
        # 检查是否包含创意元素
        creative_elements = ['比喻', '类比', '故事', '想象', '创新', '独特']
        for element in creative_elements:
            if element in content:
                score += 0.1
        
        # 检查句式多样性
        sentences = content.split('。')
        if len(sentences) > 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_persuasiveness_score(self, ad_copy: Dict) -> float:
        """计算说服力分数"""
        content = ad_copy.get('content', '')
        
        # 说服力元素检查
        persuasive_elements = ['推荐', '建议', '选择', '体验', '感受', '价值', '优势']
        score = 0.5
        
        for element in persuasive_elements:
            if element in content:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_readability_score(self, ad_copy: Dict) -> float:
        """计算可读性分数"""
        content = ad_copy.get('content', '')
        
        # 简单的可读性评估
        score = 1.0
        
        # 长度检查
        if len(content) < 20 or len(content) > 150:
            score -= 0.2
        
        # 标点符号检查
        if content.count('，') + content.count('。') < 2:
            score -= 0.1
        
        return max(score, 0.0)
    
    def _calculate_brand_consistency_score(self, ad_copy: Dict, ad_info: Dict) -> float:
        """计算品牌一致性分数"""
        content = ad_copy.get('content', '')
        brand = ad_info.get('brand', '')
        
        if not brand or brand not in content:
            return 0.3
        
        return 1.0
    
    def _generate_quality_recommendations(self, overall_score: float) -> List[str]:
        """根据质量分数生成改进建议"""
        recommendations = []
        
        if overall_score < 0.6:
            recommendations.append("建议重新生成广告文案，提高整体质量")
        elif overall_score < 0.8:
            recommendations.append("广告文案质量良好，可以进一步优化创意性")
        else:
            recommendations.append("广告文案质量优秀，可以直接使用")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def batch_generate(self, album_batch: List[Dict], 
                       ad_batch: List[Dict]) -> List[Dict]:
        """批量生成广告文案"""
        results = []
        
        for album in album_batch:
            for ad in ad_batch:
                # 这里可以添加匹配逻辑，只对匹配的专辑和广告生成文案
                result = self.generate_ad_copy(album, ad)
                results.append(result)
        
        return results
    
    def create_benchmark(self, generated_ads: List[Dict]) -> Dict:
        """创建广告质量评估基准"""
        if not generated_ads:
            return {}
        
        # 统计质量分数分布
        scores = [ad['quality_score']['overall_score'] for ad in generated_ads]
        
        benchmark = {
            'total_count': len(generated_ads),
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'quality_distribution': {
                'excellent': len([s for s in scores if s >= 0.8]),
                'good': len([s for s in scores if 0.6 <= s < 0.8]),
                'fair': len([s for s in scores if 0.4 <= s < 0.6]),
                'poor': len([s for s in scores if s < 0.4])
            },
            'recommendations': self._generate_benchmark_recommendations(scores)
        }
        
        return benchmark
    
    def _generate_benchmark_recommendations(self, scores: List[float]) -> List[str]:
        """根据基准分数生成改进建议"""
        recommendations = []
        
        if np.mean(scores) < 0.6:
            recommendations.append("整体广告质量偏低，建议优化生成策略")
        
        if np.std(scores) > 0.3:
            recommendations.append("广告质量差异较大，建议统一质量标准")
        
        if len([s for s in scores if s < 0.4]) > len(scores) * 0.2:
            recommendations.append("低质量广告比例过高，建议重新训练生成模型")
        
        return recommendations
