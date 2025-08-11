import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llm_api import request_claude_api
from typing import Dict, List, Tuple
import re

class EmotionAnalyzer:
    """情感分析器 - 分析专辑内容的情感倾向"""
    
    def __init__(self):
        self.emotion_categories = [
            "积极", "消极", "中性", "兴奋", "平静", "紧张", "放松", "悲伤", "快乐"
        ]
    
    def analyze_emotion(self, content: str, tags: List[str] = None) -> Dict:
        """
        分析内容的情感倾向
        
        Args:
            content: 专辑文本内容
            tags: 专辑标签信息
            
        Returns:
            情感分析结果字典
        """
        # 构建分析提示词
        prompt = self._build_emotion_prompt(content, tags)
        
        try:
            # 调用LLM API进行情感分析
            response = request_claude_api(prompt)
            
            # 解析响应结果
            result = self._parse_emotion_response(response)
            return result
            
        except Exception as e:
            print(f"情感分析失败: {e}")
            return self._get_default_result()
    
    def _build_emotion_prompt(self, content: str, tags: List[str] = None) -> str:
        """构建情感分析提示词"""
        prompt = f"""
请分析以下有声书专辑内容的情感倾向，并返回JSON格式的结果。

专辑内容: {content[:1000]}...

专辑标签: {', '.join(tags) if tags else '无'}

请分析以下方面：
1. 主要情感倾向（从以下选项中选择1-3个）：{', '.join(self.emotion_categories)}
2. 情感强度（1-10分，10分最强）
3. 情感变化趋势（稳定/波动/渐进）
4. 适合的受众群体
5. 情感关键词（3-5个）

请以JSON格式返回，格式如下：
{{
    "emotion_categories": ["情感类别"],
    "emotion_intensity": 8,
    "emotion_trend": "情感趋势",
    "target_audience": "目标受众",
    "emotion_keywords": ["关键词1", "关键词2", "关键词3"],
    "confidence": 0.95
}}
"""
        return prompt
    
    def _parse_emotion_response(self, response: str) -> Dict:
        """解析LLM返回的情感分析结果"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # 如果无法解析JSON，返回默认结果
                return self._get_default_result()
        except json.JSONDecodeError:
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict:
        """返回默认的情感分析结果"""
        return {
            "emotion_categories": ["中性"],
            "emotion_intensity": 5,
            "emotion_trend": "稳定",
            "target_audience": "一般受众",
            "emotion_keywords": ["内容", "信息", "知识"],
            "confidence": 0.5
        }
    
    def extract_key_information(self, content: str) -> Dict:
        """
        提取内容中的关键信息
        
        Args:
            content: 专辑文本内容
            
        Returns:
            关键信息字典
        """
        prompt = f"""
请从以下有声书专辑内容中提取关键信息，返回JSON格式结果。

内容: {content[:1500]}...

请提取以下信息：
1. 主题类别（如：历史、科技、文学、商业等）
2. 核心概念（3-5个）
3. 关键人物或事件
4. 时间背景
5. 地理背景
6. 内容难度等级（初级/中级/高级）
7. 预计受众年龄范围
8. 内容特色亮点

请以JSON格式返回。
"""
        
        try:
            response = request_claude_api(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_default_key_info()
        except Exception as e:
            print(f"关键信息提取失败: {e}")
            return self._get_default_key_info()
    
    def _get_default_key_info(self) -> Dict:
        """返回默认的关键信息"""
        return {
            "theme_category": "未分类",
            "core_concepts": ["概念1", "概念2"],
            "key_figures": [],
            "time_background": "现代",
            "geographic_background": "通用",
            "difficulty_level": "中级",
            "target_age": "18-60岁",
            "highlights": ["内容丰富", "知识性强"]
        }
