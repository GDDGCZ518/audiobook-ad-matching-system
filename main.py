#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有声书广告匹配Pipeline主程序
整合所有模块，提供完整的广告匹配和优化服务
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置
from config.config_loader import config_loader

# 导入各个模块
from src.content_understanding.emotion_analyzer import EmotionAnalyzer
from src.content_generation.ad_generator import AdGenerator
from src.matching_recommendation.recommendation_engine import RecommendationEngine
from src.deployment_optimization.rl_optimizer import DeploymentOptimizer
from src.monitoring_analysis.monitoring_dashboard import MonitoringDashboard, PerformanceMonitor
from src.monitoring_analysis.evaluation_metrics import EvaluationMetrics

class AudioBookAdPipeline:
    """有声书广告匹配Pipeline主类"""
    
    def __init__(self):
        """初始化Pipeline"""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 验证配置
        if not config_loader.validate_config():
            raise ValueError("配置验证失败")
        
        self.logger.info("初始化有声书广告匹配Pipeline...")
        
        # 初始化各个模块
        self.emotion_analyzer = EmotionAnalyzer()
        self.ad_generator = AdGenerator()
        self.recommendation_engine = RecommendationEngine()
        self.deployment_optimizer = DeploymentOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.evaluation_metrics = EvaluationMetrics()
        
        # 加载示例数据
        self.albums = self.load_sample_data('sample_albums.json')
        self.ads = self.load_sample_data('sample_ads.json')
        
        self.logger.info("Pipeline初始化完成")
    
    def setup_logging(self):
        """设置日志系统"""
        log_config = config_loader.get_config('logging')
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'logs/pipeline.log')
        
        # 创建日志目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_sample_data(self, filename: str) -> List[Dict]:
        """加载示例数据"""
        try:
            data_path = os.path.join('data', filename)
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"示例数据文件不存在: {data_path}")
                return []
        except Exception as e:
            self.logger.error(f"加载示例数据失败: {e}")
            return []
    
    def run_full_pipeline(self, album_id: str, ad_id: str, user_context: Dict = None) -> Dict:
        """运行完整的广告匹配Pipeline
        
        Args:
            album_id: 专辑ID
            ad_id: 广告ID
            user_context: 用户上下文信息
            
        Returns:
            Pipeline执行结果
        """
        self.logger.info(f"开始执行Pipeline: 专辑={album_id}, 广告={ad_id}")
        
        try:
            # 1. 获取专辑和广告信息
            album_info = self._get_album_info(album_id)
            ad_info = self._get_ad_info(ad_id)
            
            if not album_info or not ad_info:
                raise ValueError("专辑或广告信息不存在")
            
            # 2. 内容理解 - 情感分析
            self.logger.info("执行情感分析...")
            emotion_analysis = self.emotion_analyzer.analyze_emotion(
                album_info['content'], 
                album_info.get('tags', [])
            )
            
            # 3. 内容生成 - 广告文案生成
            self.logger.info("生成广告文案...")
            ad_generation_result = self.ad_generator.generate_ad_copy(
                album_info, ad_info, emotion_analysis
            )
            
            # 4. 匹配推荐 - 计算匹配度
            self.logger.info("计算匹配推荐...")
            # 添加专辑和广告到推荐系统
            self.recommendation_engine.add_album(
                album_id, 
                album_info['content'], 
                album_info.get('tags', []), 
                album_info
            )
            self.recommendation_engine.add_ad(
                ad_id, 
                ad_info['ad_content'], 
                ad_info.get('ad_tags', []), 
                ad_info
            )
            
            # 获取推荐结果
            recommendations = self.recommendation_engine.recommend(album_id, top_k=5)
            
            # 5. 投放优化 - 生成投放策略
            self.logger.info("生成投放策略...")
            # 模拟性能历史
            performance_history = self._generate_sample_performance_history()
            
            deployment_strategy = self.deployment_optimizer.optimize_deployment_strategy(
                album_info, ad_info, user_context or {}, performance_history
            )
            
            # 6. 整合结果
            result = {
                'pipeline_id': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'album_info': album_info,
                'ad_info': ad_info,
                'emotion_analysis': emotion_analysis,
                'ad_generation': ad_generation_result,
                'recommendations': recommendations,
                'deployment_strategy': deployment_strategy,
                'overall_score': self._calculate_overall_score(
                    ad_generation_result, recommendations, deployment_strategy
                )
            }
            
            self.logger.info(f"Pipeline执行完成，整体评分: {result['overall_score']:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline执行失败: {e}")
            raise
    
    def _get_album_info(self, album_id: str) -> Optional[Dict]:
        """获取专辑信息"""
        for album in self.albums:
            if album['album_id'] == album_id:
                return album
        return None
    
    def _get_ad_info(self, ad_id: str) -> Optional[Dict]:
        """获取广告信息"""
        for ad in self.ads:
            if ad['ad_id'] == ad_id:
                return ad
        return None
    
    def _generate_sample_performance_history(self) -> List[Dict]:
        """生成示例性能历史数据"""
        import random
        
        history = []
        for i in range(5):
            history.append({
                'click_rate': random.uniform(0.01, 0.05),
                'completion_rate': random.uniform(0.2, 0.5),
                'conversion_rate': random.uniform(0.005, 0.02),
                'revenue': random.uniform(100, 1000),
                'timestamp': datetime.now().isoformat()
            })
        return history
    
    def _calculate_overall_score(self, ad_generation: Dict, 
                                recommendations: List[Dict], 
                                deployment_strategy: Dict) -> float:
        """计算整体评分"""
        # 广告生成质量评分
        quality_info = ad_generation.get('quality_score', {})
        generation_score = quality_info.get('overall_score', 0.5)
        
        # 推荐匹配度评分
        if recommendations:
            recommendation_score = sum([rec.get('score', 0) for rec in recommendations]) / len(recommendations)
        else:
            recommendation_score = 0.0
        
        # 投放策略评分（简化计算）
        strategy_score = 0.5  # 这里可以根据策略的合理性计算
        
        # 加权平均
        overall_score = (
            generation_score * 0.4 + 
            recommendation_score * 0.4 + 
            strategy_score * 0.2
        )
        
        return overall_score
    
    def batch_process(self, album_ids: List[str], ad_ids: List[str]) -> List[Dict]:
        """批量处理多个专辑和广告的匹配"""
        results = []
        
        for album_id in album_ids:
            for ad_id in ad_ids:
                try:
                    result = self.run_full_pipeline(album_id, ad_id)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"处理失败: 专辑={album_id}, 广告={ad_id}, 错误={e}")
                    continue
        
        return results
    
    def start_monitoring(self):
        """启动性能监控"""
        self.logger.info("启动性能监控...")
        self.performance_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.logger.info("停止性能监控...")
        self.performance_monitor.stop_monitoring()
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        return self.performance_monitor.get_metrics_summary()
    
    def run_evaluation(self) -> Dict:
        """运行完整的系统评估"""
        self.logger.info("开始运行系统评估...")
        
        # 获取性能数据
        performance_summary = self.get_performance_summary()
        
        # 计算评估指标
        metrics_data = {
            "click_through_rate": performance_summary.get('click_rate', {}).get('current', 0),
            "conversion_rate": performance_summary.get('conversion_rate', {}).get('current', 0),
            "revenue_per_user": performance_summary.get('revenue', {}).get('current', 0),
            "response_time": performance_summary.get('response_time', {}).get('current', 0),
            "user_satisfaction": performance_summary.get('user_satisfaction', {}).get('current', 4.0)
        }
        
        # 生成评估报告
        evaluation_report = self.evaluation_metrics.generate_evaluation_report(metrics_data)
        
        # 保存评估报告
        report_path = f"logs/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.evaluation_metrics.save_evaluation_report(evaluation_report, report_path)
        
        self.logger.info("系统评估完成")
        return evaluation_report
    
    def save_pipeline_state(self, save_path: str = None):
        """保存Pipeline状态"""
        if save_path is None:
            save_path = config_loader.get_storage_path('models')
        
        try:
            # 保存推荐模型
            self.recommendation_engine.save_models(save_path)
            
            # 保存投放优化器
            self.deployment_optimizer.save_optimizer(save_path)
            
            self.logger.info(f"Pipeline状态已保存到: {save_path}")
        except Exception as e:
            self.logger.error(f"保存Pipeline状态失败: {e}")
    
    def load_pipeline_state(self, load_path: str = None):
        """加载Pipeline状态"""
        if load_path is None:
            load_path = config_loader.get_storage_path('models')
        
        try:
            # 加载推荐模型
            self.recommendation_engine.load_models(load_path)
            
            # 加载投放优化器
            self.deployment_optimizer.load_optimizer(load_path)
            
            self.logger.info(f"Pipeline状态已从以下路径加载: {load_path}")
        except Exception as e:
            self.logger.error(f"加载Pipeline状态失败: {e}")

def main():
    """主函数"""
    try:
        # 创建Pipeline实例
        pipeline = AudioBookAdPipeline()
        
        # 启动监控
        pipeline.start_monitoring()
        
        # 示例：运行单个Pipeline
        print("=== 运行单个Pipeline示例 ===")
        result = pipeline.run_full_pipeline("album_001", "ad_001")
        
        # 优化输出格式
        print(f"\n=== Pipeline执行结果 ===")
        print(f"专辑ID: {result['album_info']['album_id']}")
        print(f"专辑标题: {result['album_info']['title']}")
        print(f"广告ID: {result['ad_info']['ad_id']}")
        print(f"产品名称: {result['ad_info']['product_name']}")
        print(f"整体评分: {result['overall_score']:.3f}")
        print(f"推荐数量: {len(result['recommendations'])}")
        
        # 显示前3个推荐结果
        if result['recommendations']:
            print(f"\n前3个推荐结果:")
            for i, rec in enumerate(result['recommendations'][:3]):
                print(f"  {i+1}. 广告ID: {rec.get('ad_id', 'N/A')}, 匹配度: {rec.get('score', 0):.3f}")
        
        # 显示投放策略的关键指标
        print(f"\n投放策略:")
        print(f"  投放时机: {result['deployment_strategy'].get('timing', 'N/A')}")
        print(f"  投放频率: {result['deployment_strategy'].get('frequency', 'N/A')}次/天")
        print(f"  创意版本: {result['deployment_strategy'].get('creative_version', 'N/A')}")
        print(f"  目标强度: {result['deployment_strategy'].get('targeting_strength', 0):.3f}")
        print(f"  预算分配: {result['deployment_strategy'].get('budget_allocation', 0):.3f}")
        
        # 显示情感分析结果
        print(f"\n情感分析:")
        print(f"  主要情感: {result['emotion_analysis'].get('primary_emotion', 'N/A')}")
        print(f"  情感强度: {result['emotion_analysis'].get('emotion_intensity', 0):.3f}")
        
        print(f"\n详细结果已保存到日志文件")
        
        # 示例：批量处理
        print("\n=== 运行批量处理示例 ===")
        batch_results = pipeline.batch_process(
            ["album_001", "album_002"], 
            ["ad_001", "ad_002"]
        )
        print(f"批量处理完成，共处理 {len(batch_results)} 个任务")
        
        # 显示批量处理结果摘要
        print(f"\n批量处理结果摘要:")
        for i, result in enumerate(batch_results):
            print(f"  {i+1}. 专辑{result['album_info']['album_id']} + 广告{result['ad_info']['ad_id']}: 评分{result['overall_score']:.3f}")
        
        # 获取性能摘要
        print("\n=== 性能摘要 ===")
        performance_summary = pipeline.get_performance_summary()
        
        print("关键性能指标:")
        print(f"  点击率: {performance_summary['click_rate']['current']:.3f} (趋势: {performance_summary['click_rate']['trend']})")
        print(f"  完成率: {performance_summary['completion_rate']['current']:.3f} (趋势: {performance_summary['completion_rate']['trend']})")
        print(f"  转化率: {performance_summary['conversion_rate']['current']:.3f} (趋势: {performance_summary['conversion_rate']['trend']})")
        print(f"  收入: ${performance_summary['revenue']['current']:.2f} (趋势: {performance_summary['revenue']['trend']})")
        print(f"  响应时间: {performance_summary['response_time']['current']:.3f}s (趋势: {performance_summary['response_time']['trend']})")
        print(f"  错误率: {performance_summary['error_rate']['current']:.3f} (趋势: {performance_summary['error_rate']['trend']})")
        
        # 运行系统评估
        print("\n=== 系统评估 ===")
        evaluation_report = pipeline.run_evaluation()
        
        print(f"评估等级: {evaluation_report['summary']['overall_grade']}")
        print(f"总体评分: {evaluation_report['summary']['overall_score']:.2f}/4.0")
        print(f"优秀指标: {evaluation_report['summary']['excellent_count']}个")
        print(f"良好指标: {evaluation_report['summary']['good_count']}个")
        print(f"需改进指标: {evaluation_report['summary']['needs_improvement_count']}个")
        
        if evaluation_report['recommendations']:
            print(f"\n改进建议:")
            for i, rec in enumerate(evaluation_report['recommendations'][:3]):
                print(f"  {i+1}. {rec}")
        
        print(f"\n详细评估报告已保存到: logs/evaluation_report_*.json")
        
        # 保存状态
        pipeline.save_pipeline_state()
        
        # 停止监控
        pipeline.stop_monitoring()
        
        print("\n=== Pipeline执行完成 ===")
        
    except Exception as e:
        print(f"Pipeline执行失败: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
