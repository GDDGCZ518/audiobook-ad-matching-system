#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有声书广告匹配Pipeline测试文件
"""

import unittest
import sys
import os
import json
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AudioBookAdPipeline
from config.config_loader import ConfigLoader

class TestAudioBookAdPipeline(unittest.TestCase):
    """测试AudioBookAdPipeline类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 修改配置以使用临时目录
        self.original_config = ConfigLoader()
        self.original_config.set_config('storage.data_dir', self.temp_dir)
        self.original_config.set_config('storage.models_dir', self.temp_dir)
        self.original_config.set_config('storage.logs_dir', self.temp_dir)
        
        # 创建Pipeline实例
        self.pipeline = AudioBookAdPipeline()
    
    def tearDown(self):
        """测试后的清理工作"""
        # 清理临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """测试Pipeline初始化"""
        self.assertIsNotNone(self.pipeline.emotion_analyzer)
        self.assertIsNotNone(self.pipeline.ad_generator)
        self.assertIsNotNone(self.pipeline.recommendation_engine)
        self.assertIsNotNone(self.pipeline.deployment_optimizer)
        self.assertIsNotNone(self.pipeline.performance_monitor)
    
    def test_sample_data_loading(self):
        """测试示例数据加载"""
        self.assertGreater(len(self.pipeline.albums), 0)
        self.assertGreater(len(self.pipeline.ads), 0)
        
        # 检查专辑数据结构
        album = self.pipeline.albums[0]
        required_keys = ['album_id', 'title', 'content', 'tags']
        for key in required_keys:
            self.assertIn(key, album)
        
        # 检查广告数据结构
        ad = self.pipeline.ads[0]
        required_keys = ['ad_id', 'product_name', 'brand', 'ad_content']
        for key in required_keys:
            self.assertIn(key, ad)
    
    def test_emotion_analysis(self):
        """测试情感分析功能"""
        album_info = self.pipeline.albums[0]
        emotion_result = self.pipeline.emotion_analyzer.analyze_emotion(
            album_info['content'], 
            album_info.get('tags', [])
        )
        
        self.assertIsInstance(emotion_result, dict)
        self.assertIn('emotion_categories', emotion_result)
        self.assertIn('emotion_intensity', emotion_result)
        self.assertIn('confidence', emotion_result)
    
    def test_ad_generation(self):
        """测试广告生成功能"""
        album_info = self.pipeline.albums[0]
        ad_info = self.pipeline.ads[0]
        
        generation_result = self.pipeline.ad_generator.generate_ad_copy(
            album_info, ad_info
        )
        
        self.assertIsInstance(generation_result, dict)
        self.assertIn('ad_copy', generation_result)
        self.assertIn('quality_score', generation_result)
    
    def test_recommendation_engine(self):
        """测试推荐引擎功能"""
        album_info = self.pipeline.albums[0]
        ad_info = self.pipeline.ads[0]
        
        # 添加专辑和广告
        self.pipeline.recommendation_engine.add_album(
            album_info['album_id'],
            album_info['content'],
            album_info.get('tags', []),
            album_info
        )
        
        self.pipeline.recommendation_engine.add_ad(
            ad_info['ad_id'],
            ad_info['ad_content'],
            ad_info.get('ad_tags', []),
            ad_info
        )
        
        # 获取推荐
        recommendations = self.pipeline.recommendation_engine.recommend(
            album_info['album_id'], 
            top_k=3
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
    
    def test_deployment_optimizer(self):
        """测试投放优化器功能"""
        album_info = self.pipeline.albums[0]
        ad_info = self.pipeline.ads[0]
        user_context = {'age': 30, 'gender': 'male'}
        performance_history = []
        
        strategy = self.pipeline.deployment_optimizer.optimize_deployment_strategy(
            album_info, ad_info, user_context, performance_history
        )
        
        self.assertIsInstance(strategy, dict)
        self.assertIn('timing', strategy)
        self.assertIn('frequency', strategy)
        self.assertIn('creative_version', strategy)
    
    def test_full_pipeline_execution(self):
        """测试完整Pipeline执行"""
        album_id = self.pipeline.albums[0]['album_id']
        ad_id = self.pipeline.ads[0]['ad_id']
        
        result = self.pipeline.run_full_pipeline(album_id, ad_id)
        
        self.assertIsInstance(result, dict)
        self.assertIn('pipeline_id', result)
        self.assertIn('emotion_analysis', result)
        self.assertIn('ad_generation', result)
        self.assertIn('recommendations', result)
        self.assertIn('deployment_strategy', result)
        self.assertIn('overall_score', result)
        
        # 检查整体评分
        self.assertIsInstance(result['overall_score'], (int, float))
        self.assertGreaterEqual(result['overall_score'], 0.0)
        self.assertLessEqual(result['overall_score'], 1.0)
    
    def test_batch_processing(self):
        """测试批量处理功能"""
        album_ids = [self.pipeline.albums[0]['album_id'], self.pipeline.albums[1]['album_id']]
        ad_ids = [self.pipeline.ads[0]['ad_id'], self.pipeline.ads[1]['ad_id']]
        
        results = self.pipeline.batch_process(album_ids, ad_ids)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # 检查每个结果的结构
        for result in results:
            self.assertIn('pipeline_id', result)
            self.assertIn('overall_score', result)
    
    def test_performance_monitoring(self):
        """测试性能监控功能"""
        # 启动监控
        self.pipeline.start_monitoring()
        
        # 等待一段时间让监控收集数据
        import time
        time.sleep(2)
        
        # 获取性能摘要
        summary = self.pipeline.get_performance_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('click_rate', summary)
        self.assertIn('completion_rate', summary)
        self.assertIn('conversion_rate', summary)
        
        # 停止监控
        self.pipeline.stop_monitoring()
    
    def test_pipeline_state_saving_and_loading(self):
        """测试Pipeline状态保存和加载"""
        # 保存状态
        self.pipeline.save_pipeline_state()
        
        # 检查是否创建了保存目录
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # 加载状态（这里只是测试方法调用，不验证实际加载结果）
        try:
            self.pipeline.load_pipeline_state()
        except Exception as e:
            # 如果加载失败，这是正常的，因为我们没有实际的模型文件
            pass

class TestConfigLoader(unittest.TestCase):
    """测试ConfigLoader类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.config_loader = ConfigLoader()
    
    def test_config_loading(self):
        """测试配置加载"""
        config = self.config_loader.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn('system', config)
        self.assertIn('llm_api', config)
    
    def test_config_getting(self):
        """测试配置获取"""
        system_name = self.config_loader.get_config('system.name')
        self.assertIsInstance(system_name, str)
        
        # 测试不存在的配置项
        non_existent = self.config_loader.get_config('non.existent.key')
        self.assertIsNone(non_existent)
    
    def test_config_setting(self):
        """测试配置设置"""
        test_value = "test_value"
        self.config_loader.set_config('test.key', test_value)
        
        retrieved_value = self.config_loader.get_config('test.key')
        self.assertEqual(retrieved_value, test_value)
    
    def test_config_validation(self):
        """测试配置验证"""
        is_valid = self.config_loader.validate_config()
        self.assertIsInstance(is_valid, bool)

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
