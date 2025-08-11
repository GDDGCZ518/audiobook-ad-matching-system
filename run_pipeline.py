#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有声书广告匹配Pipeline快速运行脚本
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_demo():
    """运行演示程序"""
    print("=" * 60)
    print("有声书广告匹配Pipeline演示程序")
    print("=" * 60)
    
    try:
        # 导入主程序
        from main import AudioBookAdPipeline
        
        print("正在初始化Pipeline...")
        pipeline = AudioBookAdPipeline()
        print("✓ Pipeline初始化完成")
        
        # 运行单个示例
        print("\n" + "=" * 40)
        print("示例1: 单个Pipeline执行")
        print("=" * 40)
        
        result = pipeline.run_full_pipeline("album_001", "ad_001")
        
        print(f"Pipeline ID: {result['pipeline_id']}")
        print(f"执行时间: {result['timestamp']}")
        print(f"整体评分: {result['overall_score']:.3f}")
        
        # 显示情感分析结果
        emotion = result['emotion_analysis']
        print(f"\n情感分析结果:")
        print(f"  情感类别: {', '.join(emotion['emotion_categories'])}")
        print(f"  情感强度: {emotion['emotion_intensity']}/10")
        print(f"  置信度: {emotion['confidence']:.2f}")
        
        # 显示广告生成结果
        ad_gen = result['ad_generation']
        print(f"\n广告生成结果:")
        print(f"  质量评分: {ad_gen['quality_score']:.3f}")
        
        # 显示推荐结果
        recommendations = result['recommendations']
        print(f"\n推荐结果 (前3个):")
        for i, rec in enumerate(recommendations[:3]):
            print(f"  {i+1}. 广告ID: {rec['ad_id']}, 匹配度: {rec['score']:.3f}")
        
        # 显示投放策略
        strategy = result['deployment_strategy']
        print(f"\n投放策略:")
        print(f"  投放时机: {strategy['timing']}")
        print(f"  投放频率: {strategy['frequency']}")
        print(f"  创意版本: {strategy['creative_version']}")
        print(f"  定向强度: {strategy['targeting_strength']:.3f}")
        
        # 批量处理示例
        print("\n" + "=" * 40)
        print("示例2: 批量处理")
        print("=" * 40)
        
        batch_results = pipeline.batch_process(
            ["album_001", "album_002"], 
            ["ad_001", "ad_002"]
        )
        
        print(f"批量处理完成，共处理 {len(batch_results)} 个任务")
        
        # 计算平均评分
        avg_score = sum(r['overall_score'] for r in batch_results) / len(batch_results)
        print(f"平均评分: {avg_score:.3f}")
        
        # 性能监控
        print("\n" + "=" * 40)
        print("示例3: 性能监控")
        print("=" * 40)
        
        pipeline.start_monitoring()
        
        # 等待监控收集数据
        import time
        time.sleep(3)
        
        performance_summary = pipeline.get_performance_summary()
        print("性能指标摘要:")
        for metric, value in performance_summary.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        pipeline.stop_monitoring()
        
        # 保存状态
        print("\n" + "=" * 40)
        print("保存Pipeline状态...")
        print("=" * 40)
        
        pipeline.save_pipeline_state()
        print("✓ Pipeline状态已保存")
        
        print("\n" + "=" * 60)
        print("演示程序执行完成！")
        print("=" * 60)
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"执行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_interactive():
    """运行交互式程序"""
    print("=" * 60)
    print("有声书广告匹配Pipeline交互式程序")
    print("=" * 60)
    
    try:
        from main import AudioBookAdPipeline
        
        print("正在初始化Pipeline...")
        pipeline = AudioBookAdPipeline()
        print("✓ Pipeline初始化完成")
        
        while True:
            print("\n" + "-" * 40)
            print("请选择操作:")
            print("1. 运行单个Pipeline")
            print("2. 批量处理")
            print("3. 查看性能监控")
            print("4. 保存状态")
            print("5. 退出")
            print("-" * 40)
            
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                album_id = input("请输入专辑ID (如: album_001): ").strip()
                ad_id = input("请输入广告ID (如: ad_001): ").strip()
                
                if album_id and ad_id:
                    try:
                        result = pipeline.run_full_pipeline(album_id, ad_id)
                        print(f"\n✓ Pipeline执行成功!")
                        print(f"整体评分: {result['overall_score']:.3f}")
                    except Exception as e:
                        print(f"✗ Pipeline执行失败: {e}")
                else:
                    print("✗ 请输入有效的ID")
            
            elif choice == '2':
                album_ids = input("请输入专辑ID列表 (用逗号分隔): ").strip().split(',')
                ad_ids = input("请输入广告ID列表 (用逗号分隔): ").strip().split(',')
                
                album_ids = [aid.strip() for aid in album_ids if aid.strip()]
                ad_ids = [aid.strip() for aid in ad_ids if aid.strip()]
                
                if album_ids and ad_ids:
                    try:
                        results = pipeline.batch_process(album_ids, ad_ids)
                        print(f"\n✓ 批量处理完成，共处理 {len(results)} 个任务")
                        
                        avg_score = sum(r['overall_score'] for r in results) / len(results)
                        print(f"平均评分: {avg_score:.3f}")
                    except Exception as e:
                        print(f"✗ 批量处理失败: {e}")
                else:
                    print("✗ 请输入有效的ID列表")
            
            elif choice == '3':
                try:
                    pipeline.start_monitoring()
                    print("监控已启动，等待数据收集...")
                    
                    import time
                    time.sleep(3)
                    
                    summary = pipeline.get_performance_summary()
                    print("\n性能指标摘要:")
                    for metric, value in summary.items():
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
                    
                    pipeline.stop_monitoring()
                except Exception as e:
                    print(f"✗ 性能监控失败: {e}")
            
            elif choice == '4':
                try:
                    pipeline.save_pipeline_state()
                    print("✓ Pipeline状态已保存")
                except Exception as e:
                    print(f"✗ 保存状态失败: {e}")
            
            elif choice == '5':
                print("再见！")
                break
            
            else:
                print("✗ 无效选择，请输入1-5")
    
    except Exception as e:
        print(f"程序错误: {e}")
        sys.exit(1)

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive()
    else:
        run_demo()

if __name__ == "__main__":
    main()
