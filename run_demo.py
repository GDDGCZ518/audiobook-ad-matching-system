#!/usr/bin/env python3
"""
有声书广告匹配系统演示脚本
运行此脚本来体验完整的广告匹配Pipeline
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioBookAdPipeline

def run_demo():
    """运行演示"""
    print("🎵 有声书广告匹配系统演示")
    print("=" * 50)
    
    try:
        # 创建Pipeline实例
        print("📋 初始化系统...")
        pipeline = AudioBookAdPipeline()
        
        # 启动监控
        print("📊 启动性能监控...")
        pipeline.start_monitoring()
        
        # 运行单个Pipeline示例
        print("\n🚀 运行单个Pipeline示例...")
        result = pipeline.run_full_pipeline("album_001", "ad_001")
        
        # 显示结果
        print(f"\n✅ Pipeline执行成功!")
        print(f"📚 专辑: {result['album_info']['title']}")
        print(f"📢 广告: {result['ad_info']['product_name']}")
        print(f"⭐ 整体评分: {result['overall_score']:.3f}")
        
        # 显示推荐结果
        if result['recommendations']:
            print(f"\n🎯 推荐结果:")
            for i, rec in enumerate(result['recommendations'][:3]):
                print(f"  {i+1}. 匹配度: {rec.get('score', 0):.3f}")
        
        # 显示投放策略
        strategy = result['deployment_strategy']
        print(f"\n📈 投放策略:")
        print(f"  ⏰ 时机: {strategy.get('timing', 'N/A')}")
        print(f"  🔄 频率: {strategy.get('frequency', 'N/A')}次/天")
        print(f"  🎨 创意版本: {strategy.get('creative_version', 'N/A')}")
        
        # 运行批量处理
        print(f"\n🔄 运行批量处理...")
        batch_results = pipeline.batch_process(
            ["album_001", "album_002"], 
            ["ad_001", "ad_002"]
        )
        print(f"✅ 批量处理完成，共处理 {len(batch_results)} 个任务")
        
        # 显示性能摘要
        print(f"\n📊 性能监控结果:")
        performance = pipeline.get_performance_summary()
        print(f"  🖱️  点击率: {performance['click_rate']['current']:.3f}")
        print(f"  ✅ 完成率: {performance['completion_rate']['current']:.3f}")
        print(f"  💰 收入: ${performance['revenue']['current']:.2f}")
        print(f"  ⚡ 响应时间: {performance['response_time']['current']:.3f}s")
        
        # 保存状态
        print(f"\n💾 保存系统状态...")
        pipeline.save_pipeline_state()
        
        # 停止监控
        pipeline.stop_monitoring()
        
        print(f"\n🎉 演示完成! 系统运行正常。")
        print(f"📁 模型已保存到: models/ 目录")
        print(f"📝 详细日志请查看: logs/pipeline.log")
        
    except Exception as e:
        print(f"\n❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主函数"""
    print("欢迎使用有声书广告匹配系统!")
    print("此系统将演示:")
    print("  • 内容理解 (情感分析)")
    print("  • 内容生成 (广告文案)")
    print("  • 匹配推荐 (智能匹配)")
    print("  • 投放优化 (策略优化)")
    print("  • 性能监控 (实时监控)")
    print()
    
    # 运行演示
    success = run_demo()
    
    if success:
        print("\n🎯 系统功能验证完成，所有模块运行正常!")
    else:
        print("\n⚠️  系统运行出现问题，请检查日志文件")
        sys.exit(1)

if __name__ == "__main__":
    main()
