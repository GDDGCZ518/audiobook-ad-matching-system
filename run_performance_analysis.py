#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析脚本
专门用于分析系统性能问题并生成优化建议
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.monitoring_analysis.performance_analyzer import PerformanceAnalyzer
from src.monitoring_analysis.evaluation_metrics import EvaluationMetrics

def run_performance_analysis():
    """运行性能分析"""
    print("🔍 有声书广告匹配系统 - 性能分析")
    print("=" * 60)
    
    try:
        # 创建性能分析器
        print("📊 初始化性能分析器...")
        analyzer = PerformanceAnalyzer()
        
        # 模拟性能数据（实际使用时从系统获取）
        print("📈 加载性能数据...")
        performance_data = {
            "click_through_rate": 0.024,      # 当前点击率
            "conversion_rate": 0.008,         # 当前转化率
            "response_time": 2.163,           # 当前响应时间
            "error_rate": 0.015               # 当前错误率
        }
        
        print(f"当前性能指标:")
        for metric, value in performance_data.items():
            print(f"  {metric}: {value}")
        
        # 分析性能问题
        print(f"\n🔍 开始性能问题分析...")
        issues = analyzer.analyze_performance_issues(performance_data)
        
        if issues:
            print(f"📊 发现 {len(issues)} 个性能问题:")
            print("-" * 40)
            
            for i, issue in enumerate(issues):
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }
                emoji = severity_emoji.get(issue.severity, "⚪")
                
                print(f"{emoji} 问题 {i+1}: {issue.description}")
                print(f"   严重程度: {issue.severity}")
                print(f"   优先级: {issue.priority}")
                print(f"   影响: {issue.impact}")
                print(f"   根因: {issue.root_cause}")
                print(f"   解决方案: {issue.solution}")
                print()
            
            # 生成优化路线图
            print("🗺️ 生成优化路线图...")
            roadmap = analyzer.generate_optimization_roadmap(issues)
            
            print(f"📊 路线图概览:")
            print(f"  总问题数: {roadmap['total_issues']}")
            print(f"  关键问题: {roadmap['critical_issues']}个")
            print(f"  高优先级问题: {roadmap['high_priority_issues']}个")
            
            # 显示分阶段计划
            print(f"\n📅 分阶段优化计划:")
            for phase_name, phase_info in roadmap['phases'].items():
                if phase_info['issues']:
                    print(f"  {phase_name}: {phase_info['timeline']} - {phase_info['focus']}")
                    for issue in phase_info['issues']:
                        print(f"    • {issue.description}")
            
            # 显示预期改进效果
            print(f"\n📈 预期改进效果:")
            improvement = roadmap['estimated_improvement']
            print(f"  点击率提升: +{improvement['ctr_improvement']:.3f}")
            print(f"  转化率提升: +{improvement['conversion_improvement']:.3f}")
            print(f"  响应时间优化: -{improvement['latency_improvement']:.3f}s")
            print(f"  错误率降低: -{improvement['error_rate_improvement']:.3f}")
            print(f"  总体改进: +{improvement['overall_improvement']:.3f}")
            
            # 显示资源需求
            print(f"\n💰 资源需求:")
            resources = roadmap['resource_requirements']
            print(f"  开发工作量: {resources['developer_weeks']:.1f} 周")
            print(f"  优先级: {resources['priority']}")
            
            # 保存分析报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_file = f"logs/performance_analysis_{timestamp}.json"
            analyzer.save_analysis_report(issues, roadmap, analysis_file)
            print(f"\n💾 性能分析报告已保存到: {analysis_file}")
            
            # 生成具体的优化建议
            print(f"\n💡 具体优化建议:")
            generate_specific_recommendations(issues)
            
        else:
            print("🎉 恭喜! 未发现明显的性能问题，系统运行良好!")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 性能分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_specific_recommendations(issues):
    """生成具体的优化建议"""
    print("  📋 详细优化建议:")
    
    for i, issue in enumerate(issues):
        print(f"\n  {i+1}. {issue.description}")
        
        if issue.issue_type.value == "low_ctr":
            print("     具体措施:")
            print("       • 优化推荐算法参数，提高匹配精度")
            print("       • 改进用户画像特征工程")
            print("       • A/B测试不同广告创意")
            print("       • 优化广告展示位置和时机")
            
        elif issue.issue_type.value == "low_conversion":
            print("     具体措施:")
            print("       • 重写广告文案，增强吸引力")
            print("       • 优化落地页设计和用户体验")
            print("       • 简化用户转化流程")
            print("       • 增加用户引导和激励")
            
        elif issue.issue_type.value == "high_latency":
            print("     具体措施:")
            print("       • 优化算法复杂度，减少计算量")
            print("       • 实现智能缓存策略")
            print("       • 增加并行处理和异步处理")
            print("       • 优化数据库查询和索引")
            
        elif issue.issue_type.value == "high_error_rate":
            print("     具体措施:")
            print("       • 完善异常处理机制")
            print("       • 增加系统监控和告警")
            print("       • 实现自动错误恢复")
            print("       • 改进代码质量和测试覆盖")

def run_benchmark_analysis():
    """运行benchmark分析"""
    print("\n📊 Benchmark分析")
    print("-" * 40)
    
    try:
        # 创建评估指标实例
        evaluator = EvaluationMetrics()
        
        # 模拟不同场景的性能数据
        scenarios = {
            "当前系统": {
                "click_through_rate": 0.024,
                "conversion_rate": 0.008,
                "response_time": 2.163,
                "error_rate": 0.015
            },
            "行业平均": {
                "click_through_rate": 0.050,
                "conversion_rate": 0.100,
                "response_time": 0.300,
                "error_rate": 0.010
            },
            "行业领先": {
                "click_through_rate": 0.080,
                "conversion_rate": 0.150,
                "response_time": 0.100,
                "error_rate": 0.005
            }
        }
        
        print("📈 性能对比分析:")
        for scenario_name, metrics in scenarios.items():
            print(f"\n{scenario_name}:")
            for metric_name, value in metrics.items():
                benchmark_result = evaluator.evaluate_against_benchmark(metric_name, value)
                status_emoji = {
                    "excellent": "🟢",
                    "good": "🟡",
                    "acceptable": "🟠",
                    "needs_improvement": "🔴"
                }
                emoji = status_emoji.get(benchmark_result.status, "⚪")
                print(f"  {emoji} {metric_name}: {value:.3f} ({benchmark_result.status})")
        
        # 计算改进空间
        current = scenarios["当前系统"]
        leading = scenarios["行业领先"]
        
        print(f"\n🎯 改进空间分析:")
        for metric_name in current.keys():
            if current[metric_name] > 0:
                if metric_name == "response_time":
                    # 响应时间是越小越好
                    improvement = ((current[metric_name] - leading[metric_name]) / current[metric_name]) * 100
                    print(f"  {metric_name}: 优化空间 {improvement:+.1f}%")
                else:
                    # 其他指标是越大越好
                    improvement = ((leading[metric_name] - current[metric_name]) / current[metric_name]) * 100
                    print(f"  {metric_name}: 提升空间 {improvement:+.1f}%")
        
    except Exception as e:
        print(f"⚠️ Benchmark分析失败: {e}")

if __name__ == "__main__":
    # 运行性能分析
    run_performance_analysis()
    
    # 运行benchmark分析
    run_benchmark_analysis()
