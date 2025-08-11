#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有声书广告匹配系统评估脚本
运行完整的系统评估，生成详细的性能报告和优化建议
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioBookAdPipeline
from src.monitoring_analysis.evaluation_metrics import EvaluationMetrics
from src.monitoring_analysis.performance_analyzer import PerformanceAnalyzer

def run_comprehensive_evaluation():
    """运行全面的系统评估"""
    print("🔍 有声书广告匹配系统 - 全面评估")
    print("=" * 60)
    
    try:
        # 创建Pipeline实例
        print("📋 初始化系统...")
        pipeline = AudioBookAdPipeline()
        
        # 启动监控
        print("📊 启动性能监控...")
        pipeline.start_monitoring()
        
        # 运行评估
        print("\n🚀 开始系统评估...")
        evaluation_report = pipeline.run_evaluation()
        
        # 显示评估结果
        print(f"\n✅ 评估完成!")
        print(f"📊 评估等级: {evaluation_report['summary']['overall_grade']}")
        print(f"⭐ 总体评分: {evaluation_report['summary']['overall_score']:.2f}/4.0")
        
        # 显示指标统计
        print(f"\n📈 指标统计:")
        print(f"  优秀指标: {evaluation_report['summary']['excellent_count']}个")
        print(f"  良好指标: {evaluation_report['summary']['good_count']}个")
        print(f"  可接受指标: {evaluation_report['summary']['acceptable_count']}个")
        print(f"  需改进指标: {evaluation_report['summary']['needs_improvement_count']}个")
        
        # 显示详细指标分析
        print(f"\n🔍 详细指标分析:")
        for metric_name, analysis in evaluation_report['benchmark_analysis'].items():
            status_emoji = {
                "excellent": "🟢",
                "good": "🟡", 
                "acceptable": "🟠",
                "needs_improvement": "🔴"
            }
            emoji = status_emoji.get(analysis['status'], "⚪")
            
            print(f"  {emoji} {metric_name}:")
            print(f"    当前值: {analysis['current_value']:.3f}")
            print(f"    行业标准: {analysis['benchmark_value']:.3f}")
            print(f"    改进空间: {analysis['improvement_percent']:+.1f}%")
            print(f"    状态: {analysis['status']}")
        
        # 显示改进建议
        if evaluation_report['recommendations']:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(evaluation_report['recommendations']):
                print(f"  {i+1}. {rec}")
        else:
            print(f"\n🎉 恭喜! 所有指标都达到了良好或优秀水平!")
        
        # 生成优化方案
        print(f"\n🚀 优化方案:")
        optimization_plan = generate_optimization_plan(evaluation_report)
        for i, plan in enumerate(optimization_plan):
            print(f"  {i+1}. {plan}")
        
        # 保存优化方案
        save_optimization_plan(optimization_plan, evaluation_report)
        
        # 运行深度性能分析
        print(f"\n🔍 深度性能分析:")
        run_deep_performance_analysis(evaluation_report)
        
        # 停止监控
        pipeline.stop_monitoring()
        
        print(f"\n📁 评估报告和优化方案已保存到logs目录")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_optimization_plan(evaluation_report: dict) -> list:
    """根据评估结果生成优化方案"""
    plans = []
    
    # 分析点击率
    ctr_analysis = evaluation_report['benchmark_analysis'].get('click_through_rate', {})
    if ctr_analysis.get('status') in ['needs_improvement', 'acceptable']:
        plans.append("优化推荐算法，提高内容与广告的匹配精度")
        plans.append("改进用户画像建模，增强个性化推荐能力")
        plans.append("优化广告创意展示，提升用户点击意愿")
    
    # 分析转化率
    conversion_analysis = evaluation_report['benchmark_analysis'].get('conversion_rate', {})
    if conversion_analysis.get('status') in ['needs_improvement', 'acceptable']:
        plans.append("优化投放时机，选择用户最活跃的时间段")
        plans.append("改进广告文案，增强产品吸引力")
        plans.append("优化落地页体验，减少转化漏斗流失")
    
    # 分析收入指标
    revenue_analysis = evaluation_report['benchmark_analysis'].get('revenue_per_user', {})
    if revenue_analysis.get('status') in ['needs_improvement', 'acceptable']:
        plans.append("优化定价策略，提高单用户价值")
        plans.append("扩展产品线，增加交叉销售机会")
        plans.append("改进客户服务，提升用户忠诚度")
    
    # 分析响应时间
    latency_analysis = evaluation_report['benchmark_analysis'].get('response_time', {})
    if latency_analysis.get('status') in ['needs_improvement', 'acceptable']:
        plans.append("优化算法性能，减少计算延迟")
        plans.append("改进缓存策略，提升响应速度")
        plans.append("优化数据库查询，减少I/O等待")
    
    # 分析用户满意度
    satisfaction_analysis = evaluation_report['benchmark_analysis'].get('user_satisfaction', {})
    if satisfaction_analysis.get('status') in ['needs_improvement', 'acceptable']:
        plans.append("收集用户反馈，持续改进产品体验")
        plans.append("优化用户界面，提升操作便利性")
        plans.append("加强内容质量，满足用户需求")
    
    # 如果没有具体问题，提供通用优化建议
    if not plans:
        plans = [
            "持续监控系统性能，及时发现潜在问题",
            "定期更新推荐模型，适应市场变化",
            "优化投放策略，提升广告效果",
            "加强数据分析，挖掘用户行为洞察"
        ]
    
    return plans

def run_deep_performance_analysis(evaluation_report: dict):
    """运行深度性能分析"""
    try:
        # 创建性能分析器
        analyzer = PerformanceAnalyzer()
        
        # 提取指标数据
        metrics_data = {}
        for metric_name, analysis in evaluation_report['benchmark_analysis'].items():
            metrics_data[metric_name] = analysis['current_value']
        
        # 分析性能问题
        issues = analyzer.analyze_performance_issues(metrics_data)
        
        if issues:
            print(f"  📊 发现 {len(issues)} 个性能问题:")
            for i, issue in enumerate(issues):
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }
                emoji = severity_emoji.get(issue.severity, "⚪")
                print(f"    {emoji} {issue.description}")
                print(f"       影响: {issue.impact}")
                print(f"       根因: {issue.root_cause}")
                print(f"       解决方案: {issue.solution}")
                print()
            
            # 生成优化路线图
            roadmap = analyzer.generate_optimization_roadmap(issues)
            
            print(f"  🗺️ 优化路线图:")
            print(f"    总问题数: {roadmap['total_issues']}")
            print(f"    关键问题: {roadmap['critical_issues']}个")
            print(f"    高优先级: {roadmap['high_priority_issues']}个")
            
            print(f"\n  📅 分阶段优化计划:")
            for phase_name, phase_info in roadmap['phases'].items():
                if phase_info['issues']:
                    print(f"    {phase_name}: {phase_info['timeline']} - {phase_info['focus']}")
                    for issue in phase_info['issues']:
                        print(f"      • {issue.description}")
            
            print(f"\n  📈 预期改进效果:")
            improvement = roadmap['estimated_improvement']
            print(f"    点击率提升: +{improvement['ctr_improvement']:.3f}")
            print(f"    转化率提升: +{improvement['conversion_improvement']:.3f}")
            print(f"    响应时间优化: -{improvement['latency_improvement']:.3f}s")
            print(f"    错误率降低: -{improvement['error_rate_improvement']:.3f}")
            print(f"    总体改进: +{improvement['overall_improvement']:.3f}")
            
            print(f"\n  💰 资源需求:")
            resources = roadmap['resource_requirements']
            print(f"    开发工作量: {resources['developer_weeks']:.1f} 周")
            print(f"    优先级: {resources['priority']}")
            
            # 保存分析报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_file = f"logs/performance_analysis_{timestamp}.json"
            analyzer.save_analysis_report(issues, roadmap, analysis_file)
            print(f"\n  💾 深度分析报告已保存到: {analysis_file}")
            
        else:
            print("  🎉 恭喜! 未发现明显的性能问题，系统运行良好!")
            
    except Exception as e:
        print(f"  ⚠️ 深度性能分析失败: {e}")

def save_optimization_plan(plans: list, evaluation_report: dict):
    """保存优化方案到文件"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存优化方案
        optimization_file = f"logs/optimization_plan_{timestamp}.json"
        optimization_data = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_summary": evaluation_report['summary'],
            "optimization_plans": plans,
            "priority": "high" if evaluation_report['summary']['overall_score'] < 2.5 else "medium"
        }
        
        with open(optimization_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 优化方案已保存到: {optimization_file}")
        
    except Exception as e:
        print(f"⚠️ 保存优化方案失败: {e}")

def run_benchmark_comparison():
    """运行benchmark对比分析"""
    print("\n📊 Benchmark对比分析")
    print("-" * 40)
    
    try:
        # 创建评估指标实例
        evaluator = EvaluationMetrics()
        
        # 模拟不同场景的性能数据
        scenarios = {
            "当前系统": {
                "click_through_rate": 0.045,
                "conversion_rate": 0.08,
                "revenue_per_user": 1.50,
                "response_time": 0.25,
                "user_satisfaction": 4.2
            },
            "行业平均": {
                "click_through_rate": 0.05,
                "conversion_rate": 0.10,
                "revenue_per_user": 1.80,
                "response_time": 0.30,
                "user_satisfaction": 4.0
            },
            "行业领先": {
                "click_through_rate": 0.08,
                "conversion_rate": 0.15,
                "revenue_per_user": 2.50,
                "response_time": 0.10,
                "user_satisfaction": 4.5
            }
        }
        
        print("📈 性能对比:")
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
                improvement = ((leading[metric_name] - current[metric_name]) / current[metric_name]) * 100
                print(f"  {metric_name}: 提升空间 {improvement:+.1f}%")
        
    except Exception as e:
        print(f"⚠️ Benchmark对比分析失败: {e}")

if __name__ == "__main__":
    # 运行全面评估
    run_comprehensive_evaluation()
    
    # 运行benchmark对比
    run_benchmark_comparison()
