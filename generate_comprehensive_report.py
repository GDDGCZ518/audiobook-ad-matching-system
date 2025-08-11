#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合报告生成脚本
整合所有评估和分析结果，生成完整的系统分析报告
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.monitoring_analysis.report_generator import ReportGenerator
from src.monitoring_analysis.evaluation_metrics import EvaluationMetrics
from src.monitoring_analysis.performance_analyzer import PerformanceAnalyzer

def load_latest_reports():
    """加载最新的评估和分析报告"""
    logs_dir = "logs"
    reports = {}
    
    try:
        # 查找最新的评估报告
        evaluation_files = [f for f in os.listdir(logs_dir) if f.startswith("evaluation_report_")]
        if evaluation_files:
            latest_evaluation = max(evaluation_files)
            with open(os.path.join(logs_dir, latest_evaluation), 'r', encoding='utf-8') as f:
                reports['evaluation'] = json.load(f)
            print(f"✅ 加载评估报告: {latest_evaluation}")
        
        # 查找最新的性能分析报告
        performance_files = [f for f in os.listdir(logs_dir) if f.startswith("performance_analysis_")]
        if performance_files:
            latest_performance = max(performance_files)
            with open(os.path.join(logs_dir, latest_performance), 'r', encoding='utf-8') as f:
                reports['performance'] = json.load(f)
            print(f"✅ 加载性能分析报告: {latest_performance}")
        
        # 查找最新的优化方案
        optimization_files = [f for f in os.listdir(logs_dir) if f.startswith("optimization_plan_")]
        if optimization_files:
            latest_optimization = max(optimization_files)
            with open(os.path.join(logs_dir, latest_optimization), 'r', encoding='utf-8') as f:
                reports['optimization'] = json.load(f)
            print(f"✅ 加载优化方案: {latest_optimization}")
        
    except Exception as e:
        print(f"⚠️ 加载报告失败: {e}")
    
    return reports

def generate_fresh_analysis():
    """生成新的分析数据"""
    print("🔄 生成新的分析数据...")
    
    try:
        # 创建评估指标实例
        evaluator = EvaluationMetrics()
        
        # 模拟当前系统性能数据
        current_metrics = {
            "click_through_rate": 0.024,
            "conversion_rate": 0.008,
            "revenue_per_user": 561.085,
            "response_time": 2.163,
            "user_satisfaction": 4.0
        }
        
        # 生成评估报告
        evaluation_report = evaluator.generate_evaluation_report(current_metrics)
        
        # 创建性能分析器
        analyzer = PerformanceAnalyzer()
        
        # 分析性能问题
        performance_data = {
            "click_through_rate": current_metrics["click_through_rate"],
            "conversion_rate": current_metrics["conversion_rate"],
            "response_time": current_metrics["response_time"],
            "error_rate": 0.015
        }
        
        issues = analyzer.analyze_performance_issues(performance_data)
        roadmap = analyzer.generate_optimization_roadmap(issues)
        
        performance_analysis = {
            "performance_issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity,
                    "description": issue.description,
                    "impact": issue.impact,
                    "root_cause": issue.root_cause,
                    "solution": issue.solution,
                    "priority": issue.priority
                }
                for issue in issues
            ],
            "optimization_roadmap": roadmap
        }
        
        return {
            "evaluation": evaluation_report,
            "performance": performance_analysis
        }
        
    except Exception as e:
        print(f"❌ 生成分析数据失败: {e}")
        return {}

def main():
    """主函数"""
    print("📊 有声书广告匹配系统 - 综合报告生成器")
    print("=" * 60)
    
    try:
        # 尝试加载现有报告
        reports = load_latest_reports()
        
        # 如果没有现有报告，生成新的分析数据
        if not reports:
            print("📝 未找到现有报告，生成新的分析数据...")
            reports = generate_fresh_analysis()
        
        if not reports:
            print("❌ 无法获取分析数据，退出")
            return
        
        # 创建报告生成器
        print("📋 创建综合报告...")
        generator = ReportGenerator()
        
        # 生成综合报告
        comprehensive_report = generator.generate_comprehensive_report(
            reports.get('evaluation', {}),
            reports.get('performance', {})
        )
        
        # 保存JSON格式报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = f"logs/comprehensive_report_{timestamp}.json"
        generator.save_report(comprehensive_report, json_file)
        
        # 生成Markdown格式报告
        md_content = generator.generate_markdown_report(comprehensive_report)
        md_file = f"logs/comprehensive_report_{timestamp}.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✅ 综合报告生成完成!")
        print(f"📄 JSON格式: {json_file}")
        print(f"📝 Markdown格式: {md_file}")
        
        # 显示报告摘要
        print(f"\n📊 报告摘要:")
        print(f"  总体评级: {comprehensive_report['executive_summary']['overall_grade']}")
        print(f"  总体评分: {comprehensive_report['executive_summary']['overall_score']:.2f}/4.0")
        
        # 显示关键发现
        print(f"\n🔍 关键发现:")
        for finding in comprehensive_report['executive_summary']['key_findings']:
            print(f"  • {finding}")
        
        # 显示下一步计划
        next_steps = comprehensive_report['next_steps']
        print(f"\n📋 下一步计划:")
        print(f"  当前状态: {next_steps['current_status']}")
        print(f"  时间线: {next_steps['timeline']}")
        print(f"  成功标准: {next_steps['success_criteria']}")
        
        print("\n" + "=" * 60)
        print("🎯 报告生成完成！请查看logs目录中的详细报告文件。")
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
