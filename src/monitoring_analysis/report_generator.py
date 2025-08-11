#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合报告生成器
整合评估指标、性能分析和优化建议，生成完整的系统分析报告
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    priority: str  # "high", "medium", "low"

class ReportGenerator:
    """综合报告生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_comprehensive_report(self, evaluation_report: Dict, 
                                   performance_analysis: Dict = None) -> Dict:
        """生成综合报告"""
        report = {
            "report_info": {
                "title": "有声书广告匹配系统综合评估报告",
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "executive_summary": self._generate_executive_summary(evaluation_report),
            "performance_overview": self._generate_performance_overview(evaluation_report),
            "detailed_analysis": self._generate_detailed_analysis(evaluation_report),
            "optimization_roadmap": self._generate_optimization_roadmap(performance_analysis),
            "recommendations": self._generate_recommendations(evaluation_report, performance_analysis),
            "next_steps": self._generate_next_steps(evaluation_report, performance_analysis)
        }
        
        return report
    
    def _generate_executive_summary(self, evaluation_report: Dict) -> Dict:
        """生成执行摘要"""
        summary = evaluation_report.get('summary', {})
        
        return {
            "overall_grade": summary.get('overall_grade', 'N/A'),
            "overall_score": summary.get('overall_score', 0),
            "key_findings": [
                f"系统总体评级: {summary.get('overall_grade', 'N/A')}",
                f"优秀指标: {summary.get('excellent_count', 0)}个",
                f"需改进指标: {summary.get('needs_improvement_count', 0)}个",
                f"主要问题: 点击率和转化率低于行业标准"
            ],
            "business_impact": "当前性能水平可能影响广告收入和用户满意度，建议优先解决关键问题"
        }
    
    def _generate_performance_overview(self, evaluation_report: Dict) -> Dict:
        """生成性能概览"""
        benchmark_analysis = evaluation_report.get('benchmark_analysis', {})
        
        overview = {
            "metrics_summary": {},
            "strengths": [],
            "weaknesses": [],
            "opportunities": []
        }
        
        for metric_name, analysis in benchmark_analysis.items():
            status = analysis.get('status', 'unknown')
            current_value = analysis.get('current_value', 0)
            benchmark_value = analysis.get('benchmark_value', 0)
            
            overview["metrics_summary"][metric_name] = {
                "current": current_value,
                "benchmark": benchmark_value,
                "status": status,
                "gap": current_value - benchmark_value
            }
            
            if status == "excellent":
                overview["strengths"].append(f"{metric_name}: 表现优秀，达到行业领先水平")
            elif status == "good":
                overview["strengths"].append(f"{metric_name}: 表现良好，接近行业标准")
            elif status in ["needs_improvement", "acceptable"]:
                overview["weaknesses"].append(f"{metric_name}: 需要改进，低于行业标准")
                overview["opportunities"].append(f"{metric_name}: 有较大提升空间")
        
        return overview
    
    def _generate_detailed_analysis(self, evaluation_report: Dict) -> Dict:
        """生成详细分析"""
        benchmark_analysis = evaluation_report.get('benchmark_analysis', {})
        
        analysis = {
            "metric_analysis": {},
            "trend_analysis": "基于当前数据，系统在点击率和转化率方面有显著改进空间",
            "risk_assessment": {
                "high_risk": ["点击率低影响广告效果", "转化率低影响收入"],
                "medium_risk": ["响应时间可能影响用户体验"],
                "low_risk": ["用户满意度达到标准"]
            }
        }
        
        for metric_name, analysis_data in benchmark_analysis.items():
            current_value = analysis_data.get('current_value', 0)
            benchmark_value = analysis_data.get('benchmark_value', 0)
            improvement_percent = analysis_data.get('improvement_percent', 0)
            
            analysis["metric_analysis"][metric_name] = {
                "current_performance": current_value,
                "benchmark_comparison": benchmark_value,
                "gap_analysis": f"差距: {improvement_percent:+.1f}%",
                "improvement_potential": self._assess_improvement_potential(improvement_percent)
            }
        
        return analysis
    
    def _generate_optimization_roadmap(self, performance_analysis: Dict) -> Dict:
        """生成优化路线图"""
        if not performance_analysis:
            return {"message": "无性能分析数据"}
        
        roadmap = performance_analysis.get('optimization_roadmap', {})
        
        return {
            "timeline": {
                "phase_1": "1-2周: 解决关键问题",
                "phase_2": "2-4周: 解决高优先级问题", 
                "phase_3": "4-8周: 解决中低优先级问题"
            },
            "resource_requirements": roadmap.get('resource_requirements', {}),
            "expected_improvements": roadmap.get('estimated_improvement', {}),
            "success_metrics": [
                "点击率提升到5%以上",
                "转化率提升到10%以上",
                "响应时间降低到500ms以下"
            ]
        }
    
    def _generate_recommendations(self, evaluation_report: Dict, 
                                performance_analysis: Dict) -> Dict:
        """生成优化建议"""
        recommendations = {
            "immediate_actions": [],
            "short_term_improvements": [],
            "long_term_optimizations": [],
            "priority_order": []
        }
        
        # 基于评估报告生成建议
        if evaluation_report.get('recommendations'):
            recommendations["immediate_actions"].extend(
                evaluation_report['recommendations']
            )
        
        # 基于性能分析生成建议
        if performance_analysis and performance_analysis.get('performance_issues'):
            issues = performance_analysis['performance_issues']
            
            for issue in issues:
                priority = issue.get('priority', 3)
                solution = issue.get('solution', '')
                
                if priority == 1:
                    recommendations["immediate_actions"].append(solution)
                elif priority == 2:
                    recommendations["short_term_improvements"].append(solution)
                else:
                    recommendations["long_term_optimizations"].append(solution)
        
        # 去重并排序
        recommendations["immediate_actions"] = list(set(recommendations["immediate_actions"]))
        recommendations["short_term_improvements"] = list(set(recommendations["short_term_improvements"]))
        recommendations["long_term_optimizations"] = list(set(recommendations["long_term_optimizations"]))
        
        # 设置优先级顺序
        recommendations["priority_order"] = [
            "解决关键性能问题",
            "优化推荐算法",
            "改进用户界面",
            "增强监控系统",
            "长期架构优化"
        ]
        
        return recommendations
    
    def _generate_next_steps(self, evaluation_report: Dict, 
                            performance_analysis: Dict) -> Dict:
        """生成下一步行动计划"""
        summary = evaluation_report.get('summary', {})
        overall_score = summary.get('overall_score', 0)
        
        if overall_score >= 3.5:
            status = "excellent"
            next_steps = [
                "保持当前优秀表现",
                "持续监控关键指标",
                "探索进一步优化机会"
            ]
        elif overall_score >= 2.5:
            status = "good"
            next_steps = [
                "实施短期优化计划",
                "重点关注需改进指标",
                "制定长期优化策略"
            ]
        else:
            status = "needs_improvement"
            next_steps = [
                "立即启动关键问题修复",
                "分配高优先级资源",
                "制定紧急优化计划"
            ]
        
        return {
            "current_status": status,
            "immediate_actions": next_steps,
            "timeline": "建议在2周内完成初步优化",
            "success_criteria": "点击率和转化率提升到行业标准水平"
        }
    
    def _assess_improvement_potential(self, improvement_percent: float) -> str:
        """评估改进潜力"""
        if improvement_percent >= 100:
            return "极高"
        elif improvement_percent >= 50:
            return "高"
        elif improvement_percent >= 20:
            return "中等"
        elif improvement_percent >= 0:
            return "低"
        else:
            return "需要改进"
    
    def save_report(self, report: Dict, filepath: str):
        """保存报告到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"综合报告已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")
    
    def generate_markdown_report(self, report: Dict) -> str:
        """生成Markdown格式的报告"""
        md_content = f"""# {report['report_info']['title']}

**生成时间**: {report['report_info']['generated_at']}  
**报告版本**: {report['report_info']['version']}

## 📊 执行摘要

**总体评级**: {report['executive_summary']['overall_grade']}  
**总体评分**: {report['executive_summary']['overall_score']:.2f}/4.0

### 关键发现
"""
        
        for finding in report['executive_summary']['key_findings']:
            md_content += f"- {finding}\n"
        
        md_content += f"\n**业务影响**: {report['executive_summary']['business_impact']}\n"
        
        # 性能概览
        md_content += "\n## 📈 性能概览\n\n"
        
        overview = report['performance_overview']
        if overview['strengths']:
            md_content += "### ✅ 优势\n"
            for strength in overview['strengths']:
                md_content += f"- {strength}\n"
            md_content += "\n"
        
        if overview['weaknesses']:
            md_content += "### ⚠️ 需改进\n"
            for weakness in overview['weaknesses']:
                md_content += f"- {weakness}\n"
            md_content += "\n"
        
        # 优化路线图
        roadmap = report['optimization_roadmap']
        if roadmap.get('timeline'):
            md_content += "## 🗺️ 优化路线图\n\n"
            for phase, description in roadmap['timeline'].items():
                md_content += f"**{phase}**: {description}\n"
            md_content += "\n"
        
        # 建议
        recommendations = report['recommendations']
        if recommendations['immediate_actions']:
            md_content += "## 🚀 立即行动\n\n"
            for action in recommendations['immediate_actions']:
                md_content += f"- {action}\n"
            md_content += "\n"
        
        # 下一步
        next_steps = report['next_steps']
        md_content += f"## 📋 下一步计划\n\n"
        md_content += f"**当前状态**: {next_steps['current_status']}\n"
        md_content += f"**时间线**: {next_steps['timeline']}\n"
        md_content += f"**成功标准**: {next_steps['success_criteria']}\n\n"
        
        md_content += "### 具体行动\n"
        for step in next_steps['immediate_actions']:
            md_content += f"- {step}\n"
        
        return md_content
