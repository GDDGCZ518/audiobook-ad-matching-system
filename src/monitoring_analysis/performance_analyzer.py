#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析器
为有声书广告匹配系统提供深入的性能分析和优化建议
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class PerformanceIssue(Enum):
    """性能问题类型"""
    HIGH_LATENCY = "high_latency"
    LOW_CTR = "low_ctr"
    LOW_CONVERSION = "low_conversion"
    LOW_REVENUE = "low_revenue"
    HIGH_ERROR_RATE = "high_error_rate"
    POOR_USER_EXPERIENCE = "poor_user_experience"

@dataclass
class PerformanceIssueDetail:
    """性能问题详情"""
    issue_type: PerformanceIssue
    severity: str  # "critical", "high", "medium", "low"
    description: str
    impact: str
    root_cause: str
    solution: str
    priority: int  # 1-5, 1为最高优先级

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.issue_patterns = self._load_issue_patterns()
        
    def _load_issue_patterns(self) -> Dict:
        """加载问题模式库"""
        return {
            "click_through_rate": {
                "critical": {"threshold": 0.02, "description": "点击率极低，严重影响广告效果"},
                "high": {"threshold": 0.03, "description": "点击率偏低，需要重点关注"},
                "medium": {"threshold": 0.04, "description": "点击率略低，有改进空间"},
                "low": {"threshold": 0.05, "description": "点击率接近标准，轻微优化即可"}
            },
            "conversion_rate": {
                "critical": {"threshold": 0.02, "description": "转化率极低，用户行为转化困难"},
                "high": {"threshold": 0.05, "description": "转化率偏低，影响收入增长"},
                "medium": {"threshold": 0.08, "description": "转化率略低，有提升潜力"},
                "low": {"threshold": 0.10, "description": "转化率接近标准，小幅优化即可"}
            },
            "response_time": {
                "critical": {"threshold": 2.0, "description": "响应时间过长，严重影响用户体验"},
                "high": {"threshold": 1.0, "description": "响应时间偏长，需要优化"},
                "medium": {"threshold": 0.5, "description": "响应时间略长，有优化空间"},
                "low": {"threshold": 0.3, "description": "响应时间接近标准，轻微优化即可"}
            },
            "error_rate": {
                "critical": {"threshold": 0.10, "description": "错误率过高，系统稳定性差"},
                "high": {"threshold": 0.05, "description": "错误率偏高，需要关注"},
                "medium": {"threshold": 0.02, "description": "错误率略高，有改进空间"},
                "low": {"threshold": 0.01, "description": "错误率接近标准，轻微优化即可"}
            }
        }
    
    def analyze_performance_issues(self, metrics_data: Dict) -> List[PerformanceIssueDetail]:
        """分析性能问题"""
        issues = []
        
        # 分析点击率问题
        ctr = metrics_data.get('click_through_rate', 0)
        ctr_issue = self._analyze_ctr_issue(ctr)
        if ctr_issue:
            issues.append(ctr_issue)
        
        # 分析转化率问题
        conversion = metrics_data.get('conversion_rate', 0)
        conversion_issue = self._analyze_conversion_issue(conversion)
        if conversion_issue:
            issues.append(conversion_issue)
        
        # 分析响应时间问题
        response_time = metrics_data.get('response_time', 0)
        latency_issue = self._analyze_latency_issue(response_time)
        if latency_issue:
            issues.append(latency_issue)
        
        # 分析错误率问题
        error_rate = metrics_data.get('error_rate', 0)
        error_issue = self._analyze_error_issue(error_rate)
        if error_issue:
            issues.append(error_issue)
        
        # 按优先级排序
        issues.sort(key=lambda x: x.priority)
        
        return issues
    
    def _analyze_ctr_issue(self, ctr: float) -> Optional[PerformanceIssueDetail]:
        """分析点击率问题"""
        if ctr < self.issue_patterns["click_through_rate"]["critical"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.LOW_CTR,
                severity="critical",
                description="点击率极低，严重影响广告效果",
                impact="广告展示效果差，用户参与度低，收入损失严重",
                root_cause="推荐算法匹配精度不足，广告创意质量差，用户画像不准确",
                solution="优化推荐算法，改进广告创意，完善用户画像建模",
                priority=1
            )
        elif ctr < self.issue_patterns["click_through_rate"]["high"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.LOW_CTR,
                severity="high",
                description="点击率偏低，需要重点关注",
                impact="广告效果不佳，用户参与度偏低，收入增长受限",
                root_cause="推荐策略需要优化，广告展示时机不当",
                solution="调整推荐策略，优化投放时机，A/B测试不同创意",
                priority=2
            )
        elif ctr < self.issue_patterns["click_through_rate"]["medium"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.LOW_CTR,
                severity="medium",
                description="点击率略低，有改进空间",
                impact="广告效果一般，有提升潜力",
                root_cause="推荐算法和创意展示有优化空间",
                solution="微调推荐参数，优化创意展示，增加个性化元素",
                priority=3
            )
        
        return None
    
    def _analyze_conversion_issue(self, conversion: float) -> Optional[PerformanceIssueDetail]:
        """分析转化率问题"""
        if conversion < self.issue_patterns["conversion_rate"]["critical"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.LOW_CONVERSION,
                severity="critical",
                description="转化率极低，用户行为转化困难",
                impact="用户行为转化困难，收入损失严重，ROI极低",
                root_cause="广告文案吸引力不足，落地页体验差，产品匹配度低",
                solution="重写广告文案，优化落地页，改进产品匹配算法",
                priority=1
            )
        elif conversion < self.issue_patterns["conversion_rate"]["high"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.LOW_CONVERSION,
                severity="high",
                description="转化率偏低，影响收入增长",
                impact="收入增长受限，用户价值开发不足",
                root_cause="广告文案和落地页需要优化，用户引导流程不畅",
                solution="优化广告文案，改进落地页设计，简化用户转化流程",
                priority=2
            )
        elif conversion < self.issue_patterns["conversion_rate"]["medium"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.LOW_CONVERSION,
                severity="medium",
                description="转化率略低，有提升潜力",
                impact="有提升空间，可增加收入",
                root_cause="转化流程有优化空间，用户体验可进一步改进",
                solution="优化转化流程，增加用户引导，改进用户体验",
                priority=3
            )
        
        return None
    
    def _analyze_latency_issue(self, response_time: float) -> Optional[PerformanceIssueDetail]:
        """分析响应时间问题"""
        if response_time > self.issue_patterns["response_time"]["critical"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.HIGH_LATENCY,
                severity="critical",
                description="响应时间过长，严重影响用户体验",
                impact="用户体验极差，用户流失严重，系统可用性低",
                root_cause="算法性能差，系统架构不合理，资源不足",
                solution="重构算法，优化系统架构，增加计算资源",
                priority=1
            )
        elif response_time > self.issue_patterns["response_time"]["high"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.HIGH_LATENCY,
                severity="high",
                description="响应时间偏长，需要优化",
                impact="用户体验不佳，可能影响用户留存",
                root_cause="算法效率需要提升，缓存策略不当",
                solution="优化算法性能，改进缓存策略，增加并行处理",
                priority=2
            )
        elif response_time > self.issue_patterns["response_time"]["medium"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.HIGH_LATENCY,
                severity="medium",
                description="响应时间略长，有优化空间",
                impact="用户体验一般，有改进空间",
                root_cause="算法和系统有优化空间",
                solution="微调算法参数，优化系统配置，增加缓存",
                priority=3
            )
        
        return None
    
    def _analyze_error_issue(self, error_rate: float) -> Optional[PerformanceIssueDetail]:
        """分析错误率问题"""
        if error_rate > self.issue_patterns["error_rate"]["critical"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.HIGH_ERROR_RATE,
                severity="critical",
                description="错误率过高，系统稳定性差",
                impact="系统不可靠，用户信任度低，业务中断风险高",
                root_cause="代码质量差，异常处理不当，系统设计缺陷",
                solution="重构代码，完善异常处理，重新设计系统架构",
                priority=1
            )
        elif error_rate > self.issue_patterns["error_rate"]["high"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.HIGH_ERROR_RATE,
                severity="high",
                description="错误率偏高，需要关注",
                impact="系统可靠性不足，影响用户体验",
                root_cause="异常处理需要改进，监控不足",
                solution="改进异常处理，增加监控，完善日志",
                priority=2
            )
        elif error_rate > self.issue_patterns["error_rate"]["medium"]["threshold"]:
            return PerformanceIssueDetail(
                issue_type=PerformanceIssue.HIGH_ERROR_RATE,
                severity="medium",
                description="错误率略高，有改进空间",
                impact="系统稳定性一般，有改进空间",
                root_cause="异常处理有优化空间",
                solution="优化异常处理，增加错误恢复机制",
                priority=3
            )
        
        return None
    
    def generate_optimization_roadmap(self, issues: List[PerformanceIssueDetail]) -> Dict:
        """生成优化路线图"""
        roadmap = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i.severity == "critical"]),
            "high_priority_issues": len([i for i in issues if i.priority <= 2]),
            "phases": {
                "phase_1": {
                    "timeline": "1-2周",
                    "focus": "解决关键问题",
                    "issues": [i for i in issues if i.severity == "critical"]
                },
                "phase_2": {
                    "timeline": "2-4周",
                    "focus": "解决高优先级问题",
                    "issues": [i for i in issues if i.severity == "high" and i.severity != "critical"]
                },
                "phase_3": {
                    "timeline": "4-8周",
                    "focus": "解决中低优先级问题",
                    "issues": [i for i in issues if i.severity in ["medium", "low"]]
                }
            },
            "estimated_improvement": self._estimate_improvement(issues),
            "resource_requirements": self._estimate_resources(issues)
        }
        
        return roadmap
    
    def _estimate_improvement(self, issues: List[PerformanceIssueDetail]) -> Dict:
        """估算改进效果"""
        improvement = {
            "ctr_improvement": 0.0,
            "conversion_improvement": 0.0,
            "latency_improvement": 0.0,
            "error_rate_improvement": 0.0,
            "overall_improvement": 0.0
        }
        
        for issue in issues:
            if issue.issue_type == PerformanceIssue.LOW_CTR:
                if issue.severity == "critical":
                    improvement["ctr_improvement"] += 0.03
                elif issue.severity == "high":
                    improvement["ctr_improvement"] += 0.02
                elif issue.severity == "medium":
                    improvement["ctr_improvement"] += 0.01
            
            elif issue.issue_type == PerformanceIssue.LOW_CONVERSION:
                if issue.severity == "critical":
                    improvement["conversion_improvement"] += 0.05
                elif issue.severity == "high":
                    improvement["conversion_improvement"] += 0.03
                elif issue.severity == "medium":
                    improvement["conversion_improvement"] += 0.01
            
            elif issue.issue_type == PerformanceIssue.HIGH_LATENCY:
                if issue.severity == "critical":
                    improvement["latency_improvement"] += 0.8
                elif issue.severity == "high":
                    improvement["latency_improvement"] += 0.5
                elif issue.severity == "medium":
                    improvement["latency_improvement"] += 0.2
            
            elif issue.issue_type == PerformanceIssue.HIGH_ERROR_RATE:
                if issue.severity == "critical":
                    improvement["error_rate_improvement"] += 0.05
                elif issue.severity == "high":
                    improvement["error_rate_improvement"] += 0.03
                elif issue.severity == "medium":
                    improvement["error_rate_improvement"] += 0.01
        
        # 计算总体改进
        improvement["overall_improvement"] = (
            improvement["ctr_improvement"] * 0.3 +
            improvement["conversion_improvement"] * 0.3 +
            improvement["latency_improvement"] * 0.2 +
            improvement["error_rate_improvement"] * 0.2
        )
        
        return improvement
    
    def _estimate_resources(self, issues: List[PerformanceIssueDetail]) -> Dict:
        """估算资源需求"""
        resources = {
            "developer_weeks": 0,
            "priority": "low"
        }
        
        for issue in issues:
            if issue.severity == "critical":
                resources["developer_weeks"] += 2
            elif issue.severity == "high":
                resources["developer_weeks"] += 1
            elif issue.severity == "medium":
                resources["developer_weeks"] += 0.5
            else:
                resources["developer_weeks"] += 0.25
        
        # 确定优先级
        if resources["developer_weeks"] >= 6:
            resources["priority"] = "critical"
        elif resources["developer_weeks"] >= 3:
            resources["priority"] = "high"
        elif resources["developer_weeks"] >= 1:
            resources["priority"] = "medium"
        else:
            resources["priority"] = "low"
        
        return resources
    
    def save_analysis_report(self, issues: List[PerformanceIssueDetail], 
                           roadmap: Dict, filepath: str):
        """保存分析报告"""
        try:
            report = {
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
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"性能分析报告已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存性能分析报告失败: {e}")
