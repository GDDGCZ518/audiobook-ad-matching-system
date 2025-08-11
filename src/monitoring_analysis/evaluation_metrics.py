#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标和Benchmark系统
为有声书广告匹配系统提供全面的性能评估和基准测试
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    """指标类型"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CTR = "click_through_rate"
    CONVERSION_RATE = "conversion_rate"
    REVENUE = "revenue"
    LATENCY = "latency"
    USER_SATISFACTION = "user_satisfaction"

@dataclass
class BenchmarkResult:
    """Benchmark结果"""
    metric_name: str
    current_value: float
    benchmark_value: float
    improvement: float
    status: str  # "excellent", "good", "acceptable", "needs_improvement"
    timestamp: datetime

class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.benchmarks = self._load_benchmarks()
        
    def _load_benchmarks(self) -> Dict:
        """加载行业benchmark数据"""
        return {
            "click_through_rate": {
                "excellent": 0.08,  # 8%
                "good": 0.05,       # 5%
                "acceptable": 0.03,  # 3%
                "poor": 0.01        # 1%
            },
            "conversion_rate": {
                "excellent": 0.15,  # 15%
                "good": 0.10,       # 10%
                "acceptable": 0.05,  # 5%
                "poor": 0.02        # 2%
            },
            "revenue_per_user": {
                "excellent": 2.50,   # $2.50
                "good": 1.80,        # $1.80
                "acceptable": 1.20,  # $1.20
                "poor": 0.50         # $0.50
            },
            "response_time": {
                "excellent": 0.1,    # 100ms
                "good": 0.3,         # 300ms
                "acceptable": 0.5,   # 500ms
                "poor": 1.0          # 1s
            },
            "user_satisfaction": {
                "excellent": 4.5,    # 4.5/5.0
                "good": 4.0,         # 4.0/5.0
                "acceptable": 3.5,   # 3.5/5.0
                "poor": 3.0          # 3.0/5.0
            }
        }
    
    def calculate_accuracy_metrics(self, predictions: List[bool], actuals: List[bool]) -> Dict:
        """计算准确率相关指标"""
        if len(predictions) != len(actuals):
            raise ValueError("预测值和实际值长度不匹配")
        
        tp = sum(1 for p, a in zip(predictions, actuals) if p and a)
        tn = sum(1 for p, a in zip(predictions, actuals) if not p and not a)
        fp = sum(1 for p, a in zip(predictions, actuals) if p and not a)
        fn = sum(1 for p, a in zip(predictions, actuals) if not p and a)
        
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    def calculate_revenue_metrics(self, revenue_data: List[Dict]) -> Dict:
        """计算收入相关指标"""
        if not revenue_data:
            return {}
        
        revenues = [r.get('amount', 0) for r in revenue_data]
        costs = [r.get('cost', 0) for r in revenue_data]
        
        total_revenue = sum(revenues)
        total_cost = sum(costs)
        net_profit = total_revenue - total_cost
        roi = (net_profit / total_cost) if total_cost > 0 else 0
        avg_revenue_per_user = total_revenue / len(revenue_data) if revenue_data else 0
        
        return {
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "net_profit": net_profit,
            "roi": roi,
            "avg_revenue_per_user": avg_revenue_per_user,
            "revenue_distribution": {
                "min": min(revenues) if revenues else 0,
                "max": max(revenues) if revenues else 0,
                "mean": np.mean(revenues),
                "median": np.median(revenues),
                "std": np.std(revenues)
            }
        }
    
    def calculate_performance_metrics(self, performance_data: List[Dict]) -> Dict:
        """计算性能指标"""
        if not performance_data:
            return {}
        
        response_times = [p.get('response_time', 0) for p in performance_data]
        throughput = [p.get('throughput', 0) for p in performance_data]
        error_rates = [p.get('error_rate', 0) for p in performance_data]
        
        return {
            "response_time": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": np.mean(response_times),
                "median": np.median(response_times),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            },
            "throughput": {
                "min": min(throughput) if throughput else 0,
                "max": max(throughput) if throughput else 0,
                "mean": np.mean(throughput),
                "median": np.median(throughput)
            },
            "error_rate": {
                "mean": np.mean(error_rates),
                "max": max(error_rates) if error_rates else 0
            }
        }
    
    def evaluate_against_benchmark(self, metric_name: str, current_value: float) -> BenchmarkResult:
        """评估当前指标相对于benchmark的表现"""
        if metric_name not in self.benchmarks:
            return BenchmarkResult(
                metric_name=metric_name,
                current_value=current_value,
                benchmark_value=0,
                improvement=0,
                status="unknown",
                timestamp=datetime.now()
            )
        
        benchmark = self.benchmarks[metric_name]
        
        # 确定benchmark值（使用good作为基准）
        benchmark_value = benchmark["good"]
        
        # 计算改进程度
        if benchmark_value > 0:
            improvement = ((current_value - benchmark_value) / benchmark_value) * 100
        else:
            improvement = 0
        
        # 确定状态
        if current_value >= benchmark["excellent"]:
            status = "excellent"
        elif current_value >= benchmark["good"]:
            status = "good"
        elif current_value >= benchmark["acceptable"]:
            status = "acceptable"
        else:
            status = "needs_improvement"
        
        return BenchmarkResult(
            metric_name=metric_name,
            current_value=current_value,
            benchmark_value=benchmark_value,
            improvement=improvement,
            status=status,
            timestamp=datetime.now()
        )
    
    def generate_evaluation_report(self, metrics_data: Dict) -> Dict:
        """生成完整的评估报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "detailed_metrics": {},
            "benchmark_analysis": {},
            "recommendations": []
        }
        
        # 分析各项指标
        for metric_name, value in metrics_data.items():
            benchmark_result = self.evaluate_against_benchmark(metric_name, value)
            report["benchmark_analysis"][metric_name] = {
                "current_value": benchmark_result.current_value,
                "benchmark_value": benchmark_result.benchmark_value,
                "improvement_percent": benchmark_result.improvement,
                "status": benchmark_result.status
            }
            
            # 生成改进建议
            if benchmark_result.status in ["needs_improvement", "acceptable"]:
                report["recommendations"].append(
                    f"{metric_name}: 当前值 {benchmark_result.current_value:.3f} "
                    f"低于行业标准 {benchmark_result.benchmark_value:.3f}，"
                    f"建议优化相关算法和策略"
                )
        
        # 计算总体评分
        status_scores = {"excellent": 4, "good": 3, "acceptable": 2, "needs_improvement": 1}
        total_score = sum(status_scores[r["status"]] for r in report["benchmark_analysis"].values())
        avg_score = total_score / len(report["benchmark_analysis"]) if report["benchmark_analysis"] else 0
        
        report["summary"] = {
            "total_metrics": len(report["benchmark_analysis"]),
            "excellent_count": sum(1 for r in report["benchmark_analysis"].values() if r["status"] == "excellent"),
            "good_count": sum(1 for r in report["benchmark_analysis"].values() if r["status"] == "good"),
            "acceptable_count": sum(1 for r in report["benchmark_analysis"].values() if r["status"] == "acceptable"),
            "needs_improvement_count": sum(1 for r in report["benchmark_analysis"].values() if r["status"] == "needs_improvement"),
            "overall_score": avg_score,
            "overall_grade": self._get_grade(avg_score)
        }
        
        return report
    
    def _get_grade(self, score: float) -> str:
        """根据评分获取等级"""
        if score >= 3.5:
            return "A"
        elif score >= 2.5:
            return "B"
        elif score >= 1.5:
            return "C"
        else:
            return "D"
    
    def save_evaluation_report(self, report: Dict, filepath: str):
        """保存评估报告到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"评估报告已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存评估报告失败: {e}")
    
    def load_evaluation_report(self, filepath: str) -> Dict:
        """从文件加载评估报告"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载评估报告失败: {e}")
            return {}
