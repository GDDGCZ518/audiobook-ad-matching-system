import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime, timedelta
import time
import threading
from collections import deque

class PerformanceMonitor:
    """性能监测器 - 实时监测系统性能指标"""
    
    def __init__(self):
        self.metrics_history = {
            'click_rate': deque(maxlen=1000),
            'completion_rate': deque(maxlen=1000),
            'conversion_rate': deque(maxlen=1000),
            'revenue': deque(maxlen=1000),
            'response_time': deque(maxlen=1000),
            'error_rate': deque(maxlen=1000)
        }
        
        self.alert_thresholds = {
            'click_rate': {'warning': 0.02, 'critical': 0.01},
            'completion_rate': {'warning': 0.3, 'critical': 0.2},
            'conversion_rate': {'warning': 0.01, 'critical': 0.005},
            'response_time': {'warning': 2.0, 'critical': 5.0},
            'error_rate': {'warning': 0.05, 'critical': 0.1}
        }
        
        self.alerts = deque(maxlen=100)
        self.is_monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监测"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监测"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitoring_loop(self):
        """监测循环"""
        while self.is_monitoring:
            # 模拟获取实时指标
            current_metrics = self._get_current_metrics()
            
            # 检查预警条件
            self._check_alerts(current_metrics)
            
            # 更新历史数据
            self._update_metrics_history(current_metrics)
            
            time.sleep(5)  # 每5秒更新一次
    
    def _get_current_metrics(self) -> Dict:
        """获取当前性能指标（模拟数据）"""
        # 这里应该从实际的系统API获取数据
        return {
            'click_rate': np.random.normal(0.025, 0.005),
            'completion_rate': np.random.normal(0.35, 0.05),
            'conversion_rate': np.random.normal(0.012, 0.003),
            'revenue': np.random.normal(1000, 200),
            'response_time': np.random.normal(1.5, 0.5),
            'error_rate': np.random.normal(0.03, 0.01)
        }
    
    def _check_alerts(self, metrics: Dict):
        """检查预警条件"""
        timestamp = datetime.now()
        
        for metric_name, current_value in metrics.items():
            if metric_name in self.alert_thresholds:
                thresholds = self.alert_thresholds[metric_name]
                
                # 检查严重级别
                if current_value <= thresholds['critical']:
                    alert_level = 'CRITICAL'
                elif current_value <= thresholds['warning']:
                    alert_level = 'WARNING'
                else:
                    continue
                
                # 创建预警
                alert = {
                    'timestamp': timestamp.isoformat(),
                    'level': alert_level,
                    'metric': metric_name,
                    'value': current_value,
                    'threshold': thresholds[alert_level.lower()],
                    'message': f"{metric_name} 指标异常: {current_value:.4f} (阈值: {thresholds[alert_level.lower()]:.4f})"
                }
                
                self.alerts.append(alert)
    
    def _update_metrics_history(self, metrics: Dict):
        """更新指标历史"""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append({
                    'timestamp': timestamp,
                    'value': value
                })
    
    def get_metrics_summary(self) -> Dict:
        """获取指标摘要"""
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                values = [item['value'] for item in history]
                summary[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values) if values else 0,
                    'min': np.min(values) if values else 0,
                    'max': np.max(values) if values else 0,
                    'trend': self._calculate_trend(values)
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        recent_avg = np.mean(values[-10:]) if len(values) >= 10 else values[-1]
        previous_avg = np.mean(values[-20:-10]) if len(values) >= 20 else values[0]
        
        if recent_avg > previous_avg * 1.05:
            return 'increasing'
        elif recent_avg < previous_avg * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """获取最近的预警"""
        return list(self.alerts)[-limit:]

class BenchmarkEvaluator:
    """基准评估器 - 评估系统性能基准"""
    
    def __init__(self):
        self.benchmarks = {
            'click_rate': {'excellent': 0.05, 'good': 0.03, 'fair': 0.02, 'poor': 0.01},
            'completion_rate': {'excellent': 0.6, 'good': 0.4, 'fair': 0.3, 'poor': 0.2},
            'conversion_rate': {'excellent': 0.02, 'good': 0.015, 'fair': 0.01, 'poor': 0.005},
            'response_time': {'excellent': 1.0, 'good': 1.5, 'fair': 2.0, 'poor': 3.0},
            'error_rate': {'excellent': 0.01, 'good': 0.02, 'fair': 0.05, 'poor': 0.1}
        }
    
    def evaluate_performance(self, metrics: Dict) -> Dict:
        """评估性能水平"""
        evaluation = {}
        
        for metric_name, value in metrics.items():
            if metric_name in self.benchmarks:
                benchmark = self.benchmarks[metric_name]
                
                if value >= benchmark['excellent']:
                    level = 'excellent'
                    score = 95
                elif value >= benchmark['good']:
                    level = 'good'
                    score = 80
                elif value >= benchmark['fair']:
                    level = 'fair'
                    score = 60
                else:
                    level = 'poor'
                    score = 40
                
                evaluation[metric_name] = {
                    'value': value,
                    'level': level,
                    'score': score,
                    'benchmark': benchmark
                }
        
        return evaluation
    
    def calculate_overall_score(self, evaluation: Dict) -> float:
        """计算总体评分"""
        if not evaluation:
            return 0.0
        
        scores = [metric['score'] for metric in evaluation.values()]
        weights = [0.3, 0.25, 0.25, 0.1, 0.1]  # 各指标权重
        
        # 确保权重和分数数量匹配
        if len(weights) > len(scores):
            weights = weights[:len(scores)]
        elif len(weights) < len(scores):
            weights.extend([1.0] * (len(scores) - len(weights)))
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return sum(score * weight for score, weight in zip(scores, weights))

class MonitoringDashboard:
    """监测分析仪表板"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.evaluator = BenchmarkEvaluator()
        
        # 初始化Streamlit页面配置
        st.set_page_config(
            page_title="有声书广告匹配系统监测仪表板",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run_dashboard(self):
        """运行仪表板"""
        # 侧边栏
        self._render_sidebar()
        
        # 主页面
        st.title("📊 有声书广告匹配系统监测仪表板")
        st.markdown("---")
        
        # 开始监测
        if st.button("开始实时监测", key="start_monitoring"):
            self.monitor.start_monitoring()
            st.success("监测已启动！")
        
        if st.button("停止监测", key="stop_monitoring"):
            self.monitor.stop_monitoring()
            st.warning("监测已停止！")
        
        # 性能概览
        self._render_performance_overview()
        
        # 实时指标图表
        self._render_real_time_charts()
        
        # 预警信息
        self._render_alerts()
        
        # 基准评估
        self._render_benchmark_evaluation()
        
        # 系统状态
        self._render_system_status()
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("🔧 系统控制")
        
        # 时间范围选择
        st.sidebar.subheader("时间范围")
        time_range = st.sidebar.selectbox(
            "选择时间范围",
            ["最近1小时", "最近6小时", "最近24小时", "最近7天", "自定义"]
        )
        
        # 指标选择
        st.sidebar.subheader("显示指标")
        metrics_to_show = st.sidebar.multiselect(
            "选择要显示的指标",
            ["点击率", "完播率", "转化率", "收入", "响应时间", "错误率"],
            default=["点击率", "完播率", "转化率", "收入"]
        )
        
        # 预警设置
        st.sidebar.subheader("预警设置")
        st.sidebar.slider("预警阈值调整", 0.5, 2.0, 1.0, 0.1)
        
        # 系统信息
        st.sidebar.subheader("系统信息")
        st.sidebar.text(f"版本: 1.0.0")
        st.sidebar.text(f"状态: {'运行中' if self.monitor.is_monitoring else '已停止'}")
        st.sidebar.text(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
    
    def _render_performance_overview(self):
        """渲染性能概览"""
        st.header("📈 性能概览")
        
        # 获取指标摘要
        metrics_summary = self.monitor.get_metrics_summary()
        
        # 创建指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'click_rate' in metrics_summary:
                metric = metrics_summary['click_rate']
                st.metric(
                    label="点击率",
                    value=f"{metric['current']:.3%}",
                    delta=f"{metric['trend']}"
                )
        
        with col2:
            if 'completion_rate' in metrics_summary:
                metric = metrics_summary['completion_rate']
                st.metric(
                    label="完播率",
                    value=f"{metric['current']:.1%}",
                    delta=f"{metric['trend']}"
                )
        
        with col3:
            if 'conversion_rate' in metrics_summary:
                metric = metrics_summary['conversion_rate']
                st.metric(
                    label="转化率",
                    value=f"{metric['current']:.3%}",
                    delta=f"{metric['trend']}"
                )
        
        with col4:
            if 'revenue' in metrics_summary:
                metric = metrics_summary['revenue']
                st.metric(
                    label="收入",
                    value=f"¥{metric['current']:.0f}",
                    delta=f"{metric['trend']}"
                )
    
    def _render_real_time_charts(self):
        """渲染实时图表"""
        st.header("📊 实时指标趋势")
        
        # 获取指标历史数据
        metrics_history = self.monitor.metrics_history
        
        # 创建时间序列图表
        col1, col2 = st.columns(2)
        
        with col1:
            # 点击率和转化率
            if metrics_history['click_rate'] and metrics_history['conversion_rate']:
                fig = go.Figure()
                
                # 点击率
                click_data = list(metrics_history['click_rate'])
                fig.add_trace(go.Scatter(
                    x=[item['timestamp'] for item in click_data],
                    y=[item['value'] for item in click_data],
                    mode='lines+markers',
                    name='点击率',
                    line=dict(color='blue')
                ))
                
                # 转化率
                conversion_data = list(metrics_history['conversion_rate'])
                fig.add_trace(go.Scatter(
                    x=[item['timestamp'] for item in conversion_data],
                    y=[item['value'] for item in conversion_data],
                    mode='lines+markers',
                    name='转化率',
                    line=dict(color='green')
                ))
                
                fig.update_layout(
                    title="点击率与转化率趋势",
                    xaxis_title="时间",
                    yaxis_title="比率",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 完播率和收入
            if metrics_history['completion_rate'] and metrics_history['revenue']:
                fig = go.Figure()
                
                # 完播率
                completion_data = list(metrics_history['completion_rate'])
                fig.add_trace(go.Scatter(
                    x=[item['timestamp'] for item in completion_data],
                    y=[item['value'] for item in completion_data],
                    mode='lines+markers',
                    name='完播率',
                    line=dict(color='orange')
                ))
                
                # 收入（使用双Y轴）
                revenue_data = list(metrics_history['revenue'])
                fig.add_trace(go.Scatter(
                    x=[item['timestamp'] for item in revenue_data],
                    y=[item['value'] for item in revenue_data],
                    mode='lines+markers',
                    name='收入',
                    line=dict(color='red'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="完播率与收入趋势",
                    xaxis_title="时间",
                    yaxis_title="完播率",
                    yaxis2=dict(title="收入", overlaying="y", side="right"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts(self):
        """渲染预警信息"""
        st.header("🚨 预警信息")
        
        # 获取最近预警
        recent_alerts = self.monitor.get_recent_alerts(limit=20)
        
        if recent_alerts:
            # 按严重级别分组
            critical_alerts = [alert for alert in recent_alerts if alert['level'] == 'CRITICAL']
            warning_alerts = [alert for alert in recent_alerts if alert['level'] == 'WARNING']
            
            # 显示严重预警
            if critical_alerts:
                st.error(f"🚨 严重预警 ({len(critical_alerts)}条)")
                for alert in critical_alerts[-5:]:
                    st.error(f"**{alert['timestamp']}** - {alert['message']}")
            
            # 显示一般预警
            if warning_alerts:
                st.warning(f"⚠️ 一般预警 ({len(warning_alerts)}条)")
                for alert in warning_alerts[-5:]:
                    st.warning(f"**{alert['timestamp']}** - {alert['message']}")
            
            # 预警统计
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总预警数", len(recent_alerts))
            
            with col2:
                st.metric("严重预警", len(critical_alerts))
            
            with col3:
                st.metric("一般预警", len(warning_alerts))
        else:
            st.success("✅ 当前无预警信息")
    
    def _render_benchmark_evaluation(self):
        """渲染基准评估"""
        st.header("🎯 性能基准评估")
        
        # 获取当前指标
        metrics_summary = self.monitor.get_metrics_summary()
        current_metrics = {name: summary['current'] for name, summary in metrics_summary.items()}
        
        # 评估性能
        evaluation = self.evaluator.evaluate_performance(current_metrics)
        overall_score = self.evaluator.calculate_overall_score(evaluation)
        
        # 总体评分
        st.subheader(f"总体评分: {overall_score:.1f}/100")
        
        # 进度条显示
        st.progress(overall_score / 100)
        
        # 各指标评估
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("指标评估详情")
            for metric_name, metric_eval in evaluation.items():
                # 计算进度条颜色
                if metric_eval['level'] == 'excellent':
                    color = 'green'
                elif metric_eval['level'] == 'good':
                    color = 'blue'
                elif metric_eval['level'] == 'fair':
                    color = 'orange'
                else:
                    color = 'red'
                
                st.write(f"**{metric_name}**: {metric_eval['score']}/100 ({metric_eval['level']})")
                st.progress(metric_eval['score'] / 100)
        
        with col2:
            st.subheader("基准对比")
            benchmark_df = pd.DataFrame([
                {
                    '指标': metric_name,
                    '当前值': f"{metric_eval['value']:.4f}",
                    '优秀': f"{metric_eval['benchmark']['excellent']:.4f}",
                    '良好': f"{metric_eval['benchmark']['good']:.4f}",
                    '一般': f"{metric_eval['benchmark']['fair']:.4f}",
                    '较差': f"{metric_eval['benchmark']['poor']:.4f}"
                }
                for metric_name, metric_eval in evaluation.items()
            ])
            
            st.dataframe(benchmark_df, use_container_width=True)
    
    def _render_system_status(self):
        """渲染系统状态"""
        st.header("🔧 系统状态")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("系统组件状态")
            
            # 模拟系统组件状态
            components = {
                "内容理解模块": "✅ 正常",
                "匹配推荐模块": "✅ 正常",
                "内容生成模块": "✅ 正常",
                "投放优化模块": "✅ 正常",
                "监测分析模块": "✅ 正常",
                "数据库连接": "✅ 正常",
                "API服务": "✅ 正常"
            }
            
            for component, status in components.items():
                st.text(f"{component}: {status}")
        
        with col2:
            st.subheader("系统资源使用")
            
            # 模拟资源使用情况
            resources = {
                "CPU使用率": "45%",
                "内存使用率": "62%",
                "磁盘使用率": "38%",
                "网络带宽": "23%",
                "GPU使用率": "12%"
            }
            
            for resource, usage in resources.items():
                # 解析百分比
                usage_value = float(usage.strip('%'))
                
                # 根据使用率设置颜色
                if usage_value < 50:
                    color = "green"
                elif usage_value < 80:
                    color = "orange"
                else:
                    color = "red"
                
                st.write(f"**{resource}**: {usage}")
                st.progress(usage_value / 100)
        
        # 系统日志
        st.subheader("系统日志")
        log_entries = [
            f"[{datetime.now().strftime('%H:%M:%S')}] 系统启动完成",
            f"[{datetime.now().strftime('%H:%M:%S')}] 监测模块初始化成功",
            f"[{datetime.now().strftime('%H:%M:%S')}] 数据库连接池建立",
            f"[{datetime.now().strftime('%H:%M:%S')}] API服务启动成功"
        ]
        
        for log in log_entries:
            st.text(log)

def main():
    """主函数"""
    dashboard = MonitoringDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
