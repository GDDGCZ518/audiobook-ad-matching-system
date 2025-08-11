import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random
from collections import deque
import json
import os
from datetime import datetime

class Actor(nn.Module):
    """Actor网络 - 策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super(Actor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 输出层使用tanh激活函数，将动作限制在[-1, 1]范围内
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    """Critic网络 - 价值网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super(Critic, self).__init__()
        
        # 状态和动作的编码层
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 合并后的处理层
        merged_dim = hidden_dims[0] + hidden_dims[0] // 2
        layers = []
        prev_dim = merged_dim
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.value_network = nn.Sequential(*layers)
        
    def forward(self, state, action):
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        
        # 合并状态和动作特征
        combined = torch.cat([state_features, action_features], dim=1)
        return self.value_network(combined)

class DDPGAgent:
    """DDPG智能体 - 用于优化广告投放策略"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.001,
                 buffer_size: int = 100000, batch_size: int = 64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # 噪声参数
        self.noise_std = 0.1
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'episode_lengths': []
        }
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update_networks(self):
        """更新网络参数"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样经验批次
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 更新Critic网络
        next_actions = self.target_actor(next_states)
        target_q = self.target_critic(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor网络
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        # 记录损失
        self.training_stats['actor_losses'].append(actor_loss.item())
        self.training_stats['critic_losses'].append(critic_loss.item())
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新目标网络"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, os.path.join(save_path, 'ddpg_model.pth'))
    
    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(os.path.join(load_path, 'ddpg_model.pth'))
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

class DeploymentOptimizer:
    """投放优化器 - 使用强化学习优化广告投放策略"""
    
    def __init__(self, state_dim: int = 50, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 初始化DDPG智能体
        self.agent = DDPGAgent(state_dim, action_dim)
        
        # 投放环境状态
        self.current_state = None
        self.episode_step = 0
        self.episode_reward = 0
        
        # 性能指标
        self.performance_metrics = {
            'click_rate': 0.0,
            'completion_rate': 0.0,
            'conversion_rate': 0.0,
            'revenue': 0.0
        }
        
        # 投放策略历史
        self.deployment_history = []
        
    def get_state_representation(self, album_info: Dict, ad_info: Dict, 
                                user_context: Dict, performance_history: List[Dict]) -> np.ndarray:
        """构建状态表示"""
        state = np.zeros(self.state_dim)
        
        # 专辑特征 (20维)
        album_features = self._extract_album_features(album_info)
        state[:20] = album_features
        
        # 广告特征 (15维)
        ad_features = self._extract_ad_features(ad_info)
        state[20:35] = ad_features
        
        # 用户上下文 (10维)
        user_features = self._extract_user_features(user_context)
        state[35:45] = user_features
        
        # 历史性能 (5维)
        history_features = self._extract_history_features(performance_history)
        state[45:50] = history_features
        
        return state
    
    def _extract_album_features(self, album_info: Dict) -> np.ndarray:
        """提取专辑特征"""
        features = np.zeros(20)
        
        # 情感特征
        emotions = album_info.get('emotion_categories', [])
        emotion_mapping = {'积极': 0, '消极': 1, '中性': 2, '兴奋': 3, '平静': 4}
        for emotion in emotions:
            if emotion in emotion_mapping:
                features[emotion_mapping[emotion]] = 1
        
        # 主题特征
        theme = album_info.get('theme_category', '')
        theme_mapping = {'历史': 5, '科技': 6, '文学': 7, '商业': 8, '教育': 9}
        if theme in theme_mapping:
            features[theme_mapping[theme]] = 1
        
        # 难度等级
        difficulty = album_info.get('difficulty_level', '中级')
        difficulty_mapping = {'初级': 0.3, '中级': 0.6, '高级': 1.0}
        features[10] = difficulty_mapping.get(difficulty, 0.6)
        
        # 目标年龄
        target_age = album_info.get('target_age', '18-60岁')
        if '18-25' in target_age:
            features[11] = 0.2
        elif '26-35' in target_age:
            features[11] = 0.4
        elif '36-50' in target_age:
            features[11] = 0.6
        elif '50+' in target_age:
            features[11] = 0.8
        
        # 内容长度
        content_length = len(album_info.get('content', ''))
        features[12] = min(content_length / 10000, 1.0)  # 归一化到[0,1]
        
        # 标签数量
        tags_count = len(album_info.get('tags', []))
        features[13] = min(tags_count / 10, 1.0)
        
        # 情感强度
        emotion_intensity = album_info.get('emotion_intensity', 5)
        features[14] = emotion_intensity / 10
        
        # 其他特征
        features[15:20] = np.random.random(5) * 0.1  # 随机噪声
        
        return features
    
    def _extract_ad_features(self, ad_info: Dict) -> np.ndarray:
        """提取广告特征"""
        features = np.zeros(15)
        
        # 行业特征
        industry = ad_info.get('industry', '')
        industry_mapping = {'科技': 0, '金融': 1, '教育': 2, '娱乐': 3, '健康': 4}
        if industry in industry_mapping:
            features[industry_mapping[industry]] = 1
        
        # 品牌知名度
        brand_fame = ad_info.get('brand_fame', 0.5)
        features[5] = brand_fame
        
        # 产品价格
        price = ad_info.get('price', 100)
        features[6] = min(price / 1000, 1.0)  # 归一化
        
        # 目标受众匹配度
        target_match = ad_info.get('target_match_score', 0.5)
        features[7] = target_match
        
        # 广告创意质量
        creative_quality = ad_info.get('creative_quality', 0.5)
        features[8] = creative_quality
        
        # 投放预算
        budget = ad_info.get('budget', 1000)
        features[9] = min(budget / 10000, 1.0)
        
        # 其他特征
        features[10:15] = np.random.random(5) * 0.1
        
        return features
    
    def _extract_user_features(self, user_context: Dict) -> np.ndarray:
        """提取用户特征"""
        features = np.zeros(10)
        
        # 用户年龄
        age = user_context.get('age', 30)
        features[0] = age / 100
        
        # 用户性别
        gender = user_context.get('gender', 'unknown')
        if gender == 'male':
            features[1] = 1.0
        elif gender == 'female':
            features[1] = 0.0
        
        # 用户兴趣
        interests = user_context.get('interests', [])
        features[2] = min(len(interests) / 10, 1.0)
        
        # 历史点击率
        historical_ctr = user_context.get('historical_ctr', 0.02)
        features[3] = historical_ctr
        
        # 历史完播率
        historical_completion = user_context.get('historical_completion', 0.3)
        features[4] = historical_completion
        
        # 活跃度
        activity_level = user_context.get('activity_level', 0.5)
        features[5] = activity_level
        
        # 其他特征
        features[6:10] = np.random.random(4) * 0.1
        
        return features
    
    def _extract_history_features(self, performance_history: List[Dict]) -> np.ndarray:
        """提取历史性能特征"""
        features = np.zeros(5)
        
        if not performance_history:
            return features
        
        # 最近5次的平均性能
        recent_performance = performance_history[-5:]
        
        # 平均点击率
        avg_ctr = np.mean([p.get('click_rate', 0) for p in recent_performance])
        features[0] = avg_ctr
        
        # 平均完播率
        avg_completion = np.mean([p.get('completion_rate', 0) for p in recent_performance])
        features[1] = avg_completion
        
        # 平均转化率
        avg_conversion = np.mean([p.get('conversion_rate', 0) for p in recent_performance])
        features[2] = avg_conversion
        
        # 性能趋势
        if len(recent_performance) >= 2:
            recent_ctr = recent_performance[-1].get('click_rate', 0)
            previous_ctr = recent_performance[-2].get('click_rate', 0)
            features[3] = recent_ctr - previous_ctr
        
        # 性能稳定性
        ctr_std = np.std([p.get('click_rate', 0) for p in recent_performance])
        features[4] = 1.0 / (1.0 + ctr_std)  # 稳定性越高，值越大
        
        return features
    
    def optimize_deployment_strategy(self, album_info: Dict, ad_info: Dict, 
                                   user_context: Dict, performance_history: List[Dict]) -> Dict:
        """优化投放策略"""
        # 构建当前状态
        current_state = self.get_state_representation(album_info, ad_info, user_context, performance_history)
        
        # 选择动作
        action = self.agent.select_action(current_state)
        
        # 构建投放策略
        deployment_strategy = self._build_deployment_strategy(action, album_info, ad_info)
        
        # 记录策略
        strategy_record = {
            'timestamp': datetime.now().isoformat(),
            'album_id': album_info.get('album_id'),
            'ad_id': ad_info.get('ad_id'),
            'user_id': user_context.get('user_id'),
            'state': current_state.tolist(),
            'action': action.tolist(),
            'strategy': deployment_strategy,
            'performance': None  # 将在后续更新
        }
        
        self.deployment_history.append(strategy_record)
        
        return deployment_strategy
    
    def _build_deployment_strategy(self, action: np.ndarray, 
                                  album_info: Dict, ad_info: Dict) -> Dict:
        """根据动作构建投放策略"""
        strategy = {}
        
        # 投放时机 (action[0])
        timing_score = (action[0] + 1) / 2  # 转换到[0,1]
        if timing_score < 0.3:
            strategy['timing'] = 'early'
        elif timing_score < 0.7:
            strategy['timing'] = 'middle'
        else:
            strategy['timing'] = 'late'
        
        # 投放频率 (action[1])
        frequency_score = (action[1] + 1) / 2
        strategy['frequency'] = max(1, int(frequency_score * 5))
        
        # 创意版本选择 (action[2])
        creative_score = (action[2] + 1) / 2
        if creative_score < 0.33:
            strategy['creative_version'] = 'version_a'
        elif creative_score < 0.66:
            strategy['creative_version'] = 'version_b'
        else:
            strategy['creative_version'] = 'version_c'
        
        # 受众定向强度 (action[3])
        targeting_score = (action[3] + 1) / 2
        strategy['targeting_strength'] = targeting_score
        
        # 预算分配比例 (action[4])
        budget_ratio = (action[4] + 1) / 2
        strategy['budget_allocation'] = budget_ratio
        
        # 其他策略参数
        strategy['bid_adjustment'] = (action[5] + 1) / 2
        strategy['placement_priority'] = (action[6] + 1) / 2
        strategy['creative_rotation'] = (action[7] + 1) / 2
        strategy['audience_expansion'] = (action[8] + 1) / 2
        strategy['cross_device_targeting'] = (action[9] + 1) / 2
        
        return strategy
    
    def update_performance(self, strategy_id: str, performance_metrics: Dict):
        """更新投放策略的性能指标"""
        # 查找对应的策略记录
        for record in self.deployment_history:
            if record.get('id') == strategy_id:
                record['performance'] = performance_metrics
                break
        
        # 计算奖励
        reward = self._calculate_reward(performance_metrics)
        
        # 更新性能指标
        self.performance_metrics.update(performance_metrics)
        
        return reward
    
    def _calculate_reward(self, performance_metrics: Dict) -> float:
        """计算奖励值"""
        # 基础奖励
        reward = 0.0
        
        # 点击率奖励 (权重: 0.3)
        ctr = performance_metrics.get('click_rate', 0)
        reward += ctr * 0.3
        
        # 完播率奖励 (权重: 0.25)
        completion_rate = performance_metrics.get('completion_rate', 0)
        reward += completion_rate * 0.25
        
        # 转化率奖励 (权重: 0.25)
        conversion_rate = performance_metrics.get('conversion_rate', 0)
        reward += conversion_rate * 0.25
        
        # 收入奖励 (权重: 0.2)
        revenue = performance_metrics.get('revenue', 0)
        reward += min(revenue / 1000, 1.0) * 0.2
        
        return reward
    
    def train_agent(self, episodes: int = 1000):
        """训练智能体"""
        for episode in range(episodes):
            episode_reward = 0
            episode_length = 0
            
            # 模拟一个完整的投放周期
            while episode_length < 100:  # 最大步数限制
                # 这里应该从实际环境中获取状态
                # 为了演示，我们使用随机状态
                state = np.random.random(self.state_dim)
                
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行动作并获取奖励和下一状态
                # 这里应该调用实际的投放系统
                reward = np.random.random() * 0.1  # 模拟奖励
                next_state = np.random.random(self.state_dim)
                done = episode_length >= 99
                
                # 存储经验
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # 更新网络
                self.agent.update_networks()
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # 记录训练统计
            self.agent.training_stats['episode_rewards'].append(episode_reward)
            self.agent.training_stats['episode_lengths'].append(episode_length)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.agent.training_stats['episode_rewards'][-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
    
    def save_optimizer(self, save_path: str):
        """保存优化器"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存DDPG模型
        self.agent.save_model(save_path)
        
        # 保存部署历史
        with open(os.path.join(save_path, 'deployment_history.json'), 'w') as f:
            json.dump(self.deployment_history, f, indent=2, ensure_ascii=False)
        
        # 保存性能指标
        with open(os.path.join(save_path, 'performance_metrics.json'), 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, ensure_ascii=False)
    
    def load_optimizer(self, load_path: str):
        """加载优化器"""
        # 加载DDPG模型
        self.agent.load_model(load_path)
        
        # 加载部署历史
        history_path = os.path.join(load_path, 'deployment_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.deployment_history = json.load(f)
        
        # 加载性能指标
        metrics_path = os.path.join(load_path, 'performance_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.performance_metrics = json.load(f)
