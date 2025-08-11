import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """配置加载器 - 负责加载和管理系统配置"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            else:
                print(f"配置文件不存在: {self.config_path}")
                self.config = self._get_default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self.config = self._get_default_config()
    
    def get_config(self, key_path: str = None) -> Any:
        """获取配置值
        
        Args:
            key_path: 配置键路径，如 'llm_api.base_url'
            
        Returns:
            配置值
        """
        if key_path is None:
            return self.config
        
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def set_config(self, key_path: str, value: Any):
        """设置配置值
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 遍历到最后一个键的父级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置最后一个键的值
        config[keys[-1]] = value
    
    def save_config(self, save_path: str = None):
        """保存配置到文件"""
        if save_path is None:
            save_path = self.config_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def reload_config(self):
        """重新加载配置文件"""
        self.load_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'system': {
                'name': '有声书广告匹配Pipeline',
                'version': '1.0.0',
                'debug': True,
                'log_level': 'INFO'
            },
            'llm_api': {
                'base_url': 'http://deepgate.ximalaya.local',
                'api_key': '161332424d5d43649599351e2a20f0f1',
                'model': 'gpt-4o',
                'max_tokens': 8092,
                'timeout': 30
            },
            'storage': {
                'data_dir': 'data',
                'models_dir': 'models',
                'logs_dir': 'logs',
                'cache_dir': 'cache'
            }
        }
    
    def get_storage_path(self, path_type: str) -> str:
        """获取存储路径
        
        Args:
            path_type: 路径类型 ('data', 'models', 'logs', 'cache')
            
        Returns:
            完整路径
        """
        base_path = self.get_config(f'storage.{path_type}_dir')
        if base_path:
            full_path = os.path.join(os.getcwd(), base_path)
            os.makedirs(full_path, exist_ok=True)
            return full_path
        return None
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        required_keys = [
            'system.name',
            'llm_api.base_url',
            'llm_api.api_key',
            'storage.data_dir'
        ]
        
        for key in required_keys:
            if self.get_config(key) is None:
                print(f"缺少必需的配置项: {key}")
                return False
        
        return True

# 全局配置实例
config_loader = ConfigLoader()
