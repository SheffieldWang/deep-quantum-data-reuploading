import jax 
import jax.numpy as jnp  
import optax
from flax import nnx
from metric import Metrics,MetricComputer
from tqdm.auto import tqdm
import wandb
import os 
import logging
from datetime import datetime
from .loss_fn import mse_loss_fn,hinge_loss_fn,cross_entropy_loss_fn


@nnx.jit(static_argnames=['loss_fn'])
def train_step(model,optimizer,batch,loss_fn):
    data,target = batch
    (loss,predictions), grads = nnx.value_and_grad(loss_fn,has_aux=True)(model,data,target)
    optimizer.update(grads)
    return loss,predictions

@nnx.jit(static_argnames=['loss_fn'])
def eval_step(model,batch,loss_fn):
    data,target = batch
    (loss,predictions), _ = nnx.value_and_grad(loss_fn,has_aux=True)(model,data,target)
    return loss,predictions


import logging
import wandb
from tqdm import tqdm
import jax.numpy as jnp
from typing import Dict, List, Tuple

class BaseTrainer():
    def __init__(self, config, model, train_loader, test_loader):
        """
        训练器基类
        
        参数:
        - config: 包含训练配置的字典，必须包含:
            * n_epochs: 训练轮数
            * project_name: wandb 项目名称 (如果use_wandb=True)
            * use_wandb: 是否使用wandb记录实验
            * save_epoch_metrics: 是否存储每个epoch的详细指标 (新增)
        - model: 要训练的模型
        - train_loader: 训练数据加载器
        - test_loader: 测试数据加载器
        - loss_fn: 损失函数
        """
        self.config = config
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.additional_metrics = []
        self.best_model = None
        self.monitor_metric_name = None
        self.monitor_mode = 'max'  # 'max' means higher is better, 'min' means lower is better
        if self.monitor_mode == 'max':
            self.best_monitor_metric = -float('inf')
        else:
            self.best_monitor_metric = float('inf')
        
        self.epoch_history = {
            'train': [],
            'test': []
        }
        
        
        self.metrics = Metrics()
        self.metric_computer = MetricComputer()

        
        # 初始化基础组件
        self.setup_logging()
        self.setup_config()
        self.setup_loss_metrics()




    def setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)


    def setup_loss_metrics(self):
        for split in ["train", "test"]:
            self.metrics.register_metric("loss", split, "epoch")

    def setup_config(self):
        self.epochs = self.config['n_epochs']
        self.learning_rate = self.config['learning_rate']
        self.optimizer = self.config['optimizer']
        self.loss_fn = self.config['loss_fn']
        self.use_wandb = self.config.get('use_wandb', False)
        self.save_epoch_metrics = self.config.get('save_epoch_metrics', False)
        self.save_best_model = self.config.get('save_best_model', False)
        self.test_every_epoch = self.config.get('test_every_epoch', False)
        self.setup_optimizer()
        self.setup_loss_fn()
        
    def setup_optimizer(self):
        opt_map = {
            'adam': optax.adam,
            'sgd': optax.sgd,
            'rmsprop': optax.rmsprop,
            'adagrad': optax.adagrad
        } 
        opt_func = opt_map.get(self.optimizer)
        if not opt_func:
            raise ValueError(f"无效优化器: {self.optimizer}")
        self.optimizer = nnx.Optimizer(self.model, opt_func(self.learning_rate))
        
    def setup_loss_fn(self):
        loss_fn_map = {
            'mse': mse_loss_fn,
            'hinge': hinge_loss_fn,
            'cross_entropy': cross_entropy_loss_fn
        } 
        loss_fn_func = loss_fn_map.get(self.loss_fn)
        if not loss_fn_func:
            raise ValueError(f"无效损失函数: {self.loss_fn}")
        self.loss_fn = loss_fn_func

    def _update_best_model(self,monitor_metric):
        if self.monitor_mode == 'max':
            if monitor_metric > self.best_monitor_metric:
                self.best_monitor_metric = monitor_metric
                self.best_model = self.model
        else:
            if monitor_metric < self.best_monitor_metric:
                self.best_monitor_metric = monitor_metric
                self.best_model = self.model
                    
    def run_epoch(self, epoch: int, split: str = "train") -> Dict[str, float]:
        """
        执行完整epoch并返回指标字典
        
        参数:
        - epoch: 当前epoch序号 (从0开始)
        - split: 执行模式，支持 "train" 或 "test"
        
        返回:
        - epoch_metrics: 包含本epoch所有指标的字典
        """
        
        # data loader
        data_loader = self.train_loader if split == "train" else self.test_loader
        
        # metrics init
        total_loss = 0.0
        metric_sums = {name: 0.0 for name in self.additional_metrics}
        
        # desc
        desc = f"{split.capitalize()} Epoch {epoch+1}/{self.epochs}"
        with tqdm(data_loader, desc=desc, postfix={"loss": 0.0}, leave=False) as pbar:
            for batch in pbar:
                _, targets = batch
                
                # 执行训练/评估步骤
                if split == "train":
                    loss, outputs = train_step(self.model, self.optimizer, batch, self.loss_fn)
                else:
                    loss, outputs = eval_step(self.model, batch, self.loss_fn)
                
                
                # compute metrics
                total_loss += loss
                for metric_name in self.additional_metrics:
                    metric_sums[metric_name] += self.metric_computer.compute_metric(metric_name, outputs, targets)

        
        # compute avg metrics
        avg_metrics = {
            "loss": total_loss / len(data_loader),
            **{
                metric_name: metric_sums[metric_name] / len(data_loader)
                for metric_name in self.additional_metrics
            }
        }
        
        # update global metrics recorder
        for name, value in avg_metrics.items():
            self.metrics.update(name, value, split=split, index_type="epoch")
        
        
        if self.save_epoch_metrics:
            self.epoch_history[split].append(avg_metrics)
            
        return avg_metrics

    def train(self) -> Tuple[any, Dict, Dict]:
        """
        完整训练流程
        
        返回:
        - model: 最终模型
        - final_metrics: 最终指标字典
        - epoch_metrics: 每个epoch的详细指标（如果启用）
        """
        if self.use_wandb:
            os.makedirs(f"../wandb/{self.config['project_name']}", exist_ok=True)
            wandb.init(
                project=self.config["project_name"],
                name=f'run_{self.config["seed"]}',  # 设置此次run的名字，如果config中没有则为None
                config=self.config,
                dir=f"../wandb/{self.config['project_name']}"  # 设置wandb文件存储路径，默认为./wandb，不存在则创建
            )
        

        try:
            for epoch in tqdm(range(self.epochs), desc='Training epochs', leave=True):
                # 训练并获取当前epoch指标
                train_metrics = self.run_epoch(epoch, "train")
                
                if self.test_every_epoch:
                    test_metrics = self.run_epoch(epoch, "test")
                
                self._update_best_model(train_metrics[self.monitor_metric_name])
                
                # 记录到wandb
                if self.use_wandb:
                    wandb.log({
                        f"train_{name}": value
                        for name, value in train_metrics.items()
                    })
                    if self.test_every_epoch:
                        wandb.log({
                            f"test_{name}": value
                            for name, value in test_metrics.items()
                        })
                
                
            # 获取最终指标
            if self.save_best_model:
                # Get the best index based on monitor metric
                monitor_metric_data = self.metrics.get_metric(self.monitor_metric_name, "train")
                if self.monitor_mode == 'max':
                    best_idx = monitor_metric_data['values'].index(monitor_metric_data['stats']['max'])
                else:
                    best_idx = monitor_metric_data['values'].index(monitor_metric_data['stats']['min'])
                
                # Get metrics at best index
                final_train_metrics = {
                    name: self.metrics.get_metric(name, "train", best_idx)
                    for name in ["loss"] + self.additional_metrics
                }
            else:
                final_train_metrics = {
                    name: self.metrics.get_metric(name, "train", -1)
                    for name in ["loss"] + self.additional_metrics
                }
            
            # 组装返回结果
            results = {
                "final_metrics": final_train_metrics,
                "epoch_metrics": {
                    "train": self.epoch_history['train'] if self.save_epoch_metrics else None,
                    "test": self.epoch_history['test'] if self.save_epoch_metrics and self.test_every_epoch else None
                }
            }
            
            self.logger.info(f"Training completed, final metrics: {final_train_metrics}")
            
            if self.save_best_model:
                return self.best_model, results
            else:
                return self.model, results
        
        finally:
            if self.use_wandb:
                wandb.finish()

    def test(self) -> Tuple[any, Dict, Dict]:
        """
        执行测试流程
        
        返回:
        - model: 最终模型
        - test_metrics: 测试指标字典
        - epoch_metrics: 测试的详细指标（如果启用）
        """
        # 执行测试epoch
        test_metrics = self.run_epoch(0, "test")
        
        # 获取最终指标
        final_test_metrics = {
            name: self.metrics.get_metric(name, "test", -1)
            for name in ["loss"] + self.additional_metrics
        }
        
        # 组装返回结果
        results = {
            "final_metrics": final_test_metrics,
            "epoch_metrics": self.epoch_history['test'] if self.save_epoch_metrics else None
        }
        
        self.logger.info(f"Testing completed, final metrics: {final_test_metrics}")
        return self.model, results

class RegressionTrainer(BaseTrainer):
    """ 回归任务训练器 """
    def __init__(self, config, model, train_loader, test_loader):
        super().__init__(config, model, train_loader, test_loader)
        self.additional_metrics = ['error']
        self.monitor_metric_name = 'error'
        self.monitor_mode = 'min'
        
        
        self.setup_metrics()

    def setup_metrics(self):
        for split in ["train", "test"]:
            self.metrics.register_metric("error", split, "epoch")


class ClassificationTrainer(BaseTrainer):
    """ 分类任务训练器 """
    def __init__(self, config, model, train_loader, test_loader):
        super().__init__(config, model, train_loader, test_loader)
        self.additional_metrics = ['accuracy','pred_error']
        self.monitor_metric_name = 'accuracy'
        self.monitor_mode = 'max'
        
        self.setup_metrics()
        
    def setup_metrics(self):
        for split in ["train", "test"]:
            self.metrics.register_metric("accuracy", split, "epoch")
            self.metrics.register_metric("pred_error", split, "epoch")
