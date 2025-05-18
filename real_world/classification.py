import sys
import os
current_dir = os.getcwd()  # Get current working directory
parent_dir = os.path.dirname(current_dir)  # Get parent directory
sys.path.append(parent_dir)

import pandas as pd
import pennylane as qml
import pickle

import logging 
from datetime import datetime
from tqdm import *
import argparse

import jax 
import jax.numpy as jnp  
import optax
from flax import nnx 


from datasets_utils import get_quantum_dataloaders
from model import DataReuploading
from metric import Metrics
from train_utils import ClassificationTrainer
import wandb


jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)





class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.results_df = pd.DataFrame()
        self.metrics = Metrics()
        self.model_list = []
        
        self.setup_config()
        self.setup_metrics()
        
       
    def setup_config(self):
        self.n_repeats = self.config['n_repeats']
     
    def setup_metrics(self):
        self.metrics.register_metric("loss",split="train",index_type="repeat")
        self.metrics.register_metric("accuracy",split="train",index_type="repeat")
        self.metrics.register_metric("pred_error",split="train",index_type="repeat")
        self.metrics.register_metric("loss",split="test",index_type="repeat")
        self.metrics.register_metric("accuracy",split="test",index_type="repeat")
        self.metrics.register_metric("pred_error",split="test",index_type="repeat")
        
        
        
    def run_experiments(self):
        """运行多次实验"""
        data = {}
        for i in range(self.n_repeats):
            print(f"\nRunning experiment {i+1}/{self.n_repeats}")
            # 每次实验使用不同的随机种子
            seed = i
            self.config['seed'] = seed
            qnet = DataReuploading(n_qubits=n_qubits, n_reps=n_reps, n_layers=n_layers,max_layers=max_layers,measurement_type="probs",measure_wires = [0],seed=seed,ansatz_type="zero_padding")
            trainer = ClassificationTrainer(self.config,qnet,train_loader,test_loader)
            qnet,train_metrics = trainer.train()
            _,test_metrics = trainer.test()
            self.metrics.update("loss", train_metrics['final_metrics']['loss'], split="train", index_type="repeat")
            self.metrics.update("accuracy", train_metrics['final_metrics']['accuracy'], split="train", index_type="repeat")
            self.metrics.update("pred_error", train_metrics['final_metrics']['pred_error'], split="train", index_type="repeat")
            self.metrics.update("loss", test_metrics['final_metrics']['loss'], split="test", index_type="repeat")
            self.metrics.update("accuracy", test_metrics['final_metrics']['accuracy'], split="test", index_type="repeat")
            self.metrics.update("pred_error", test_metrics['final_metrics']['pred_error'], split="test", index_type="repeat")
            
            model_params = qnet.quantum_model.get_params()
            self.model_list.append(model_params)
            
            
            
            train_accuracy = [train_metrics['epoch_metrics']['train'][k]['accuracy'].item() for k in range(len(train_metrics['epoch_metrics']['train']))]
            test_accuracy = [train_metrics['epoch_metrics']['test'][k]['accuracy'].item() for k in range(len(train_metrics['epoch_metrics']['test']))]

            data[f'train_accuracy_{i}'] = train_accuracy
            data[f'test_accuracy_{i}'] = test_accuracy[:-1]
            

            
        # Create DataFrame with results
        


        self.results_df = pd.DataFrame(data)

        
        print("\n" + "="*50)
        print("✨ All experiments completed successfully! ✨")
        print("="*50 + "\n")



        # 保存结果
        self.save_results()
        
    
    def save_results(self):
        """保存实验结果"""
        results_dir = '../results'
        os.makedirs(f'{results_dir}/{self.config["project_name"]}', exist_ok=True)
        # 保存DataFrame为CSV
        csv_path = f'{results_dir}/{self.config["project_name"]}/experiment_results_{self.config["group_name"]}.csv'
        self.results_df.to_csv(csv_path, index=False)
        
        # 保存所有数据（包括参数）到pickle文件
        full_results = {
            'config': self.config,
            'model_list': self.model_list
        }
        pickle_path = f'{results_dir}/{self.config["project_name"]}/full_results_{self.config["group_name"]}.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(full_results, f)
        
        print(f"Results saved to {csv_path} and {pickle_path}")   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum Regression Model Parameters')
    parser.add_argument('--n_qubits', type=int, default=1,
                        help='Number of qubits (default: 1)')
    parser.add_argument('--n_layers', type=int, default=8,
                        help='Number of quantum circuit layers (default: 8)') 
    parser.add_argument('--max_layers', type=int, default=8,
                        help='Maximum number of quantum circuit layers for parameters (default: 8)')
    parser.add_argument('--n_reps', type=int, default=8,
                        help='Number of repetitions of quantum circuit layers (default: 8)')
    parser.add_argument('--n_samples', type=int, default=600,
                        help='Number of samples (default: 600)')
    parser.add_argument('--n_test', type=int, default=10000,
                        help='Number of samples (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='Number of epochs (default: 1000)')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of repeats of experiment (default: 10)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--data_type', type=str, default="linear",
                        help='Data type (default: linear)')
    
    
    args = parser.parse_args()

    n_qubits = args.n_qubits
    n_layers = args.n_layers  
    max_layers = args.max_layers
    n_reps = args.n_reps
    n_samples = args.n_samples
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs
    n_repeats = args.n_repeats
    seed = args.seed
    data_type = args.data_type
    n_test = args.n_test
    
    
    config = {
    'n_qubits': n_qubits,
    
    'n_layers': n_layers,
    'max_layers':max_layers,
    'n_reps': n_reps,
    'optimizer': 'adam',
    'loss_fn': 'cross_entropy',
    'batch_size': batch_size,
    'learning_rate': lr,
    'n_epochs': n_epochs,
    'n_repeats': n_repeats,
    'seed':seed,
    'use_wandb': True,
    'save_epoch_metrics': True,
    'test_every_epoch': True,
    'save_best_model': True,
    'project_name': f'Classification_{data_type}',
    'group_name': f'qubits_{n_qubits}_layers_{n_layers}_reps_{n_reps}_samples_{n_samples}'
}
    
    train_loader, test_loader = get_quantum_dataloaders(n_qubits=n_qubits, n_layers=n_layers, n_samples=n_samples, data_type=data_type,batch_size=batch_size,n_test=n_test)
    
    experiment_manager = ExperimentManager(config)
    experiment_manager.run_experiments()
