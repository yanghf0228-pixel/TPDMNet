import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
from datetime import datetime
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader, get_adjacency_matrix2
from lib.TrainInits import print_model_parameters
from model.TPDMNet import TPDMNet as Network
import warnings
warnings.filterwarnings('ignore')


#*************************************************************************#

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        # if scaler:
        #     preds = scaler.inverse_transform(preds)
        #     labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss



#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default='PEMSD8', type=str)
args.add_argument('--type', default='P', type=str)
args.add_argument('--mode', default='train', type=str)
args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
args.add_argument('--debug', default='False', type=eval)
args.add_argument('--model', default='TPDMNet', type=str)
args.add_argument('--cuda', default=True, type=bool)
args1, _ = args.parse_known_args()

#get configuration
config_file = './config_file/{}/{}_{}.conf'.format(args1.type,args1.dataset, args1.model)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

def get_config_value(section, key, default, cast):
    if config.has_option(section, key):
        return cast(config[section][key])
    return default


#data
args.add_argument('--val_ratio', default=get_config_value('data', 'val_ratio', 0.2, float), type=float)
args.add_argument('--test_ratio', default=get_config_value('data', 'test_ratio', 0.2, float), type=float)
args.add_argument('--lag', default=get_config_value('data', 'lag', 12, int), type=int)
args.add_argument('--horizon', default=get_config_value('data', 'horizon', 12, int), type=int)
args.add_argument('--num_nodes', default=get_config_value('data', 'num_nodes', 170, int), type=int)
args.add_argument('--tod', default=get_config_value('data', 'tod', False, eval), type=eval)
args.add_argument('--normalizer', default=get_config_value('data', 'normalizer', 'std', str), type=str)
args.add_argument('--column_wise', default=get_config_value('data', 'column_wise', False, eval), type=eval)
args.add_argument('--default_graph', default=get_config_value('data', 'default_graph', True, eval), type=eval)
args.add_argument('--steps_per_day', default=get_config_value('data', 'steps_per_day', 288, int), type=int)
args.add_argument('--steps_per_week', default=get_config_value('data', 'steps_per_week', 7, int), type=int)
#model
args.add_argument('--input_dim', default=get_config_value('model', 'input_dim', 1, int), type=int)
args.add_argument('--output_dim', default=get_config_value('model', 'output_dim', 1, int), type=int)
args.add_argument('--time_dim', default=get_config_value('model', 'time_dim', 20, int), type=int)
args.add_argument('--embed_dim', default=get_config_value('model', 'embed_dim', 10, int), type=int)
args.add_argument('--rnn_units', default=get_config_value('model', 'rnn_units', 64, int), type=int)
args.add_argument('--num_layers', default=get_config_value('model', 'num_layers', 1, int), type=int)
args.add_argument('--cheb_k', default=get_config_value('model', 'cheb_order', 2, int), type=int)
args.add_argument('--use_day', default=get_config_value('model', 'use_day', True, eval), type=eval)
args.add_argument('--use_week', default=get_config_value('model', 'use_week', True, eval), type=eval)
args.add_argument('--dropout', default=get_config_value('model', 'dropout', 0.1, float), type=float)
args.add_argument('--memory_retrieval', default=get_config_value('model', 'memory_retrieval', 'global', str), type=str)
args.add_argument('--num_prototypes', default=get_config_value('model', 'num_prototypes', 64, int), type=int)
args.add_argument('--memory_temperature', default=get_config_value('model', 'memory_temperature', 0.5, float), type=float)
args.add_argument('--memory_gate_bias', default=get_config_value('model', 'memory_gate_bias', -3.0, float), type=float)
args.add_argument('--memory_scale', default=get_config_value('model', 'memory_scale', 0.85, float), type=float)
#train
args.add_argument('--loss_func', default=get_config_value('train', 'loss_func', 'mae', str), type=str)
args.add_argument('--seed', default=get_config_value('train', 'seed', 10, int), type=int)
args.add_argument('--batch_size', default=get_config_value('train', 'batch_size', 64, int), type=int)
args.add_argument('--epochs', default=get_config_value('train', 'epochs', 500, int), type=int)
args.add_argument('--lr_init', default=get_config_value('train', 'lr_init', 0.0015, float), type=float)
args.add_argument('--weight_decay', default=get_config_value('train', 'weight_decay', 1e-4, float), type=float)
args.add_argument('--lr_decay', default=get_config_value('train', 'lr_decay', True, eval), type=eval)
args.add_argument('--lr_decay_rate', default=get_config_value('train', 'lr_decay_rate', 0.3, float), type=float)
args.add_argument('--lr_decay_step', default=get_config_value('train', 'lr_decay_step', '60,90,120', str), type=str)
args.add_argument('--early_stop', default=get_config_value('train', 'early_stop', True, eval), type=eval)
args.add_argument('--early_stop_patience', default=get_config_value('train', 'early_stop_patience', 30, int), type=int)
args.add_argument('--grad_norm', default=get_config_value('train', 'grad_norm', True, eval), type=eval)
args.add_argument('--max_grad_norm', default=get_config_value('train', 'max_grad_norm', 5.0, float), type=float)
args.add_argument('--memory_diversity_weight', default=get_config_value('train', 'memory_diversity_weight', 0.01, float), type=float)
args.add_argument('--memory_usage_weight', default=get_config_value('train', 'memory_usage_weight', 0.0, float), type=float)
args.add_argument('--delta_loss_weight', default=get_config_value('train', 'delta_loss_weight', 0.0, float), type=float)
if args1.type == 'R':
    args.add_argument('--teacher_forcing', default=config['train']['teacher_forcing'], type=eval)
    args.add_argument('--teacher_decay_step', default=config['train']['teacher_decay_step'], type=int)
args.add_argument('--real_value', default=get_config_value('train', 'real_value', True, eval), type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=get_config_value('test', 'mae_thresh', None, eval), type=eval)
args.add_argument('--mape_thresh', default=get_config_value('test', 'mape_thresh', 0.0, float), type=float)
# Additional TPDMNet compatibility parameters
try:
    args.add_argument('--dev_hidden_dim', default=config['model']['dev_hidden_dim'], type=int)
except KeyError:
    args.add_argument('--dev_hidden_dim', default=32, type=int)
try:
    args.add_argument('--ma_kernel', default=config['model']['ma_kernel'], type=int)
except KeyError:
    args.add_argument('--ma_kernel', default=3, type=int)
try:
    args.add_argument('--use_dev_graph', default=config['model']['use_dev_graph'], type=eval)
except KeyError:
    args.add_argument('--use_dev_graph', default=False, type=eval)
try:
    args.add_argument('--dev_graph_mode', default=config['model']['dev_graph_mode'], type=str)
except KeyError:
    args.add_argument('--dev_graph_mode', default='none', type=str)
try:
    args.add_argument('--use_dynamic_coupling', default=config['model']['use_dynamic_coupling'], type=eval)
except KeyError:
    args.add_argument('--use_dynamic_coupling', default=True, type=eval)
try:
    args.add_argument('--lambda_dyn', default=config['model']['lambda_dyn'], type=float)
except KeyError:
    args.add_argument('--lambda_dyn', default=0.1, type=float)
try:
    args.add_argument('--return_pattern_feat', default=config['model']['return_pattern_feat'], type=eval)
except KeyError:
    args.add_argument('--return_pattern_feat', default=True, type=eval)
try:
    args.add_argument('--kan_hidden_dim', default=config['model']['kan_hidden_dim'], type=int)
except KeyError:
    args.add_argument('--kan_hidden_dim', default=0, type=int)
try:
    args.add_argument('--bilateral_gate_hidden_dim', default=config['model']['bilateral_gate_hidden_dim'], type=int)
except KeyError:
    args.add_argument('--bilateral_gate_hidden_dim', default=4, type=int)
try:
    args.add_argument('--dynamic_coupling_debug', default=config['model']['dynamic_coupling_debug'], type=eval)
except KeyError:
    args.add_argument('--dynamic_coupling_debug', default=False, type=eval)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=get_config_value('log', 'log_step', 10000, int), type=int)
args.add_argument('--plot', default=get_config_value('log', 'plot', False, eval), type=eval)
args = args.parse_args()

print(args)
if not 0.0 <= args.dropout <= 1.0:
    raise ValueError(f"dropout must be in [0, 1], got {args.dropout}")
print(
    "Regularization settings: "
    f"weight_decay={args.weight_decay}, "
    f"dropout={args.dropout}, "
    f"grad_norm={args.grad_norm}, "
    f"max_grad_norm={args.max_grad_norm}, "
    f"memory_diversity_weight={args.memory_diversity_weight}, "
    f"memory_usage_weight={args.memory_usage_weight}, "
    f"delta_loss_weight={args.delta_loss_weight}"
)
print("Prediction head: linear")
print(f"TPMR retrieval: memory_retrieval={args.memory_retrieval}")
print(
    "TPMR settings: "
    f"num_prototypes={args.num_prototypes}, "
    f"memory_temperature={args.memory_temperature}, "
    f"memory_gate_bias={args.memory_gate_bias}, "
    f"memory_scale={args.memory_scale}"
)

def find_physical_adjacency_path(dataset):
    candidates = {
        'PEMSD3': [
            './data/PeMS03/PEMS03.csv',
            './data/PEMS03/PEMS03.csv',
            './data/PeMS03/distance.csv',
        ],
        'PEMSD4': [
            './data/PeMS04/PEMS04.csv',
            './data/PEMS04/PEMS04.csv',
            './data/PeMS04/distance.csv',
        ],
        'PEMSD7': [
            './data/PeMS07/PEMS07.csv',
            './data/PEMS07/PEMS07.csv',
        ],
        'PEMSD8': [
            './data/PEMS08/PEMS08.csv',
            './data/PeMS08/PEMS08.csv',
            './data/PeMS08/distance.csv',
        ],
    }
    for path in candidates.get(dataset, []):
        if os.path.exists(path):
            return path
    return None


args.adj_mx = None
adj_path = find_physical_adjacency_path(args.dataset)
if adj_path is not None:
    args.adj_mx = get_adjacency_matrix2(adj_path, args.num_nodes, type_='connectivity')
    edge_count = int(np.count_nonzero(args.adj_mx))
    density = edge_count / float(args.num_nodes * args.num_nodes)
    print(
        "Physical adjacency loaded: "
        f"dataset={args.dataset}, path={adj_path}, "
        f"shape={args.adj_mx.shape}, nonzero={edge_count}, density={density:.6f}"
    )
else:
    print(f"Physical adjacency not found for {args.dataset}; using learned/dense graph fallback.")

init_seed(args.seed)
print(f"Seed initialized: seed={args.seed}")
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args)
print(
    "TPDMNet physical local mask: "
    f"has_local_mask={getattr(model, 'has_local_mask', False)}, "
    f"mask_nonzero={int(model.local_mask.sum().item()) if hasattr(model, 'local_mask') else 0}, "
    f"memory_retrieval={getattr(model, 'memory_retrieval', 'unknown')}"
)
model = model.to(args.device)
for name, p in model.named_parameters():
    if (
        name.endswith("residual_scale")
        or name == "local_global_fusion.gate_proj.bias"
    ):
        continue
    if name.endswith("norm.weight"):
        nn.init.ones_(p)
        continue
    if p.dim() == 1 and name.endswith("bias"):
        nn.init.zeros_(p)
        continue
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p, -0.1, 0.1)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, args.type, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    checkpoint = torch.load('./pre-trained/{}/{}.pth'.format(args.type,args.dataset), map_location=args.device)
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys:
        print('Missing keys while loading checkpoint:', incompatible.missing_keys)
    if incompatible.unexpected_keys:
        print('Unexpected keys while loading checkpoint:', incompatible.unexpected_keys)
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
