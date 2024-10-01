import argparse
import torch
import random
import numpy as np
from exp.exp_DADA import DADA


parser = argparse.ArgumentParser(description='DADA')
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
parser.add_argument('--model', type=str, default='DADA', help='model name')
parser.add_argument('--data', type=str, default='MSL', help='dataset type')
parser.add_argument('--root_path', type=str, default='/workspace/dataset/dataset', help='root path of the data file')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of input data')
parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
parser.add_argument('--patch_len', type=int, default=5, help='patch length')
parser.add_argument('--stride', type=int, default=5, help='stride')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--hidden_dim', type=int, default=64, help='for DADA')
parser.add_argument('--depth', type=int, default=10, help='for DADA')
parser.add_argument('--bn_dims', type=int, nargs="+", default=[8, 16, 32, 64, 128, 256], help='for DADA')
parser.add_argument('--k', type=int, default=3, help='for DADA')
parser.add_argument("--mask_mode", type=str, default='c', help="for DADA")
parser.add_argument('--copies', type=int, default=2, help='')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence')
parser.add_argument('--metric', type=str, default="affiliation", help="metric")
parser.add_argument('--q', type=float, nargs="+", default=[0.03], help="for SPOT")
parser.add_argument('--t', type=float, nargs="+", default=[0.06], help="threshold found by SPOT")
parser.add_argument('--max_iters', type=int, default=100000, help='for DADA')
parser.add_argument("--percentage", type=float, default=1, help="the percentage(*100) of train data")
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--des', type=str, default='zero_shot', help='exp description')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() else False
print(torch.cuda.is_available())
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

dada = DADA(args)
dada.zero_shot(setting=f"{args.data}")