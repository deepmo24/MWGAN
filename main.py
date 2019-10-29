import argparse
import yaml
from torch.backends import cudnn

from trainer import Trainer
from data_loader import get_loader
from utils import *

parser = argparse.ArgumentParser()

# basic opts.
parser.add_argument('--num_domains', type=int, default=5, help='how many domains(including source domain)')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test-single', 'generate'])
parser.add_argument('--use_tensorboard', type=str2bool, default=True)
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--checkpoint_dir', type=str, default='models/')
parser.add_argument('--sample_dir', type=str, default='samples/')
parser.add_argument('--result_dir', type=str, default='results/')
parser.add_argument('--result_root', type=str, default='results_mwgan/')
# data opts.
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--src_domain', type=str, default='domain0/', help='a sub-dir of data_root representing the source domain.')
parser.add_argument('--crop', type=str2bool, default=True)
parser.add_argument('--crop_size', type=int, default=178, help='crop size')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers to load data.')
# model opts.
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=3, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss to update D')
parser.add_argument('--lambda_info', type=float, default=1, help='weight for mutual information maximization to update G')
parser.add_argument('--lambda_reg', type=float, default=100, help='weight for regularization')
parser.add_argument('--lambda_idt', type=float, default=0, help='weight for identity loss')
parser.add_argument('--Lf', type=float, default=1, help='a constant with respect to the inter-domain constraint')
parser.add_argument('--cls_loss', type=str, default='LS', choices=['LS', 'BCE'], help='least square loss or binary cross entropy loss')
# Training opts.
parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=500)
parser.add_argument('--model_save_step', type=int, default=10000)
# Test opts.
parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

opts = parser.parse_args()
opts.log_dir = os.path.join(opts.result_root, opts.log_dir)
opts.checkpoint_dir = os.path.join(opts.result_root, opts.checkpoint_dir)
opts.sample_dir = os.path.join(opts.result_root, opts.sample_dir)
opts.result_dir = os.path.join(opts.result_root, opts.result_dir)


if __name__ == '__main__':

    # For fast training.
    cudnn.benchmark = True

    if opts.mode == 'train':
        # Create directories if not exist.
        create_dirs_if_not_exist([opts.log_dir, opts.checkpoint_dir, opts.sample_dir, opts.result_dir])

        # log opts.
        with open(os.path.join(opts.result_root, 'opts.yaml'), 'w') as f:
            f.write(yaml.dump(vars(opts)))

        # Load data: data_root must contain sub-dirs of each domain.
        data_dirs = os.listdir(opts.data_root)
        data_dirs.sort()  # sort
        if (len(data_dirs) != opts.num_domains) or (opts.src_domain not in data_dirs):
            print(data_dirs)
            raise ValueError('Please make your data organization be consistent with the input.')

        tgt_loaders = []
        for domian_dir in data_dirs:
            domain_path = os.path.join(opts.data_root, domian_dir)
            if domian_dir == opts.src_domain:
                src_loader = get_loader(domain_path, opts.crop, opts.crop_size, opts.image_size, opts.batch_size, opts.mode, opts.num_workers)
            else:
                tgt_loader = get_loader(domain_path, opts.crop, opts.crop_size, opts.image_size, opts.batch_size, opts.mode, opts.num_workers)
                tgt_loaders.append(tgt_loader)

        # Build trainer and train.
        trainer = Trainer(src_loader, tgt_loaders, opts)
        trainer.train()

    elif opts.mode == 'test':
        # Create directories if not exist.
        create_dirs_if_not_exist([opts.result_dir])
        # Load data
        domain_path = os.path.join(opts.data_root,opts.src_domain)
        src_loader = get_loader(domain_path, opts.crop, opts.crop_size, opts.image_size, opts.batch_size, opts.mode, opts.num_workers)

        # Build trainer and test.
        trainer = Trainer(src_loader, None, opts)
        trainer.test()