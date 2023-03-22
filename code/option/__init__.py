import argparse
from option import template

parser = argparse.ArgumentParser(description='Video_Deblur')

parser.add_argument('--template', default='VIDUE_WORSU',
                    help='You can set various templates in options.py')   #CDVD_TSP  GPA SWINT

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=8,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/sw/sw/dataset/Event-data/GOPRO_large_all/train/images',
                    help='dataset directory')
# parser.add_argument('--dir_data2', type=str, default='/mnt/disk10T_2/shangwei/data/deblur/Random/train',
#                     help='dataset directory')
parser.add_argument('--dir_data_test', type=str, default='/home/sw/sw/dataset/Event-data/GOPRO_large_all/val/images',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='GOPRO',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='GOPRO',
                    help='test dataset name')
parser.add_argument('--process', action='store_true',
                    help='if True, load all dataset at once at RAM')
parser.add_argument('--patch_size', type=int, default=256+64+32,
                    help='output patch size')
parser.add_argument('--size_must_mode', type=int, default=1,
                    help='the size of the network input must mode this number')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--m', type=int, default=5,
                    help='Number of exposure frames')
parser.add_argument('--n', type=int, default=3,
                    help='Number of readout frames')
parser.add_argument('--random', action='store_true',
                    help='random create all kinds of [m,n] if true')
parser.add_argument('--halve', action='store_true',
                    help='halve xN if true')
parser.add_argument('--blur_deg', type=int, default=1,
                        help='Degree of blur')

# Model specifications
parser.add_argument('--model', default='.',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--offset_network_path', type=str, default='.',
                    help='pre-trained offset model directory')
model_choices = ["unet_18", "unet_34"]
parser.add_argument('--submodel', choices=model_choices, type=str, default="unet_18")
# parser.add_argument('--nbr_frame' , type=int , default=5)
# parser.add_argument('--nbr_width' , type=int , default=1)
parser.add_argument('--joinType', choices=["concat" , "add" , "none"], default="concat")
parser.add_argument('--upmode', choices=["transpose","upsample"], type=str, default="transpose")
parser.add_argument('--n_outputs' , type=int, default=8,
                        help="For Kx FLAVR, use n_outputs k-1")

# Training specifications
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--flow_epochs', type=int, default=30,
                    help='number of epochs to only train refinenet')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--mid_loss_weight', type=float, default=1.,
                    help='the weight of mid loss in trainer')
parser.add_argument('--patience', type=int, default=2,
                    help='learning rate decay without increase in N epochs')
parser.add_argument('--opti', type=str, default='Adam',
                    help='optimizer')

# Log specifications
parser.add_argument('--experiment_dir', type=str, default='../experiment/',
                    help='file name to save')
parser.add_argument('--pretrain_models_dir', type=str, default='../pretrain_models/',
                    help='file name to save')
parser.add_argument('--save', type=str, default='VIDUE',
                    help='file name to save')
parser.add_argument('--save_middle_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', action='store_true',
                    help='resume from the latest if true')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_images', default=True, action='store_true',
                    help='save images')

args = parser.parse_args()
template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
