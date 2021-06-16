import argparse


def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=100,
						help="number of epochs")
	parser.add_argument("--batch_size", type=int, default=32,
						help="size of each image batch")
	parser.add_argument('--learning_rate', default=0.1, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
	parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
	parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
	parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
	parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
	parser.set_defaults(nesterov=False)
	parser.add_argument("--n_cpu", type=int, default=8,
						help="number of cpu threads to use during batch generation")
	parser.add_argument("--log_interval", type=int, default=10,
						help="interval of display metrics")
	parser.add_argument("--dataset_path", type=str,
						default="Images", help="Dataset directory")
	parser.add_argument("--resume_path", type=str,
						default=None, help="resume training")
	parser.add_argument("--save_interval", type=int,
						default=1, help="saving weights interval")
	parser.add_argument('--lr_patience', default=5, type=int,
						help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
	parser.add_argument("--gpu", type=int,
						default=0, help="gpu id")
	parser.add_argument("--dataset", type=str,
						default='Stanford40', help="Dataset type")
	parser.add_argument("--depth", type=int,
						help='18|34|50|101', default=50)
	parser.add_argument("--num_classes", type=int,
						default=40, help="Number of classes")
	parser.add_argument("--wt_decay", type=float,
						default=5e-4, help="weight decay")
	parser.add_argument("--start_epoch", type=int,
						default=1)
	parser.add_argument("--img_path", type=str)
	opt = parser.parse_args()
	return opt
