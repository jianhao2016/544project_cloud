import argparse

'''
def str2bool(v):
  if v.lower() in ('yes', 'true', '1', 'y', 't'):
    return True
  elif v.lower() in ('no', 'false', '0', 'n', 'f'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected')
'''

parser = argparse.ArgumentParser()

# General options
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to dataset.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Options: imagenet | cifar10 | svhn | frgc | mnist')
parser.add_argument('--save', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

# Data options
parser.add_argument('--trsize', type=int, default=2000,
                    help='# of train data')
parser.add_argument('--tstsize', type=int, default=1000,
                    help='# of test data')

# Train Options
parser.add_argument('--nEpochs', type=int, default=0,
                    help='# of total epochs to run')
parser.add_argument('--epochNumber', type=int, default=1,
                    help='Manual epoch number (useful on restarts)')
parser.add_argument('--batchSize', type=int, default=20,
                    help='mini-batch size (1 = pure stochastic)')
parser.add_argument('--testOnly', action='store_true', default=False, help='Run on validation set only')
parser.add_argument('--tenCrop', action='store_true', default=False, help='Ten-crop testing')
parser.add_argument('-resume', type=str, default=None,
                    help='Path to directory containing checkpoint')

# optimization option
parser.add_argument('--LR', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weightDecay', type=float,
                    default=1e-4, help='weight decay')

# model option
parser.add_argument('--depth', type=int, default=20, help='ResNet depth: 18 | 34 | 50 | 101 | ...number')
parser.add_argument('--retrain', type=str, default=None, help='Path to model to retrain with')
parser.add_argument('--optimState', type=str, default=None, help='Path to an optimState to reload from')
parser.add_argument('--shareGradInput', action='store_true', default=False, help='Share gradInput tensors to reduce memory usage, better than optnet')
parser.add_argument('--shareWeights', action='store_true', default=False, help='share weight or Not')
parser.add_argument('--optnet', action='store_true', default=False, help='Use optnet to reduce memory usage')
parser.add_argument('--resetClassifier', action='store_true', default=False, help='Reset the fully connected layer for fine-tuning')
parser.add_argument('--nClasses', type=int, default=10,
                    help='Number of classes in the dataset')
parser.add_argument('--stride', type=int, default=1,
                    help='Striding for Convolution, equivalent to pooling')
parser.add_argument('--sparsity', type=float, default=0.9,
                    help='Percentage of sparsity in pre-defined LB filters')
parser.add_argument('--nInputPlane', type=int, default=3,
                    help='number of input channels')
parser.add_argument('--numChannels', type=int, default=128,
                    help='number of intermediate channels')
parser.add_argument('--full', type=int, default=512,
                    help='number of hidden units in FC')
parser.add_argument('--num_of_B', type=int, default=512,
                    help='number of fixed binary weights')
parser.add_argument('--convSize', type=int, default=3,
                    help='LB convolutional filter size')


