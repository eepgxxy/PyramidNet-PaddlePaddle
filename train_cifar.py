from config import config_parameters
import paddle.vision.transforms as T
from PyramidNet import PyramidNet
import paddle
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyramidNet Training')

parser.add_argument('--net', default='no',  type=str,
                    help='the arch to use')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--alpha', default=300, type=int,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--num_classes', default=config_parameters['class_dim'], type=int,
                    help='number of classes for classification')
parser.add_argument('--bottleneck', default=False,  type=bool,
                    help='whether to use bottleneck')
parser.add_argument('--weights', default='no',  type=str,
                    help='the path for pretrained model')
parser.add_argument('--pretrained', default=False,  type=bool,
                    help='whether to load pretrained weights')
parser.add_argument('--batch_size', default=config_parameters['batch_size'],  type=int,
                    help='batch_size')
parser.add_argument('--lr', default=config_parameters['lr'], type=float)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type = int, default = config_parameters['epochs'])
parser.add_argument('--warmup', type = int, default = 10)
args = parser.parse_args()

train_transforms = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])

eval_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
])

if args.dataset == 'cifar10':
    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=train_transforms)
    eval_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=eval_transforms)
elif args.dataset == 'cifar100':
    train_dataset = paddle.vision.datasets.Cifar100(mode='train', transform=train_transforms)
    eval_dataset = paddle.vision.datasets.Cifar100(mode='test', transform=eval_transforms)

train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CUDAPlace(0), batch_size=args.batch_size, shuffle=True)

eval_loader = paddle.io.DataLoader(eval_dataset, places=paddle.CUDAPlace(0), batch_size=args.batch_size)

if args.net == 'pyramidnet':
    model = PyramidNet(args.dataset, args.depth, args.alpha, args.num_classes, args.bottleneck)
if args.net == 'pyramidnet_bottleneck':
    model = PyramidNet(args.dataset, args.depth, args.alpha, args.num_classes, True)
elif args.net == 'resnet':
    model = paddle.vision.models.resnet50(num_classes=args.num_classes)

if args.pretrained:
    weights = paddle.load(args.weights)
    model.set_state_dict(weights)
    print('loading pretrained models')

class SaveBestModel(paddle.callbacks.Callback):
    def __init__(self, target=0.5, path='./best_model', verbose=0):
        self.target = target
        self.epoch = None
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def on_eval_end(self, logs=None):
        if logs.get('acc') > self.target:
            self.target = logs.get('acc')
            self.model.save(self.path)
            print('best acc is {} at epoch {}'.format(self.target, self.epoch))

callback_visualdl = paddle.callbacks.VisualDL(log_dir=f'{args.net}_{args.dataset}')
callback_savebestmodel = SaveBestModel(target=0.5, path=f'{args.net}_{args.dataset}')
callbacks = [callback_visualdl, callback_savebestmodel]

base_lr = args.lr
wamup_steps = args.warmup
epochs = args.epochs

def make_optimizer(parameters=None):
    momentum = 0.9

    learning_rate= paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=epochs, verbose=False)

    learning_rate = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=wamup_steps,
        start_lr=base_lr / 5.,
        end_lr=base_lr,
        verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=args.weight_decay,
        parameters=parameters)
    return optimizer

optimizer = make_optimizer(model.parameters())

model = paddle.Model(model)

model.prepare(optimizer,
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy()) 

model.fit(train_loader,
          eval_loader,
          epochs=epochs,
          callbacks=callbacks,
          verbose=1)

        