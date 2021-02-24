# A PaddlePaddle implementation for PyramidNets (Deep Pyramidal Residual Networks)

This repository contains a [PaddlePaddle](https://paddlepaddle.org.cn/) implementation for the paper: [Deep Pyramidal Residual Networks](https://arxiv.org/pdf/1610.02915.pdf) (CVPR 2017, Dongyoon Han*, Jiwhan Kim*, and Junmo Kim, (equally contributed by the authors*)). The code in this repository is based on the example provided in [PyTorch implementation](https://github.com/dyhan0920/PyramidNet-PyTorch).

Three other implementations with [LuaTorch](http://torch.ch/) and [Caffe](http://caffe.berkeleyvision.org/) are provided:
1. [A LuaTorch implementation](https://github.com/jhkim89/PyramidNet) for PyramidNets,
2. [A Caffe implementation](https://github.com/jhkim89/PyramidNet-caffe) for PyramidNets.
3. [A PyTorch implementation](https://github.com/dyhan0920/PyramidNet-PyTorch) for PyramidNets.

## Usage examples
To train additive PyramidNet-227 (alpha=200 with bottleneck) on CIFAR-10 dataset with a single-GPU:
```
python train_cifar.py --net 'pyramidnet_bottleneck' --dataset 'cifar10' --alpha 200 --depth 272 --num_classes 10 --epochs 300 --lr 0.1 --batch_size 128
```
To train additive PyramidNet-227 (alpha=200 with bottleneck) on CIFAR-100 dataset with a single-GPU:
```
python train_cifar.py --net 'pyramidnet_bottleneck' --dataset 'cifar100' --alpha 200 --depth 272 --num_classes 10 --epochs 300 --lr 0.1 --batch_size 128
```

### Notes
1. This implementation contains the training (+test) code for add-PyramidNet architecture on ImageNet-1k dataset, CIFAR-10 and CIFAR-100 datasets.
2. The traditional data augmentation for ImageNet and CIFAR datasets are used.
3. The example code for ResNet is also included.  

### Experimental Results
1. Illustration of train/eval for cifar10 is shown by ![image](cifar10.png)

2. Illustration of train/eval for cifar100 is shown by ![image](cifar100.png)

3. Top1 eval acc for cifar10 is 0.9542 and top1 eval acc for cifar100 is 0.7997.


## Citation
Please cite the paper if PyramidNets are used: 
```
@article{DPRN,
  title={Deep Pyramidal Residual Networks},
  author={Han, Dongyoon and Kim, Jiwhan and Kim, Junmo},
  journal={IEEE CVPR},
  year={2017}
}
```
If this implementation is useful, please cite or acknowledge this repository on your work.