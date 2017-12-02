# small tasks
1) Wu, write the argument parser. Replace "numWeights" with "number_of_b".
Add "shared_weights" to parser. Refer to the cifar10_main.py how to pass boolean into parser. Get familiar with Google Cloud. How to send code to cloud and run on Google Cloud. Try some tutorial code on it.

2) Yu, tensor size in conv layer/number of nodes in fully connect layer. In each layer, kernel/filter size in each layer, strides, padding or not(what padding). Smallest modules are conv2d, fully-connected, batch-normalization, relu/sigmoid/activation function, softmax, etc. 
Also, what kind of loss/regularization, what optimizer, learning rate, training epoch, batch size, may refer to their paper setup.
Better be shown in figures.

3) Peng, will update the resnet model with LBC.
# Things to do.

1) Make the ResNets model running. KEEP/CALCULATE the model size with respect to numbers of parameters. [similar to what they did in their paper]
2) We also need to plot the figure of error/accuracy versus number of iteration. And compare with the vanilla model in order to show that LBCNN is faster.
3) If possible, sample the output from Conv layers and compare with the output of LBC layers. [Not sure feasible]
4) Do the exact same thing to other networks with similar framework, i.e. multiple conv layers.

# ResNet in TensorFlow

Deep residual networks, or ResNets for short, provided the breakthrough idea of identity mappings in order to enable training of very deep convolutional neural networks. This folder contains an implementation of ResNet for the ImageNet dataset written in TensorFlow.

See the following papers for more background:

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

Please proceed according to which dataset you would like to train/evaluate on:


## CIFAR-10

### Setup

You simply need to have the latest version of TensorFlow installed.

First download and extract the CIFAR-10 data from Alex's website, specifying the location with the `--data_dir` flag. Run the following:

```
python cifar10_download_and_extract.py
```

Then to train the model, run the following:

```
python cifar10_main.py
```

Use `--data_dir` to specify the location of the CIFAR-10 data used in the previous step. There are more flag options as described in `cifar10_main.py`.


## ImageNet

### Setup
To begin, you will need to download the ImageNet dataset and convert it to TFRecord format. Follow along with the [Inception guide](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) in order to prepare the dataset.

Once your dataset is ready, you can begin training the model as follows:

```
python imagenet_main.py --data_dir=/path/to/imagenet
```

The model will begin training and will automatically evaluate itself on the validation data roughly once per epoch.

Note that there are a number of other options you can specify, including `--model_dir` to choose where to store the model and `--resnet_size` to choose the model size (options include ResNet-18 through ResNet-200). See [`imagenet_main.py`](imagenet_main.py) for the full list of options.
