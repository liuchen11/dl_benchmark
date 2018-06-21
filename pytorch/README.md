## Benchmarks for Computer Vision in Pytorch

### MNIST

#### Multi-Layer Perceptron

```
python cv/mnist/run/mnist_mlp.py --batch_size_test 100 --output_folder $OUTPUT_FOLDER$ --model_name $MODEL_NAME$
```

A 3-layer fully-connected model with 300, 100 as the dimension of hidden layer. RELU is used as the activation function. Test accuracy is around 98.3%

#### ConvNet

```
python cv/mnist/run/mnist_cnn1.py --batch_size_test 100 --output_folder $OUTPUT_FOLDER$ --model_name $MODEL_NAME$
```

The same architecture as the [convnet for mnist in tensorflow tutorial](https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py). Test accuracy is around 99.4%.

### Cifar10

#### ConvNet

```
python cv/cifar10/run/cifar10_cnn1.py --batch_size_test 100 --output_folder $OUTPUT_FOLDER$ --model_name $MODEL_NAME$
```

The same architecture as the [convnet for cifar10 in tensorflow tutorial](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py). Test accuracy is around 85%.

#### ResNet

```
python cv/cifar10/run/cifar10_resnet3b.py --batch_size_test 100 --depth $DEPTH$ --output_folder $OUTPUT_FOLDER$ --model_name $MODEL_NAME$
```

Deep residual neural network. The number in the table below indicates the depth.

| Model    | Test Accuracy |
|----------|---------------|
| ResNet20 |    92.1%      |
| ResNet44 |    92.7%      |
| ResNet68 |    93.5%      |


### ImageNet


