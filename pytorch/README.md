## Benchmarks for Computer Vision in Pytorch

###Cifar10###

####ConvNet####

```
python cv/cifar10/run/cifar10_cnn1.py --batch_size_test 100 --output_folder $OUTPUT_FOLDER$ --model_name $MODEL_NAME$
```

####ResNet###

```
python cv/cifar10/run/cifar10_resnet3b.py --batch_size_test 100 --depth $DEPTH$ --output_folder $OUTPUT_FOLDER$ --model_name $MODEL_NAME$
```

| Model    | Test Accuracy |
|----------|---------------|
| ResNet20 |               |
| ResNet44 | 92.8%         |