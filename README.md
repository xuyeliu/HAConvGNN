# Auto Code Report Generation--IBM
This repository contains the public release code for our new accepted paper: HAConvGNN: Hierarchical Attention Based Convolutional Graph NeuralNetwork for Code Documentation Generation in Jupyter Notebooks

The related package requirement is in the requirement.txt file.

com.test file is my ground true output file.

Because the dataset file and trained model file is very large, you can download it in the following website.

The reproducibility package has three parts:
1. the code found in this repository
2. the fully processed data (as a pkl file) can be downloaded [HERE](https://icpc2020.s3.us-east-2.amazonaws.com/dataset.pkl)

This code uses Pytorch v1.5.0 and Pytorch_geometric framework

Pytorch_geometric framework can be downloaded [HERE](https://github.com/khuangaf/PyTorch-Geometric-YooChoose)

## Running the code and models

In my code, I have HAConvGNN model

To run the trained models from the paper download the three parts of the reproducibility package and run predict.py. Predict.py takes the path to the model file as a positional argument and will output the prediction file to ./modelout/predictions.

`python3 predict.py {path to model} --gpu 0 --data {path to data download}`

For example:

`python3 predict.py ./modelout/HAConvGNN.h5 --gpu 0 --data ./final_data`

To train a new model run train.py with the modeltype and gpu options set.

`python3 train.py --gpu 0 --data ./final_data`


