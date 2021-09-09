# HAConvGNN: Hierarchical Attention Based Convolutional Graph Neural Network for Code Documentation Generation in Jupyter Notebooks
This repository contains the public release code for our new accepted paper: HAConvGNN: Hierarchical Attention Based Convolutional Graph Neural Network for Code Documentation Generation in Jupyter Notebooks

The related package requirement is in the requirement.txt file.

com.test file is my ground true output file.

You'd better use Cuda to run this project.

Because the dataset file and trained model file is very large, you can download it in the following website.

The reproducibility package has three parts:
1. the code found in this repository
2. the fully processed data (as a pkl file) can be downloaded [HERE](https://ibm.biz/Bdfpk6)
3. the model file can be downloaded [HERE](https://ibm.biz/BdfpkU)(If you want to load this model with CPU, you should add ```map_location=torch.device('cpu')``` when loading the model file)

This code uses Pytorch v1.5.0

## notebookCDG Dataset

Inspired by [Wang et al. 2021](https://dl.acm.org/doi/abs/10.1145/3411763.3451617), we decided to utilize the top-voted and well-documented Kaggle notebooks to construct the notebookCDGdataset 

We collected the top 10% highly-voted notebooks from the top 20 popular competitions on Kaggle (e.g. Titanic). We checked the data policy of each of the 20 competitions, none of them has copyright issues. We also contacted the Kaggle administrators to make sure our data collection complies with the platform’s policy.  

In total, we collected 3,944 notebooks as raw data. After data preprocessing, the final dataset contains 2,476 notebooks out of the 3,944 notebooks from the raw data. It has 28,625 code–documentation pairs. The overall code-to-markdown ratio is 2.2195

![data](https://user-images.githubusercontent.com/3362714/132431560-7f68606f-d0a1-4d9b-b46e-8537fd65c123.png)

## Running the code and models

In my code, I have HAConvGNN model.

To run *prediction* with the trained models from the paper, download the three parts of the reproducibility package and run predict.py. Predict.py takes the path to the model file as a positional argument and will output the prediction file to ./modelout/predictions.

`python3 predict.py {path to model} --gpu 0 --data {path to data download}`

For example:

`python3 predict.py ./modelout/HAConvGNN.h5 --gpu 0 --data ./final_data`

As a reference, running the prediction on one Tesla T4 GPU takes about 1 hour.

To *train* a new model run train.py with the modeltype and gpu options set.

`python3 train.py --gpu 0 --data ./final_data`

## Model

![avatar](/img/model.png)

## Result
| Models                       | ROUGE-1 |       |       | ROUGE-2 |      |      | ROUGE-L |       |       |
|------------------------------|---------|-------|-------|---------|------|------|---------|-------|-------|
|                              | P       | R     | F1    | P       | R    | F1   | P       | R     | F1    |
| Baselines   |
| code2seq                     | 11.45   | 8.46  | 8.23  | 1.67    | 1.11 | 1.11 | 13.13   | 10.28 | 10.24 |
| graph2seq                    | 13.21   | 9.87  | 9.51  | 2.86    | 1.99 | 2.03 | 14.46   | 11.4  | 11.18 |
| Our Model & Ablation Study   |         |       |       |         |      |      |         |       |       |
| HAConvGNN(Our Model)         | 22.87   | 16.92 | 16.58 | 6.72    | 4.86 | 4.97 | 24.03   | 18.6  | 18.54 |
| HAConvGNN   <br> withlow-level attention  <br> without high-level attention <br> with uniform attention      | 20.66   | 15.65 | 14.91 | 4.74    | 3.92 | 3.8  | 21.84   | 17.27 | 16.81 |
| HAConvGNN <br> with low-level attention  <br>  without high-level attention <br> without uniform attention  | 19.57   | 14.59 | 14.23 | 4.87    | 3.56 | 3.63 | 20.83   | 16.24 | 16.12 |
| HAConvGNN <br> without low-level attention <br> without high-level attention <br> with uniform attention      | 11.39   | 7.73  | 7.82  | 1.58    | 1.06 | 1.08 | 13.13   | 9.47  | 9.82  |

## Bibliographic Citations

Our work is published at [EMNLP'21 Finding](https://arxiv.org/abs/2104.01002). You can cite: 

```
@misc{liu2021haconvgnn,
      title={HAConvGNN: Hierarchical Attention Based Convolutional Graph Neural Network for Code Documentation Generation in Jupyter Notebooks}, 
      author={Xuye Liu and Dakuo Wang and April Wang and Yufang Hou and Lingfei Wu},
      year={2021},
      eprint={2104.01002},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```
