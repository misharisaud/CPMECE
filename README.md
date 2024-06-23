# Description
This repository contains code aiming to surpass the results presented in the paper "Turnover number predictions for kinetically uncharacterized enzymes using machine and deep learning." The objective is to predict the Kcat value (enzyme efficiency) based on two main features: Enzyme Sequence and chemical reaction.

## Input Features
### Enzyme Sequence
- Enzyme Sequence is a categorical feature (e.g., `MNTVRSEKDSMGAIDVPADKLWGAQTQRSLEHFRISTEKMPTSLIH...`) converted into a numeric vector using [ESM-1b](https://github.com/facebookresearch/esm) (A transformer-based Deep learning Model). Two features for numeric enzyme representation are utilized: `esm1b_ts` (task specific) and `esm1b`.
  - `esm1b_ts`: Numeric representation obtained by fine-tuning the esm1b model on an enzyme sequence dataset.
  - `esm1b`: Numeric representation obtained by pre-training the esm1b model.

### Chemical Reaction
- Chemical reaction consists of substrate and product, converted into binary numeric representation (Fingerprint representation). Different fingerprint representations include:
  - DRFP
  - Difference FP
  - Structural FP

# Performance
To enhance performance and surpass the previous technique for predicting the Kcat value, we implemented the following:
- Utilized a Convolutional Neural Network (CNN) as a deep learning model.
- Improved the performance of the previous model, which includes XGBOOST.

## Previous:
The best-performing model was **XGBOOST**; here are the performance metrics for the XGBOOST Model found in the last paper. The table combines metrics for the single best model and ensemble results, making it easy to compare performance.

| Metric                | Best Single Model | Ensemble Results |
|-----------------------|-------------------|-------------------|
| R2 score              | 0.40              | 0.44              |
| MSE                   | 0.86              | 0.81              |
| Pearson coefficient   | 0.64              | 0.67              |

## Current:
We have not only increased performance using CNN but also improved the performance of the previous XGBOOST model.

**XGBOOST Model Performance**

| Metric               | Single Best Model | Ensemble Result   |
|----------------------|-------------------|-------------------|
| R2 score             | 0.43              | 0.47              |
| MSE                  | 0.65              | 0.70              |
| Pearson coefficient  | 0.76              | 0.69              |

**CNN Model Performance:**

| Metric               | Single Best Model | Ensemble Result   |
|----------------------|-------------------|-------------------|
| R2 score             | 0.48              | 0.54              |
| MSE                  | 0.69              | 0.61              |
| Pearson coefficient  | 0.69              | 0.74              |

# Directories:
- **improved_code** directory contains the following subdirectories of code:
    - **Preprocessing**: This directory contains code for preprocessing, including:
        - **Data Preprocessing.ipynb**: A notebook for preprocessing to set the final split dataset (train and test). It includes changes like normalizing the dataset before the train and test split, updating the code for Enzyme Sequence to numeric feature representation mapping, and identifying Sequence ID differences in df1 and df2. It also adds code for correctly mapping chemical reactions to numeric fingerprints.
    - **model_training**: This directory contains notebooks and a script:
        - **Training CNN models with enzyme and reaction information**: For training and hyperparameter tuning of the CNN model. 
          - **Hyperparameter tuning code**:
            - It is responsible for finding the optimized hyperparameter that gives the best result. 
            - This will only save the best-trained model and save its hyperparameter. 
            - For hyperparameter tuning, we have used custom hyperparameter tuning code to monitor which iteration score, save the best-performing results to avoid any accidental code stop. 
            - If the code stops, we can start the same iteration number where it stopped, (but keep in mind to save the previous best model; the current pipeline just considers the current best models to avoid space issues.) 
          - **Training code**: It is is responsible for training the CNN model using trained hyperparameters.
        - **utils.py**: A script containing helper functions for hyperparameter tuning and training of CNN models.
        - **Training xgboost models with enzyme and reaction information**: For training and hyperparameter tuning of the XGBOOST model.
    - **Inference**: This directory contains the following notebook:
        - **ensemble.ipynb**: A notebook for ensembling the output of the best-performing trained models of CNN and XGBOOST.
- **data**: This directory contains the dataset required to train and test the model.
- **models**: This directory contains subdirectories for trained models:
  - **best_model**: Contains the best-performing models of CNN and XGBOOST.
  - **hyperparameter_tune_model**: Contains the best-performing model when running the hyperparameter optimization code of CNN.
  - **train_model**: Contains the saved model when running the training code of CNN.
- **hyperparameters**: This directory contains a text file of hyperparameters with an R2 score. This hyperparameters text file is saved during the hyperparameter tuning code.
  ```
  Best Score: 0.45
  filters_1: 2
  filters_2: 12
  filters_3: 14
  kernel_size_1: 17
  kernel_size_2: 9
  kernel_size_3: 7
  dense_units_1: 512
  dense_units_2: 8
  dropout_rate: 0.4
  optimizer: rmsprop
  batch_size: 24
  ```

## Usage:
**Download Dataset**
- To download the full dataset which also contain the enzyme to numeric representation dataset (for finetuning the ESM1b Model) follow these instructions.
  - Before running the Jupyter notebooks, download and unzip a [data folder from figshare](https://figshare.com/articles/dataset/cpmece-data_rar/26085421). Afterwards, this repository should have the following structure:
      ├── .ipynb_checkpoints                   
      ├── __MACOSX
	  ├── BiGG_data
	  ├── DLKcat
	  ├── enzyme_data
	  ├── kcat_data
	  ├── KM_data
	  ├── metabolite_data
	  ├── reaction_data
	  ├── Source Data
      └── training_results
- Otherwise use given dataset which only contain dataset to train and test the turnover prediction.

**Install Requirements**
To set up the project locally, set up the virtual env using the following instructions.
- If you do not have Miniconda, please install it from [here](https://docs.conda.io/projects/miniconda/en/latest/) according to your OS.
- Create a conda environment and install requirements using the following commands.
  ```
  conda create -n turnover-pred python=3.10.12 -y
  pip install -r requirements_final.txt
  ```