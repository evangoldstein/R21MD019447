# R21MD019447

This project contains machine learning training scripts for label-based classification with support for both individual label training and K-fold cross-validation with multiple seeds.

## Important Note

**Dataset Privacy**: Due to the sensitive nature of the dataset and privacy concerns, we cannot share the dataset used in this project. To run these scripts, you will need to have your own dataset and modify the dataset paths accordingly in the training scripts.

## Project Structure

### Files in Root Directory

- **`requirements.txt`**: Contains the list of all required Python libraries needed to run the project
- **`setup.sh`**: Bash script to create the environment and install all required libraries
- **`LICENSE`**: Project license file

### Training_for_each_label/ Folder

This folder contains scripts for training models on individual labels:

- **`run.sh`**: Bash script that runs `training.py` for each label. The labels are defined within the script and can be modified according to your dataset's label names
- **`training.py`**: Training code for training models on each individual label of the dataset. You can modify hyperparameters in this file according to your needs and requirements

### KFoldCV+Seeds/ Folder

This folder contains scripts for K-fold cross-validation training on the most important labels:

- **`run.sh`**: Bash script that runs `training_k_cv_seeds.py` for the 13 most important labels. The labels are defined within the script and can be modified according to your dataset's label names
- **`training_k_cv_seeds.py`**: Training code for K-fold cross-validation training on the 13 most important labels of the dataset. You can modify hyperparameters in this file according to your needs and requirements

## How to Run

### 1. Environment Setup

First, you must run the setup script to create the environment and install all required libraries:

```bash
./setup.sh
```

### 2. Training for All Labels

If you want to run training for all labels:

1. Navigate to the `Training_for_each_label` folder:
   ```bash
   cd Training_for_each_label
   ```

2. Modify the `PATH` variable in the `training.py` file to point to your dataset location

3. Run the bash script:
   ```bash
   ./run.sh
   ```

### 3. K-Fold Cross-Validation Training

If you want to run K-fold cross-validation training on the 13 most important labels:

1. Navigate to the `KFoldCV+Seeds` folder:
   ```bash
   cd KFoldCV+Seeds
   ```

2. Modify the `PATH` variable in the `training_k_cv_seeds.py` file to point to your dataset location

3. Run the bash script:
   ```bash
   ./run.sh
   ```

## Customization

- **Dataset Path**: Modify the `PATH` variable in both `training.py` and `training_k_cv_seeds.py` files to point to your dataset location before running
- **Labels**: Modify the label lists in the respective `run.sh` scripts to match your dataset's label names
- **Hyperparameters**: Adjust hyperparameters in `training.py` and `training_k_cv_seeds.py` files according to your requirements