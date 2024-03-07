# aichallenge
Submission for the National AI Student Challenge Track 1: AI Professionals Association (AIP)
---
This project works on the [PetFinder.my Adoption Prediction Dataset](https://www.kaggle.com/c/petfinder-adoption-prediction/data) and consists of the following parts:
- Exploratory Data Analysis (EDA) of the dataset at eda.ipynb
- Code to train a classification model using the categorical columns in the dataset
- Code to train a neural network using the description column in the dataset
- A pipeline and configuration file to run the code

## Quickstart
1. Create a virtual environment
```bash
python3 -m venv venv
```

2. Activate the virtual environment

Windows:
```bash
venv\Scripts\activate
```

Linux:
```bash
source venv/bin/activate
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

4. Insert the dataset as `pets_preapred.csv` to the root folder. The other files are not required.

5. The code in the `eda.ipynb` file can now be run!

6. To run the code in the pipeline, run the following command on Linux/WSL:
```bash
bash ./run.sh
```

## Pipeline configuration
Configuration options can be found and edited in the config.yaml file. If you are encountering errors or would like to reset the configuration, 
copy the contents of `config.yaml.default` into `config.yaml`. Do not edit `config.yaml.default` directly.

### Configuration options
- `df_name`: The path to the data file on which to train the models

- `train_cat_model`: Whether to train a new categorical model - set this to false if you have already trained a model and would like to use it
- `cat_model_type`: Type of model to train. Options are `LOGISTIC_REGRESSION`, `GRADIENT BOOSTING`, `RANDOM_FOREST`, `DECISION_TREE`, `LINEAR_REGRESSION`

- There are hyperparameters available for `GRADIENT_BOOSTING`, `RANDOM_FOREST`, and `DECISION_TREE`

- `save_cat_model`: Whether to save the trained categorical model
- `cat_model_filename`: The path to save the trained categorical model to

- `train_text_model`: Whether to train a new neural network model - set this to false if you have already trained a model and would like to use it
- `text_model_type`: Type of model to train. Options are `LSTM` and `CNN`
- `correct_spellings`: Whether to correct spellings in the dataset before training the model. Note that this will take roughly 3-5 minutes to run

- There are hyperparameters available for `LSTM` and `CNN`

- `save_text_model`: Whether to save the trained neural network
- `text_model_filename`: The path to save the trained neural network to

- `predictions`: Whether to predict the adoption speed of the pets in the dataset using the trained models
- `predictions_filename`: The path to the dataset to make predictions on
- `predictions_output_filename`: The path to save the predictions to. The script will output the original dataset with the predictions appended as new columns