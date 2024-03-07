# aichallenge
Submission for the National AI Student Challenge Track 1: AI Professionals Association (AIP)
---
This project works on the [PetFinder.my Adoption Prediction Dataset](https://www.kaggle.com/c/petfinder-adoption-prediction/data) and consists of the following parts:
- Exploratory Data Analysis (EDA) of the dataset at eda.ipynb
- Code to train a classification model using the categorical columns in the dataset (src/cat_models.py)
- Code to train a neural network using the description column in the dataset (src/text_models.py)
- Methods for dataset preprocessing (src/dataprep.py), yaml processing(src/read_yaml.py) and model predictions (src/predictions.py)
- A pipeline(run.sh) and configuration file(config.yaml) to run the code
- A pretrained gradient boosting model (saved_models/cat_model.pkl) and a pretrained CNN model (saved_models/text_model.keras) for the dataset

eda.ipynb was primarily run on Windows Python 3.9.9 and the pipeline was run on Ubuntu 22.04.3 LTS on Python 3.10.6. Due to a lack of foresight and experience, the requirements.txt file was generated for the pipeline, and some parts of the EDA may not run with the same package versions.

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

## Choice/evaluation of models
The gradient boosting model was chosen for its accuracy (0.416) against the other models. Gradient boosting tends to perform well on tabular data which fits our dataset, and it is a strong choice which has won many competitions. I think further tuning of the hyperparameters could even further increase the accuracy of the model.

The CNN model was chosen for both its accuracy (0.374) and speed over the LSTM model. I suspect that this may be because of insufficient tuning of the LSTM model however, since LSTM models are known to perform well on text data. Nevertheless, the CNN model as of now trains at a much faster rate and achieves better accuracy too. This makes sense because the convolution layer can capture features in the text data which means more data is extracted for the later layers to work on.

It was a pity I couldn't successfully combine the predictions of the two models, as they yielded worse results when combined than when used individually. However, I think there is some merit to keeping the models separate especially as the CNN model can serve as a benchmark highlighting that a certain description could potentially be improved. The gradient boosting model can then be used for recommendation, including to identify which pets may need to be displayed to users more to get adopted.