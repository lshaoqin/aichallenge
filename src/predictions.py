from read_yaml import read_yaml
import os
import cat_models
import text_models
import dataprep
import numpy as np

def predict():
    """
    Make predictions using saved models
    """
    config = read_yaml("config.yaml")

    if not config['predictions']:
        return
    
    predictions_filename = config['predictions_filename']
    predictions_output_filename = config['predictions_output_filename']

    df = dataprep.read_df(predictions_filename)

    if os.path.exists(config['cat_model_filename']):
        cat_model = cat_models.load_model(config['cat_model_filename'])
        x_text, x_cat, y = dataprep.split_text_categorical(df, has_y=False)
        
        if config['cat_model_type'] == "LOGISTIC_REGRESSION" or config['cat_model_type'] == "LINEAR_REGRESSION":
            x_cat = dataprep.one_hot_encode_one(x_cat)
        else:
            x_cat = dataprep.ordinal_encode_one(x_cat)

        y_pred = cat_model.predict(x_cat)
        df['cat_pred'] = y_pred

    if os.path.exists(config['text_model_filename']):
        text_model = text_models.load_model(config['text_model_filename'])
        x_text, x_cat, y = dataprep.split_text_categorical(df, has_y=False)
        x_text = dataprep.preprocess_text(x_text)

        if config['correct_spellings']:
            print("Correcting spellings - this may take a while")
            x_text = dataprep.correct_spellings(x_text)

        y_pred = text_model.predict(x_text)
        y_pred = np.argmax(y_pred, axis=1)
        df['text_pred'] = y_pred

    df.to_csv(predictions_output_filename, index=False)
    print(f"Predictions saved to {predictions_output_filename}")

if __name__ == "__main__":
    predict()


