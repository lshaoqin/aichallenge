from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from read_yaml import read_yaml
import dataprep
import pickle

def logistic_regression(X, y):
    """
    Train a logistic regression model
    """
    model = LogisticRegression(max_iter = 1000)
    model.fit(X, y)
    return model

def gradient_boosting(X, y, learning_rate = 0.1, n_estimators = 100, min_samples_split = 2, min_samples_leaf = 1, max_depth = 3, max_features = 'auto'):
    """
    Train a gradient boosting model
    """
    model = GradientBoostingClassifier(
        learning_rate = learning_rate, 
        n_estimators = n_estimators, 
        min_samples_split = min_samples_split, 
        min_samples_leaf = min_samples_leaf, 
        max_depth = max_depth, 
        max_features = max_features
    )
    model.fit(X, y)
    return model

def random_forest(X, y, n_estimators = 100, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto'):
    """
    Train a random forest model
    """
    model = RandomForestClassifier(
        n_estimators = n_estimators, 
        max_depth = max_depth, 
        min_samples_split = min_samples_split, 
        min_samples_leaf = min_samples_leaf, 
        max_features = max_features
    )
    model.fit(X, y)
    return model

def decision_tree(X, y, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = None):
    """
    Train a decision tree model
    """
    model = DecisionTreeClassifier(
        max_depth = max_depth, 
        min_samples_split = min_samples_split, 
        min_samples_leaf = min_samples_leaf, 
        max_features = max_features
    )
    model.fit(X, y)
    return model

def linear_regression(X, y):
    """
    Train a linear regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def save_model(model, filename):
    """
    Save the model to a file
    """
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    """
    Load a model from a file
    """
    return pickle.load(open(filename, 'rb'))

def main():
    """
    Main function for shell script
    """
    config = read_yaml("config.yaml")

    if not config['train_cat_model']:
        return
    
    df_name = config['df_name']
    model_type = config['cat_model_type']
    save = config['save_cat_model']
    filename = config['cat_model_filename']

    df = dataprep.read_df(df_name)
    df = dataprep.add_desclength_col(df)
    x_text, x_cat, y = dataprep.split_text_categorical(df)
    x_train, x_test, y_train, y_test = dataprep.split_train_test(x_cat, y)

    if model_type == "LOGISTIC_REGRESSION":
        x_train, x_test = dataprep.one_hot_encode(x_train, x_test)
        model = logistic_regression(x_train, y_train)

    elif model_type == "GRADIENT_BOOSTING":
        x_train, x_test = dataprep.ordinal_encode(x_train, x_test)
        params = config['GRADIENT_BOOSTING']
        learning_rate = params['learning_rate']
        n_estimators = params['n_estimators']
        min_samples_split = params['min_samples_split']
        min_samples_leaf = params['min_samples_leaf']
        max_depth = params['max_depth']
        max_features = params['max_features']
        model = gradient_boosting(x_train, y_train, learning_rate, n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features)

    elif model_type == "RANDOM_FOREST":
        x_train, x_test = dataprep.ordinal_encode(x_train, x_test)
        params = config['RANDOM_FOREST']
        n_estimators = params['n_estimators']
        max_depth = params['max_depth']
        min_samples_split = params['min_samples_split']
        min_samples_leaf = params['min_samples_leaf']
        max_features = params['max_features']
        model = random_forest(x_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
    
    elif model_type == "DECISION_TREE":
        x_train, x_test = dataprep.ordinal_encode(x_train, x_test)
        params = config['DECISION_TREE']
        max_depth = params['max_depth']
        min_samples_split = params['min_samples_split']
        min_samples_leaf = params['min_samples_leaf']
        max_features = params['max_features']
        model = decision_tree(x_train, y_train, max_depth, min_samples_split, min_samples_leaf, max_features)
    
    elif model_type == "LINEAR_REGRESSION":
        x_train, x_test = dataprep.one_hot_encode(x_train, x_test)
        model = linear_regression(x_train, y_train)

    accuracy = evaluate_model(model, x_test, y_test)

    print(f"{model_type.lower()} accuracy: {accuracy}")

    if save:
        save_model(model, filename)
        print(f"Model saved to {filename}")

    return model

if __name__ == "__main__":
    main()