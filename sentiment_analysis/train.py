import pandas as pd
import os
import pickle

from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def train(model, features):
    classifier = model
    X = [vec[:-1] for vec in features]
    y = [vec[-1] for vec in features]

    # Splitting train- test by 75% - 25%
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    return accuracy


def predict(text, debug=False):
    filename = "trained_mlp.sav"
    if not os.path.exists(filename):
        # Get data
        if os.path.exists("./features.csv"):
            if debug:
                print("Getting features.csv...")

            df = pd.read_csv("./features.csv")
            train_features = df.values.tolist()
        else:
            if debug:
                print("features.csv does not exist.")
                print("Pre-processing financial_news.csv...")

            df = pd.read_csv("./financial_news.csv")
            # feature extraction
            train_features = preprocess_df(df, train=True)

            # save features
            if debug:
                print("Saving features.csv...")

            df_features = pd.DataFrame(train_features)
            df_features.to_csv("./features.csv")

        classifiers = {
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(),
            "MLPClassifier": MLPClassifier(max_iter=1000),
            "AdaBoostClassifier": AdaBoostClassifier(),
        }

        print("Training...")
        classifier = classifiers["MLPClassifier"]
        accuracy = train(classifier, train_features)
        print(f"Accuracy: {accuracy}")

        pickle.dump(classifiers["MLPClassifier"], open(filename, 'wb'))

    else:
        classifier = pickle.load(open(filename, 'rb'))

    df = pd.DataFrame([text], columns=['text'])
    X = [preprocess_df(df, train=False)]
    prediction = classifier.predict(X)

    if prediction[0] == 0:
        return 'NEGATIVE'
    elif prediction[0] == 1:
        return 'POSITIVE'
    else:
        return 'NEUTRAL'

if __name__ == "__main__":
    # Testing an example - feature extraction
    pos_text = "With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability."
    neg_text = "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported."

    for text in (neg_text, pos_text):
        print(f"Text: {text}")
        print(predict(text, debug=True))