import flair
import nltk
import pandas as pd
import os

from statistics import mean
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download([
    'names',
    'stopwords',
    'wordnet',
    "vader_lexicon",
])

flair_classifier = flair.models.TextClassifier.load('en-sentiment')
lmt = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()


def top_n_words(texts, n=100, unwanted=[]):
    """Generates top positive/negative n words in a given set of texts.
    """
    positive_words = list()
    negative_words = list()
     # By hashing flair scores in flair_dict, we speed up the algorithm,
     # leveraging constant access time for repeating words.
    flair_dict = {}

    # text -> sentence -> word -> lemmatize -> take flair score -> collect to positive/negative
    for text in texts:
        for sentence in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sentence):
                word_lower = word.lower()
                lemmatized = lmt.lemmatize(word_lower)
                if word.isalpha() and lemmatized not in unwanted:
                    if lemmatized in flair_dict:
                        sentiment = flair_dict[lemmatized]
                    else:
                        flair_sentence = flair.data.Sentence(word)
                        flair_classifier.predict(flair_sentence)
                        sentiment = flair_sentence.labels[0].value
                        flair_dict[lemmatized] = sentiment

                    if sentiment == 'POSITIVE':
                        positive_words.append(lemmatized)
                    else:
                        negative_words.append(lemmatized)

    positive_fd = nltk.FreqDist(positive_words)
    negative_fd = nltk.FreqDist(negative_words)

    common_set = set(positive_fd).intersection(negative_fd)
    for word in common_set:
        del positive_fd[word]
        del negative_fd[word]

    top_n_positive = {word for word, count in positive_fd.most_common(n)}
    top_n_negative = {word for word, count in negative_fd.most_common(n)}

    return top_n_positive, top_n_negative


def extract_features(text, top_n_positive=[], top_n_negative=[]):
    features = dict()
    compound_scores = list()
    positive_scores = list()
    negative_scores = list()
    flair_scores = list()
    positive_wordcount = 0
    negative_wordcount = 0

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if lmt.lemmatize(word.lower()) in top_n_positive:
                positive_wordcount += 1
            elif lmt.lemmatize(word.lower()) in top_n_negative:
                negative_wordcount += 1

        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])
        negative_scores.append(sia.polarity_scores(sentence)["neg"])

        lemmatized = [lmt.lemmatize(word) for word in word_tokenize(sentence)]
        flair_sentence = flair.data.Sentence(lemmatized)
        flair_classifier.predict(flair_sentence)
        if flair_sentence.labels[0].value == 'POSITIVE':
            flair_score = flair_sentence.labels[0].score
        else:
            flair_score = -flair_sentence.labels[0].score

        flair_scores.append(flair_score)

    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    # features["mean_negative"] = mean(negative_scores)
    # features["positive_wordcount"] = positive_wordcount
    # features["negative_wordcount"] = negative_wordcount
    features["mean_flair"] = mean(flair_scores) + 1

    return features


def preprocess_df(df: pd.DataFrame, train=False):
    if not os.path.exists("./top_n_positive.csv"):
        # TODO: Extend unwanted with countries and nationalities.
        unwanted = nltk.corpus.stopwords.words("english")
        unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

        top_n_positive, top_n_negative = top_n_words(df['text'], 100, unwanted)

        # Save top_n_words outputs to csv files.
        positive_df = pd.DataFrame(top_n_positive)
        negative_df = pd.DataFrame(top_n_negative)
        positive_df.to_csv("./top_n_positive.csv")
        negative_df.to_csv("./top_n_negative.csv")

    else:
        positive_df = pd.read_csv("./top_n_positive.csv")
        negative_df = pd.read_csv("./top_n_negative.csv")
        top_n_positive, top_n_negative = positive_df['0'].tolist(), negative_df['0'].tolist()

    if train:
        features = [
            [v for k,v in extract_features(X, top_n_positive, top_n_negative).items()] + [y] for X, y in zip(df['text'], df['label'])
        ]
    else: # predict
        X = df['text'][0]
        features = [v for k,v in extract_features(X, top_n_positive, top_n_negative).items()]
    return features


if __name__ == "__main__":
    # Testing an example - feature extraction
    pos_text = "With the new production plant the company would increase its capacity\
        to meet the expected increase in demand and would improve the use of raw \
        materials and therefore increase the production profitability."

    neg_text = "The international electronic industry company Elcoteq has laid off\
        tens of employees from its Tallinn facility ; contrary to earlier layoffs the\
        company contracted the ranks of its office workers , the daily Postimees reported."

    features = extract_features(pos_text)
    print(f"Text: {pos_text}")
    for key in features:
        print(f"{key}: {features[key]}")