import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

data["label"] = data["label"].map({"ham": 0, "spam": 1})

data["message"] = data["message"].apply(
    lambda x: x.lower().translate(str.maketrans("", "", string.punctuation))
)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["message"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

def check_message(msg):
    msg = msg.lower().translate(str.maketrans("", "", string.punctuation))
    vec = vectorizer.transform([msg])
    return "Spam" if model.predict(vec)[0] == 1 else "Not Spam"

print(check_message("You have won a free ticket. Claim now!"))
