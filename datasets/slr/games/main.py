import bibtexparser as bib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np

#filename = 'round1-todos.bib'
filename = 'round1-google.bib'
with open(filename) as bibFile:
    bibDb = bib.load(bibFile)

abstracts = []
y = []
for e in bibDb.entries:
    abstracts.append(e['abstract'])
    if e['inserir'] == 'true':
        y.append(1)
    else:
        y.append(0)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(abstracts)

classifier = DecisionTreeClassifier()
classifier.fit(X, y)

print('Decision Tree')
print(export_text(classifier))
print(accuracy_score(y, classifier.predict(X)))

# Linear SVC
classifier3 = LinearSVC()
classifier3.fit(X, y)
print('Linear SVC')
print(classifier3.decision_function(X))
print(accuracy_score(y, classifier3.predict(X)))

# SVC
classifier2 = SVC()
classifier2.fit(X, y)
print('SVC')
print(classifier2.decision_function(X))
print(accuracy_score(y, classifier2.predict(X)))

# Random Forest
classifier4 = RandomForestClassifier()
classifier4.fit(X, y)
print('Random Forest')
print(classifier4.decision_path(X))
print(accuracy_score(y, classifier4.predict(X)))

# Logistic Regression
classifier5 = LogisticRegression()
classifier5.fit(X, y)
print('Logistic Regression')
print(classifier5.decision_function(X))
print(accuracy_score(y, classifier5.predict(X)))

