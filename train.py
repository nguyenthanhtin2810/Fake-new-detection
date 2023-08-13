import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Read CSV files
data = pd.read_csv("fake_or_real_news.csv")
data = data.drop(["Unnamed: 0"], axis=1)

# Split the data into features (x) and target labels (y)
target = "label"
x = data.drop(target, axis=1)
y = data[target]
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Define preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "title"),
    ("text", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1), min_df=0.01, max_df=0.95), "text"),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selector", SelectPercentile(chi2, percentile=10)),
    ("regressor", None),
])

parameters = [
    {
        'regressor': [SVC()],
        'regressor__kernel': ['linear', 'rbf'],
        'regressor__C': [0.1, 1, 10]
    },
    {
        'regressor': [RandomForestClassifier(random_state=42)],
        'regressor__n_estimators': [50, 100],
        'regressor__criterion': ["gini", "entropy"]
    }
]
# Find best model and parameters
model = GridSearchCV(pipeline, parameters, scoring="accuracy", cv=5, verbose=2)
# Train the model on the training data
model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)
# Make predictions on the test data
y_pred = model.predict(x_test)
# Classification report
print(classification_report(y_test, y_pred))
# Create a heatmap of the confusion matrix
cm = confusion_matrix(y_test, y_pred)
confusion = pd.DataFrame(cm, index=["Fake", "Real"], columns=["Fake", "Real"])
plt.figure(figsize=(6, 4))
sn.heatmap(confusion, annot=True, cmap='Blues', linecolor='black', linewidths=1, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# plt.show()

plt.savefig("confusion_matrix.jpg")

with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

