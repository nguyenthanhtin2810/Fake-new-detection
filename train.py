import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.metrics import classification_report, confusion_matrix
import pickle


# Read CSV files
df1 = pd.read_csv("data/Fake.csv", dtype=str)
df2 = pd.read_csv("data/True.csv", dtype=str, encoding='latin1')
# Add a label
df1["label"] = ["Fake" for _ in range(df1.shape[0])]
df2["label"] = ["True" for _ in range(df2.shape[0])]
# Concatenate dataframes1
data = pd.concat([df1, df2])
data = data.drop(["date", "subject"], axis=1)
# Split the data into features (x) and target labels (y)
target = "label"
x = data.drop(target, axis=1)
y = data[target]
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
# Define preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "title"),
    ("text", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1), min_df=0.01, max_df=0.95), "text"),
    # ("subject", OneHotEncoder(handle_unknown="ignore"), ["subject"])
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selector", SelectPercentile(chi2, percentile=10)),
    ("regressor", RandomForestClassifier(random_state=42)),
])
# Train the model on the training data
model.fit(x_train, y_train)
# Make predictions on the test data
y_pred = model.predict(x_test)
# Classification report
print(classification_report(y_test, y_pred))
# Create a heatmap of the confusion matrix
cm = confusion_matrix(y_test, y_pred)
confusion = pd.DataFrame(cm, index=["Fake", "True"], columns=["Fake", "True"])
plt.figure(figsize=(6, 4))
sn.heatmap(confusion, annot=True, cmap='Blues', linecolor='black', linewidths=1, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# plt.show()

plt.savefig("confusion_matrix.jpg")

with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)



