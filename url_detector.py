
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("urls.csv")

print("Dataset Loaded")
print(data.head())


# Input features
X = data[
    [
        "length",
        "has_at",
        "dots",
        "http_only",
        "keyword"
    ]
]


# Output label
# 0 = Safe
# 1 = Suspicious

y = data["label"]


# Split dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create ML model

model = LogisticRegression()


# Train model

model.fit(
    X_train,
    y_train
)


# Test model

y_prediction = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_prediction
)


print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


# User input

print("\nEnter URL Features")

length = int(input("Enter URL length: "))

has_at = int(
    input("Contains @ symbol? (1-Yes, 0-No): ")
)

dots = int(
    input("Number of dots in URL: ")
)

http_only = int(
    input("Uses HTTP instead of HTTPS? (1-Yes, 0-No): ")
)

keyword = int(
    input("Contains suspicious keywords? (1-Yes, 0-No): ")
)


# Create dataframe for new URL

new_url = pd.DataFrame(
    [[
        length,
        has_at,
        dots,
        http_only,
        keyword
    ]],
    columns=[
        "length",
        "has_at",
        "dots",
        "http_only",
        "keyword"
    ]
)


# Prediction

result = model.predict(new_url)


if result[0] == 1:
    print("\nResult: Suspicious URL ⚠️")
else:
    print("\nResult: Safe URL ✅")