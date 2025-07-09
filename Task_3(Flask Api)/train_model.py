import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns

# Load dataset
df = sns.load_dataset("titanic")

# Drop irrelevant or too-null columns
df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male"])
df.dropna(inplace=True)

# Encode categorical (both object and category types)
cat_cols = df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split into features and label
X = df.drop("survived", axis=1)
y = df["survived"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and column names
joblib.dump((model, X.columns.tolist()), "model.pkl")
print("✅ Model saved as model.pkl")
