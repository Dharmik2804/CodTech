# 📝 TASK 1: ETL Pipeline using Pandas + Scikit-learn

# ✅ STEP 1: IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# ✅ STEP 2: DEFINE ETL FUNCTIONS

# 📥 Extract Function
def extract_data():
    # Titanic dataset from seaborn for simplicity
    df = sns.load_dataset("titanic")
    return df

# 🔄 Transform Function
def transform_data(df):
    # Separate features and target
    X = df.drop(columns=["survived"])
    y = df["survived"]

    # Define numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Drop columns with too many missing values or irrelevant ones
    drop_cols = ["deck", "embark_town", "alive"]  # too many NaNs or duplicates
    X = X.drop(columns=drop_cols, errors='ignore')

    # Recompute after drop
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both
    full_pipeline = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Fit and transform
    X_transformed = full_pipeline.fit_transform(X)

    # Get feature names after OneHotEncoding
    encoded_cat_cols = full_pipeline.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols)
    all_columns = np.concatenate([num_cols, encoded_cat_cols])

    # Return as DataFrame
    X_cleaned_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed,
                                columns=all_columns)

    # Include target back
    X_cleaned_df["survived"] = y.reset_index(drop=True)

    return X_cleaned_df

# 💾 Load Function
def load_data(cleaned_df, file_path="cleaned_titanic.csv"):
    cleaned_df.to_csv(file_path, index=False)
    print(f"✅ Cleaned data saved to {file_path}")

# ✅ STEP 3: RUN THE ETL PIPELINE
if __name__ == "__main__":
    print("🚀 Starting ETL Process...")
    raw_data = extract_data()
    cleaned_data = transform_data(raw_data)
    load_data(cleaned_data)
    print("✅ ETL Pipeline Completed.")

