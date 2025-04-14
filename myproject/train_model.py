import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib 

# ✅ Load dataset
df = pd.read_csv("cognitive_load_dataset.csv")

# ✅ Drop rows with missing key features (not everything)
df = df.dropna(subset=["task_difficulty", "error_rate", "response_time"])

# ✅ Encode label
label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
df = df[df["label"].isin(label_map)]  # remove any rows with invalid labels
df["label"] = df["label"].map(label_map)

# ✅ Prepare features and target
X = df.drop("label", axis=1)
y = df["label"]

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ✅ Save model
joblib.dump(model, "trained_cognitive_load_model.pkl")
print("✅ Model saved as trained_cognitive_load_model.pkl")
