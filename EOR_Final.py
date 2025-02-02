import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_excel("D:/eor/eordata.xlsx")
df = df.dropna().reset_index(drop=True)

print(df.head())
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

class_counts = df['Observed'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.xlabel("Class", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Class Distribution", fontsize=14)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(class_counts)))
plt.title("Class Distribution (Pie Chart)", fontsize=14)
plt.show()

df = df.drop(columns=["Country", "Number"])
X = df.drop(columns=["Observed"])
y = df["Observed"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print(f"Train set: {X_train_scaled.shape}, Validation set: {X_val_scaled.shape}, Test set: {X_test_scaled.shape}")

rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [11, 20, 30],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2, return_train_score=True)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")

y_val_pred = best_rf.predict(X_val_scaled)
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")

y_test_pred = best_rf.predict(X_test_scaled)
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred))
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train_scaled, y_train)

y_pred = xgb_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print(f"Train set: {X_train_scaled.shape}, Validation set: {X_val_scaled.shape}, Test set: {X_test_scaled.shape}")

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
]

meta_model = LogisticRegression()
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_model, passthrough=True)
stacking_clf.fit(X_train_scaled, y_train)

y_pred = stacking_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Accuracy: {accuracy:.4f}")

y_pred_labels = label_encoder.inverse_transform(y_pred)

input_shape = X_train_scaled.shape[1]
num_classes = len(label_encoder.classes_)

model = Sequential([
    Dense(256, input_shape=(input_shape,)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(128),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(64),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test_scaled)
y_pred_classes = y_pred.argmax(axis=1)
