# ================================================
# RNN Temperature Forecasting + Fairness in Dataset
# With Before/After Cleaning Comparison (Final Version)
# ================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# =====================================================
# 1. LOAD RAW DATA (Dirty Data for Fairness EDA)
# =====================================================
df_dirty = pd.read_csv(
    r"C:\Users\LENOVO\task1_rnn\daily-minimum-temperatures-in-me.csv",
    header=None,
    names=["Date", "Temp"],
    on_bad_lines="skip"
)

# Convert types (for plotting only, still keeps the messy data)
df_dirty["Date"] = pd.to_datetime(df_dirty["Date"], errors="ignore")
df_dirty["Temp"] = pd.to_numeric(df_dirty["Temp"], errors="ignore")


# =====================================================
# BEFORE CLEANING: EDA (Shows Messy Data)
# =====================================================

plt.figure(figsize=(14,5))

# Histogram BEFORE cleaning (shows corruption)
plt.subplot(1,2,1)
plt.hist(df_dirty["Temp"], bins=30, color="red", edgecolor="black")
plt.title("Temperature Distribution (Before Cleaning)")
plt.xlabel("Temp")
plt.ylabel("Count")

# Trend BEFORE cleaning (messy on purpose)
plt.subplot(1,2,2)
plt.plot(df_dirty["Temp"], color="red")
plt.title("Temperature Trend (Before Cleaning)")
plt.xlabel("Record Index")
plt.ylabel("Temp")

plt.tight_layout()
plt.show()


# =====================================================
# 2. CLEAN DATA (Actual Cleaning For Model)
# =====================================================
df = df_dirty.copy()

df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Temp"] = pd.to_numeric(df["Temp"], errors='coerce')

df.dropna(inplace=True)   # remove invalid rows


# =====================================================
# AFTER CLEANING: EDA
# =====================================================

plt.figure(figsize=(14,5))

# Histogram AFTER cleaning
plt.subplot(1,2,1)
plt.hist(df["Temp"], bins=30, color="green", edgecolor="black")
plt.title("Temperature Distribution (After Cleaning)")
plt.xlabel("Temp")
plt.ylabel("Count")

# Trend AFTER cleaning
plt.subplot(1,2,2)
plt.plot(df["Temp"].values, color="green")
plt.title("Temperature Trend (After Cleaning)")
plt.xlabel("Record Index")
plt.ylabel("Temp")

plt.tight_layout()
plt.show()

# Boxplot AFTER cleaning
plt.figure(figsize=(6,4))
plt.boxplot(df["Temp"])
plt.title("Temperature Boxplot (After Cleaning)")
plt.ylabel("Temperature")
plt.show()

# Rolling mean AFTER cleaning
plt.figure(figsize=(10,4))
plt.plot(df["Temp"].rolling(30).mean(), label="30-Day Rolling Mean", color="blue")
plt.title("Rolling Average Temperature (After Cleaning)")
plt.xlabel("Days")
plt.ylabel("Temp")
plt.legend()
plt.show()


# =====================================================
# 3. NORMALIZATION (Fairness Step)
# =====================================================
scaler = MinMaxScaler()
df["Temp"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))


# =====================================================
# 4. PREPARE DATA FOR RNN
# =====================================================
def prepare_data(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 7
data = df["Temp"].values

X, y = prepare_data(data, time_step)
X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# =====================================================
# 5. BUILD RNN MODEL
# =====================================================
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()


# =====================================================
# 6. TRAINING
# =====================================================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# =====================================================
# 7. PREDICTION + PERFORMANCE
# =====================================================
y_pred = model.predict(X_test)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print("\n===== MODEL PERFORMANCE =====")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")


# =====================================================
# 8. VISUALIZE PREDICTION
# =====================================================
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual Temperature", linewidth=2)
plt.plot(y_pred_actual, label="Predicted Temperature", linestyle="dashed")
plt.title("RNN Prediction: Daily Minimum Temperature")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()
# =====================================================
# 9. DIRTY MODEL TRAINING (Fixed)
# =====================================================
def create_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(7, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


df_dirty_cleaned = df_dirty.copy()
df_dirty_cleaned["Temp"] = pd.to_numeric(df_dirty_cleaned["Temp"], errors="coerce")
df_dirty_cleaned = df_dirty_cleaned.dropna(subset=["Temp"])

data_dirty = df_dirty_cleaned["Temp"].values

# Create scaler for dirty dataset
scaler2 = MinMaxScaler()

data_dirty_scaled = scaler2.fit_transform(data_dirty.reshape(-1,1))

X_dirty, y_dirty = prepare_data(data_dirty_scaled, time_step=7)

model_dirty = create_model()
model_dirty.fit(X_dirty[:100], y_dirty[:100], epochs=3, verbose=0)

y_pred_dirty = model_dirty.predict(X_dirty[-len(y_test):])
y_pred_dirty_actual = scaler.inverse_transform(y_pred_dirty)

mse_dirty = mean_squared_error(y_test_actual, y_pred_dirty_actual)
rmse_dirty = np.sqrt(mse_dirty)

mse_clean = mean_squared_error(y_test_actual, y_pred_actual)
rmse_clean = np.sqrt(mse_clean)

print("RMSE before cleaning :", rmse_dirty)
print("RMSE after cleaning  :", rmse_clean)

