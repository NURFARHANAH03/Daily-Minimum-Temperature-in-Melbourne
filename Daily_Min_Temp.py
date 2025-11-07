#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


#Load & Clean Dataset
df = pd.read_csv(
    r"C:\Users\LENOVO\task1_rnn\daily-minimum-temperatures-in-me.csv",
    header=None,
    names=["Date", "Temp"],
    on_bad_lines="skip"
)

df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Temp"] = pd.to_numeric(df["Temp"], errors='coerce')
df.dropna(inplace=True)


#Prepare Data
scaler = MinMaxScaler()
df["Temp"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

def prepare_data(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 7   # using last 7 days to predict next day
data = df["Temp"].values
X, y = prepare_data(data, time_step)

X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8) #80% training, 20% testing
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


#Build Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()


#Train Model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

#Prediction
y_pred = model.predict(X_test)

#Inverse scale back to original temperature
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_actual = scaler.inverse_transform(y_pred)


#Evaluation
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


#Plot Results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual Temperature", linewidth=2)
plt.plot(y_pred_actual, label="Predicted Temperature", linestyle='dashed')
plt.title("RNN Prediction: Daily Minimum Temperature")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()
