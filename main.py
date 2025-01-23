########################
# TensorFlow Algorithm #
########################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Loading data
securities = ["META", "^GSPC", "EURUSD=X"]
print('''
1. META
2. S&P500
3. EUR/USD
''')
security_select = int(input("Select the security to predict >> "))

security_name = securities[security_select-1]
ticker = yf.Ticker(security_name)

start = dt.datetime(1990,1,1)
end = dt.datetime(2022,1,1)

data = ticker.history(period = "max")

# Prep data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

#prediction_days = 60
prediction_days = int(input("How many previous days do you want to consider >> "))


x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))

# Building model
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1)) # Prediction of next closing price
#epochs = 25
#batch_size = 32
epochs = int(input("How many epochs to train the model >> "))
batch_size = int(input("What batch size for training the model >> "))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

# Testing model accuracy on past data

# Loading test data
test_start = dt.datetime(2022,1,1)
test_end = dt.datetime.now()

test_data = ticker.history(period = "2y")
real_prices = test_data["Close"].values

total_dataset = pd.concat((data["Close"], test_data["Close"]))

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Making predictions for test data

x_test = []

for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i - prediction_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting
plt.plot(real_prices, color = "blue", label = f"Real {security_name} Price")
plt.plot(predicted_prices, color = "red", label = f"Predicted {security_name} Price")
plt.title(f"{security_name} Real vs Predicted Share Price (Prediction Days: {prediction_days}, Epochs: {epochs}, Batch Size: {batch_size})")
plt.xlabel("Time")
plt.ylabel("Share Price")
plt.legend()
plt.show()

#################
# ARP Algorithm #
#################
import yfinance as yf
import pandas as pd

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period = "max")


# In[3]:


print(sp500)
print(sp500.index)

sp500.plot.line(y = "Close", use_index = True)

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)

print(sp500)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# this makes it so that a True is given as a 1 and False as a 0
print(sp500)

sp500 = sp500.loc["1990-01-01":].copy()

print(sp500)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])

print(preds)

preds = pd.Series(preds, index = test.index)

print(preds)

precision_score(test["Target"], preds)

# Here it is shown that the model we have trained is only accurate 50% of the time.

combined = pd.concat([test["Target"], preds], axis = 1)

combined.plot()


# START OF BACKTESTING

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

def backtest(data, model, predictors, start = 2500 , step = 250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)


predictions["Predictions"].value_counts()


precision_score(predictions["Target"], predictions["Predictions"])


# To see the percentage of days the market went up, we can do this by taking the number of each outcomes of the target and dividing it by the number of days (rows)

predictions["Target"].value_counts() / predictions.shape[0]


# ADDING NEW PREDICTORS

horizons = [2, 5, 60, 250, 1000] #2 days, 5 days, 3 months, 1 year and 4 years
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

print(sp500)

sp500 = sp500.dropna()

print(sp500)


# just new predictors and using probs

# *********

# UPDATING MODEL V1 (old and new predictors)

both_predictors = predictors + new_predictors

print(both_predictors)


print(predictors)

print(new_predictors)

model = RandomForestClassifier(n_estimators = 200, min_samples_split = 50, random_state = 1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

predictions = backtest(sp500, model, both_predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])


# UPDATING MODEL V2 (just new predictors)

predictions = backtest(sp500, model, new_predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

# UPDATING MODEL V3 (the best predictor combination used and prababilities)

model = RandomForestClassifier(n_estimators = 200, min_samples_split = 50, random_state = 1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

predictions = backtest(sp500, model, new_predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

# *********

# MAKING OPTIMISE FUNCTION

def opt_predict(train, test, predictors, model, prob):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= prob] = 1
    preds[preds < prob] = 0
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

def opt_backtest(data, model, predictors, prob, start = 2500 , step = 250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = opt_predict(train, test, predictors, model, prob)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def optimise(data, model, predictors, lower, upper, step):
    results = [[],[]]
    test_prob = lower
    while test_prob < (upper + step):
        predictions = opt_backtest(sp500, model, predictors, test_prob)
        results[1].append(precision_score(predictions["Target"], predictions["Predictions"]))
        results[0].append(round(test_prob, 2))
        test_prob = test_prob + round(step, 2)
    return results

import matplotlib.pyplot as plt
import numpy as np

results = optimise(sp500, model, new_predictors, 0.5, 0.75, 0.005)

print(results)

xpoints = np.array(results[0])
ypoints = np.array(results[1])

plt.plot(xpoints, ypoints)
plt.show()

################
# CustomTk GUI #
################
import customtkinter as ctk

# Making window
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("NEA Custom Theme.json")

window = ctk.CTk()  # Use CTk instead of Tk
window.title("CustomTkinter Transition")
window.geometry("1000x800")

# Making notebook (CustomTkinter doesn't have a direct notebook widget, so I use CTkTabview)
notebook = ctk.CTkTabview(master=window, width=1000, height=1000)
notebook.pack(expand=True, fill="both")

# Adding tabs
notebook.add("Tab 1")
notebook.add("Tab 2")
notebook.add("Tab 3")

# Getting frames for each tab
frame1 = notebook.tab("Tab 1")
frame2 = notebook.tab("Tab 2")
frame3 = notebook.tab("Tab 3")

# Making UI elements
# Frame 1
label1 = ctk.CTkLabel(master=frame1, text="This is tab 1")
label1.pack(expand=True, fill="both")

from customtkinter import filedialog
def openFile():
    filepath = filedialog.askopenfile(title = "Open a CSV file",
                                          filetypes = [("Comma Separated Values", "*.csv")])
    file = open(filepath, "r")
    print(file.read())
    file.close()

file_button = ctk.CTkButton(master=frame1, text="Open", command=openFile)
file_button.pack()

# Frame 2
label2 = ctk.CTkLabel(frame2, text="This is tab 2")
button1 = ctk.CTkButton(frame2, text="Button 1")
button2 = ctk.CTkButton(frame2, text="Button 2")
# Adding a slider to Tab 2
slider = ctk.CTkSlider(frame2, from_=0, to=100)  # Slider range from 0 to 100
slider.pack(pady=20)  # Add vertical padding for spacing
button1.pack(pady=10)
button2.pack(pady=10)
label2.pack(expand=True, fill="both")

# Frame 3
label3 = ctk.CTkLabel(frame3, text="This is tab 3")
entry1 = ctk.CTkEntry(frame3)
entry2 = ctk.CTkEntry(frame3)
label3.grid(row=0, column=0, padx=10, pady=10)
entry1.grid(row=1, column=0, padx=10, pady=10)
entry2.grid(row=1, column=1, padx=10, pady=10)

# frame 3 grid weighting
frame3.columnconfigure(0, weight=1)
frame3.columnconfigure(1, weight=1)
frame3.rowconfigure(0, weight=1)
frame3.rowconfigure(1, weight=1)

# run
window.mainloop()



