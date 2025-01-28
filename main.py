########################
# TensorFlow Algorithm #
########################

def TensorFlow(data, prediction_days, epochs, batch_size, future_days):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import datetime as dt
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM

    # Prep data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    # Make training arrays
    x_train = []
    y_train = []

    # Append historic data
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # Convert to NumPy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Building model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of next closing price
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Setting start time and future target
    test_start = dt.datetime(2022, 1, 1)
    future_days = future_days

    # Loading data
    test_data = ticker.history(period="2y")
    real_prices = test_data["Close"].values
    total_dataset = pd.concat((data["Close"], test_data["Close"]))

    # Prepping inputs
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Prepare the last `prediction_days` inputs for future predictions
    last_sequence = model_inputs[-prediction_days:]  # Take the last `prediction_days` values
    last_sequence = np.reshape(last_sequence, (1, prediction_days, 1))  # Reshape to match model input

    # Store predictions
    future_predictions = []

    # Predict future prices
    for i in range(future_days):
        # Predict the next price
        predicted_price = model.predict(last_sequence)

        # Inverse transform to get the price back to the original scale
        predicted_price = scaler.inverse_transform(predicted_price)
        future_predictions.append(predicted_price[0, 0])  # Save the predicted price

        # Update `last_sequence` with the new prediction
        new_input = scaler.transform([[predicted_price[0, 0]]])  # Scale the new prediction
        new_input = np.reshape(new_input, (1, 1, 1))  # Reshape to match (1, 1, 1) for concatenation
        last_sequence = np.append(last_sequence[:, 1:, :], new_input, axis=1)

    # STORING HISTORICAL PREDICTIONS
    # Storing past prices
    x_past = []

    # Filling x_past
    for i in range(prediction_days, len(model_inputs)):
        x_past.append(model_inputs[i - prediction_days:i, 0])

    # Prepping x_past
    x_past = np.array(x_past)
    x_past = np.reshape(x_past, (x_past.shape[0], x_past.shape[1], 1))

    # Predicting for x_past
    predicted_prices = model.predict(x_past)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot real prices and future predictions
    plt.plot(predicted_prices, color="red", label=f"Predicted {security_name} Price")
    plt.plot(real_prices, label="Real Prices")
    plt.plot(range(len(real_prices), len(real_prices) + future_days), future_predictions, label="Future Predictions")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

#################
# ARP Algorithm #
#################
def ARP_algorithm(prices, prob):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier

    # Drop unnecessary columns
    prices.drop(columns=["Dividends", "Stock Splits"], inplace=True)

    # Add target column
    prices["Tomorrow"] = prices["Close"].shift(-1)
    prices["Target"] = (prices["Tomorrow"] > prices["Close"]).astype(int)

    # Limit data to after 1990
    prices = prices.loc["1990-01-01":].copy()

    # Define training and testing sets
    train = prices.iloc[:-100]
    test = prices.iloc[-100:]

    # Initial predictors
    predictors = ["Close", "Volume", "Open", "High", "Low"]

    # Add rolling predictors
    horizons = [2, 5, 60, 250, 1000]  # Different time horizons
    new_predictors = []

    for horizon in horizons:
        # Rolling averages
        rolling_averages = prices["Close"].rolling(horizon).mean()
        prices[f"Close_Ratio_{horizon}"] = prices["Close"] / rolling_averages

        # Rolling trend
        trend_sum = prices["Target"].shift(1).rolling(horizon).sum()
        prices[f"Trend_{horizon}"] = trend_sum

        # Add columns to predictors
        new_predictors += [f"Close_Ratio_{horizon}", f"Trend_{horizon}"]

    # Drop rows with NaNs introduced by rolling computations
    prices.dropna(inplace=True)

    # Update training and testing sets after dropping NaNs
    train = prices.iloc[:-100]
    test = prices.iloc[-100:]

    # Combine initial and new predictors
    both_predictors = predictors + new_predictors

    # Define model
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

    # Define prediction function
    def predict(train, test, predictors, model, prob):
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:, 1]
        preds[preds >= prob] = 1
        preds[preds < prob] = 0
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined

    # Generate predictions
    predictions = predict(train, test, new_predictors, model, prob)

    # Check the newest value in the predictions and print BUY or SELL
    latest_prediction = predictions["Predictions"].iloc[-1]
    if latest_prediction == 1:
        action = "Buy"
        colour = "green"
    else:
        action = "Sell"
        colour = "red"

    # Plot the prices
    plt.figure(figsize=(10, 6))
    plt.plot(prices["Close"], label="Actual Prices", color="blue", alpha=0.7)
    plt.title("Stock Prices with Prediction", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)

    # Add the text on the chart for the newest prediction
    plt.text(
        x=0.02,  # Relative x-position (2% from the left)
        y=0.93,  # Relative y-position (93% from the bottom)
        s=action,  # Buy/Sell
        color=colour,
        fontsize=20,
        weight="bold",
        transform=plt.gca().transAxes,  # Use axes coordinates for positioning
        horizontalalignment="left"
    )

    # Add legend and grid
    plt.legend(["Actual Prices"])
    plt.grid(alpha=0.3)
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

#########
# START #
#########

# FOR TF
import yfinance as yf
import datetime as dt
securities = ["META", "^GSPC", "EURUSD=X"]
print('''
1. META
2. S&P500
3. EUR/USD
''')
# taking inputs for TF
security_select = int(input("Select the security to predict >> "))
prediction_days = int(input("Enter the number of prediction days"))
epochs = int(input("Enter the number of training epochs"))
batch_size = int(input("Enter the training batch size"))
future_days = int(input("Enter number of days you want to predict"))
security_name = securities[security_select-1]
ticker = yf.Ticker(security_name)

start = dt.datetime(1990,1,1)
end = dt.datetime(2022,1,1)

data = ticker.history(period = "max")

TensorFlow(data, prediction_days, epochs, batch_size, 15)

# FOR APR
prices = yf.Ticker("^GSPC")
prices = prices.history(period="max")
ARP_algorithm(prices, 0.68)
