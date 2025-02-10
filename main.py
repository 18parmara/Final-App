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
window.title("Learn Stocks")
window.geometry("1000x800")

# Making notebook (CustomTkinter doesn't have a direct notebook widget, so I use CTkTabview)
notebook = ctk.CTkTabview(master=window, width=1000, height=1000)
notebook.pack(expand=True, fill="both")

# Adding tabs
notebook.add("Data")
notebook.add("Settings")
notebook.add("Output")

# Getting frames for each tab
Stage1 = notebook.tab("Data")
Stage2 = notebook.tab("Settings")
Stage3 = notebook.tab("Output")

# CONTENT FOR STAGE1
# main layout widgets
S1_banner_frame = ctk.CTkFrame(master=Stage1)
S1_main_frame = ctk.CTkFrame(master=Stage1)
S1_main_left_frame = ctk.CTkFrame(master=S1_main_frame)
S1_main_right_frame = ctk.CTkFrame(master=S1_main_frame)

# GRID CONFIG
# left column
S1_main_left_frame.columnconfigure((0, 1, 2), weight = 1)
S1_main_left_frame.rowconfigure((0, 1, 2, 3, 4), weight = 1)

# right column
S1_main_right_frame.columnconfigure((0, 1, 2), weight = 1)
S1_main_right_frame.rowconfigure((0, 1, 2, 3, 4), weight = 1)

# Banner
S1_banner_frame.place(x = 0, y = 0, relwidth = 1, relheight = 0.2)
ctk.CTkLabel(S1_banner_frame, bg_color ="green", text="HEADER").pack(expand = True, fill ="both")
S1_main_frame.place(x = 0, rely = 0.2, relwidth = 1, relheight = 0.8)

# Left column
S1_main_left_frame.place(relx = 0, rely = 0, relwidth = 0.5, relheight = 1)
ctk.CTkLabel(S1_main_left_frame, text="Online").grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(S1_main_left_frame, text="L IMAGE", bg_color ="red").grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
STAGE1_L_button = ctk.CTkButton(master=S1_main_left_frame, text="LEFT").grid(row = 4, column = 1, sticky="ew", columnspan = 1)

# Right column
S1_main_right_frame.place(relx = 0.5, rely = 0, relwidth = 0.5, relheight = 1)
ctk.CTkLabel(master=S1_main_right_frame, text="Local").grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(master=S1_main_right_frame, text="R IMAGE", bg_color ="red").grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
STAGE1_R_button = ctk.CTkButton(master=S1_main_right_frame, text="RIGHT").grid(row = 4, column = 1, sticky="ew", columnspan = 1)

# CONTENT FOR STAGE2
# main layout widgets
S2_banner_frame = ctk.CTkFrame(master=Stage2)
S2_main_frame = ctk.CTkFrame(master=Stage2)
S2_main_left_frame = ctk.CTkFrame(master=S2_main_frame)
S2_main_right_frame = ctk.CTkFrame(master=S2_main_frame)


# Grid Config
# left column
S2_main_left_frame.columnconfigure((0,1,2), weight = 1)
S2_main_left_frame.rowconfigure((0,1,2,3,4), weight = 1)

# right column
S2_main_right_frame.columnconfigure((0,1,2), weight = 1)
S2_main_right_frame.rowconfigure((0,1,2,3,4), weight = 1)


# place layout
S2_banner_frame.place(x = 0, y = 0, relwidth = 1, relheight = 0.2)
ctk.CTkLabel(S2_banner_frame, bg_color = "green", text="HEADER").pack(expand = True, fill = "both")

S2_main_frame.place(x = 0, rely = 0.2, relwidth = 1, relheight = 0.8)
#ctk.CTkLabel(S2_main_frame, bg_color = "blue").pack(expand = True, fill = "both")

S2_main_left_frame.place(relx = 0, rely = 0, relwidth = 0.5, relheight = 1)
S2_main_right_frame.place(relx = 0.5, rely = 0, relwidth = 0.5, relheight = 1)

# APR settings pane
S2_APR_settings_frame = ctk.CTkFrame(master=S2_main_frame)
prob_slider = ctk.DoubleVar(value=5)  # Initial value set to 50

# Function to update the label with prob_slider value
def prob_update_value(value):
    prob_slider.set(value)
    label.configure(text=f"Value: {round(prob_slider.get())}")  # Format to 2 decimal places

# Label to display the value
label = ctk.CTkLabel(S2_APR_settings_frame, text=f"Value: {prob_slider.get()}")
label.pack(pady=20)

# Slider
prob_slider = ctk.CTkSlider(S2_APR_settings_frame, from_=0, to=10, command=prob_update_value)
prob_slider.set(prob_slider.get())  # Set initial value
prob_slider.pack(pady=20)

# Dropdown box
company = ctk.CTkComboBox(S2_APR_settings_frame, values=["company 1", "company 2"])
company.set("select a company")
company.pack(pady=20)
# APR settings pane
S2_APR_settings_frame = ctk.CTkFrame(master=S2_main_frame)
prob_slider = ctk.DoubleVar(value=5)  # Initial value set to 5

# Function to update the label with prob_slider value
def update_value(value):
    prob_slider.set(value)
    label.configure(text=f"Value: {round(prob_slider.get())}")  # Format to 2 decimal places

# Label to display the value
label = ctk.CTkLabel(S2_APR_settings_frame, text=f"Value: {prob_slider.get()}")
label.pack(pady=20)

# Slider
prob_slider = ctk.CTkSlider(S2_APR_settings_frame, from_=0, to=10, command=update_value)
prob_slider.set(prob_slider.get())  # Set initial value
prob_slider.pack(pady=20)

# Dropdown box
company = ctk.CTkComboBox(S2_APR_settings_frame, values=["company 1", "company 2"])
company.set("select a company")
company.pack(pady=20)

# TF settings pane
S2_TF_settings_frame = ctk.CTkFrame(master=S2_main_frame)

# Sliders
# Slider update functions
def epoch_update_value(value):
    epoch_slider.set(value)
    epoch_label.configure(text=f"Value: {round(epoch_slider.get())}")  # Format to 2 decimal places

def pred_days_update_value(value):
    pred_days_slider.set(value)
    pred_days_label.configure(text=f"Value: {round(pred_days_slider.get())}")  # Format to 2 decimal places

def batch_size_update_value(value):
    batch_size_slider.set(value)
    batch_size_label.configure(text=f"Value: {round(batch_size_slider.get())}")  # Format to 2 decimal places

epoch_slider = ctk.CTkSlider(S2_TF_settings_frame, from_=0, to=10, command=epoch_update_value)
epoch_label = ctk.CTkLabel(S2_TF_settings_frame, text=f"Value: {epoch_slider.get()}")
epoch_label.pack(pady=20)
epoch_slider.pack(pady=20)

pred_days_slider = ctk.CTkSlider(S2_TF_settings_frame, from_=0, to=10, command=pred_days_update_value)
pred_days_label = ctk.CTkLabel(S2_TF_settings_frame, text=f"Value: {pred_days_slider.get()}")
pred_days_label.pack(pady=20)
pred_days_slider.pack(pady=20)

batch_size_slider = ctk.CTkSlider(S2_TF_settings_frame, from_=0, to=10, command=batch_size_update_value)
batch_size_label = ctk.CTkLabel(S2_TF_settings_frame, text=f"Value: {batch_size_slider.get()}")
batch_size_label.pack(pady=20)
batch_size_slider.pack(pady=20)


def STAGE2_L_button_action():
    if S2_main_right_frame.winfo_ismapped():
        S2_main_right_frame.place_forget()
        S2_APR_settings_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    else:
        S2_main_right_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        S2_APR_settings_frame.place_forget()

def STAGE2_R_button_action():
    if S2_main_left_frame.winfo_ismapped():
        S2_main_left_frame.place_forget()
        S2_TF_settings_frame.place(relx=0, rely=0, relwidth=0.5, relheight=1)
    else:
        S2_main_left_frame.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        S2_TF_settings_frame.place_forget()


# left column content
ctk.CTkLabel(S2_main_left_frame, text="APR").grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(S2_main_left_frame, text="", bg_color = "red").grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
STAGE2_L_button = ctk.CTkButton(master=S2_main_left_frame, text="LEFT", command=STAGE2_L_button_action).grid(row = 4, column = 1, sticky="ew", columnspan = 1)

# right column content
ctk.CTkLabel(master=S2_main_right_frame, text="TensorFlow").grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(master=S2_main_right_frame, text="R IMAGE", bg_color = "red").grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
ctk.CTkLabel(master=S2_main_right_frame, text="How to keep text in a frame").grid(row = 3, column = 0, sticky="nesw", columnspan = 3)
STAGE2_R_button = ctk.CTkButton(master=S2_main_right_frame, text="RIGHT", command=STAGE2_R_button_action).grid(row = 4, column = 1, sticky="ew", columnspan = 1)

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
