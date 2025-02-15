########################
# TensorFlow Algorithm #
########################

def TensorFlow(data, prediction_days, epochs, batch_size, future_days, security_name):
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
    test_data = data
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

    # Plot real prices and future prediction
    global plt
    plt.plot(predicted_prices, color="red", label=f"Predicted {security_name} Price")
    plt.plot(real_prices, label="Real Prices")
    plt.plot(range(len(real_prices), len(real_prices) + future_days), future_predictions, label="Future Predictions")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    return plt
    #plt.show()

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
from tkinter import *
from PIL import Image

# Content
data = None # to store pricing data
source = "" # stores if data is being pulled from a local file or online
plot = None # stores plot data

# Text content
TF_desc = "Make longer-term predictions using Google's TensorFlow AI"
APR_desc = "This uses pattern recognition to make shorter-term predictions "
Online_desc = "Use data loaded from the internet"
Local_desc = "Use a CSV file from your local drive"
Stage1_header = "Choose where your data will come from"
Stage2_header = "What algorithm will you use to make predictions"
Stage3_header = "Press the button to show the predictions on a graph!"
APR_settings_desc = '''
Threshold Probability is how 'sure' the model needs to be for it to predict a buy. The higher the number the more 'sure'.
'''
TF_settings_desc = '''
Prediction days ⇒ The number of days’ worth of data the model 
will take into consideration when making a prediction.

Epochs ⇒ The number of training iterations a model will undergo.

Batch Size ⇒ The amount of data that will be passed through the 
network in one training iteration before the model's internal parameters are updated.

'''

# Fonts
header = ("Gill Sans MT",50)
title = ("Gill Sans MT",30)

# images
APR_img = ctk.CTkImage(Image.open("Images/APR image.jpg"), size=(676,407))
TF_img = ctk.CTkImage(Image.open("Images/TF image.jpg"), size=(814,407))
Online_img = ctk.CTkImage(Image.open("Images/Online image.jpg"), size=(631,407))
Local_img = ctk.CTkImage(Image.open("Images/Local image.jpg"), size=(400,411))

# Making window
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("NEA Custom Theme.json")

window = ctk.CTk()  # Use CTk instead of Tk
window.title("Learn Stocks")
window.geometry("2500x1500")

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
S1_online_settings_frame = ctk.CTkFrame(master=S1_main_frame)

# GRID CONFIG
# left column
S1_main_left_frame.columnconfigure((0, 1, 2), weight = 1)
S1_main_left_frame.rowconfigure((0, 1, 2, 3, 4), weight = 1)

# right column
S1_main_right_frame.columnconfigure((0, 1, 2), weight = 1)
S1_main_right_frame.rowconfigure((0, 1, 2, 3, 4), weight = 1)

# Banner
S1_banner_frame.place(x = 0, y = 0, relwidth = 1, relheight = 0.2)
ctk.CTkLabel(S1_banner_frame, bg_color = "#2c5881", text=Stage1_header, font=header).pack(expand = True, fill ="both")
S1_main_frame.place(x = 0, rely = 0.2, relwidth = 1, relheight = 0.8)

# Stage 1 button functions
def STAGE1_Online_button_action():
    if S1_main_right_frame.winfo_ismapped():
        global source
        source = "Online"
        S1_main_right_frame.place_forget()
        S1_online_settings_frame.place(relx = 0.5, rely = 0, relwidth = 0.5, relheight = 1)
    else:
        S1_online_settings_frame.place_forget()
        S1_main_right_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

def STAGE1_Local_button_action():
    # opening files
    global data
    global source
    source = "Local"
    from customtkinter import filedialog
    filepath = filedialog.askopenfile(title="Open a CSV file", filetypes=[("Comma Separated Values", "*.csv")])
    data = open(filepath, "r")
    print(data.read())
    data.close()

# Left column
S1_main_left_frame.place(relx = 0, rely = 0, relwidth = 0.5, relheight = 1)
ctk.CTkLabel(S1_main_left_frame, text="Online", font=title).grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(S1_main_left_frame, text="", image=Online_img).grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
ctk.CTkLabel(master=S1_main_left_frame, text=Online_desc).grid(row = 3, column = 0, sticky="nesw", columnspan = 3)
STAGE1_L_button = ctk.CTkButton(master=S1_main_left_frame, text="Online", command=STAGE1_Online_button_action).grid(row = 4, column = 1, sticky="ew", columnspan = 1)

# Right column
S1_main_right_frame.place(relx = 0.5, rely = 0, relwidth = 0.5, relheight = 1)
ctk.CTkLabel(master=S1_main_right_frame, text="Local", font=title).grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(master=S1_main_right_frame, image=Local_img, text="").grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
ctk.CTkLabel(master=S1_main_right_frame, text=Local_desc).grid(row = 3, column = 0, sticky="nesw", columnspan = 3)
STAGE1_R_button = ctk.CTkButton(master=S1_main_right_frame, text="Local", command=STAGE1_Local_button_action).grid(row = 4, column = 1, sticky="ew", columnspan = 1)

# Online settings frame
# Dropdown box
company = ctk.CTkComboBox(S1_online_settings_frame, values=["META", "^GSPC", "EURUSD=X"])
company.set("select a company")
company.pack(padx=300, pady=40, fill="x")

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
ctk.CTkLabel(S2_banner_frame, bg_color = "#2c5881", font=header, text=Stage2_header).pack(expand = True, fill = "both")

S2_main_frame.place(x = 0, rely = 0.2, relwidth = 1, relheight = 0.8)

S2_main_left_frame.place(relx = 0, rely = 0, relwidth = 0.5, relheight = 1)
S2_main_right_frame.place(relx = 0.5, rely = 0, relwidth = 0.5, relheight = 1)

# APR settings pane
S2_APR_settings_frame = ctk.CTkFrame(master=S2_main_frame)

# Function to update the label with prob_slider value
def prob_update_value(value):
    prob_slider.set(value)
    prob_label.configure(text=f"Threshold Probability: {round(prob_slider.get(), 2)}")  # Format to 2 decimal places

# Slider
prob_slider = ctk.DoubleVar(value=0.68)  # Initial value set to 0.68
prob_slider = ctk.CTkSlider(S2_APR_settings_frame, from_=0, to=1, command=prob_update_value)
prob_slider.set(0.68)  # Set initial value

prob_label = ctk.CTkLabel(S2_APR_settings_frame, text=f"Threshold Probability: {round(prob_slider.get(), 2)}")
prob_label.pack(pady=20)
prob_slider.pack(pady=20)

# APR run button
def APR_run_func():
    import yfinance as yf
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    global source
    global data
    if source == "Online":
        ticker = yf.Ticker(company.cget("state"))
        data = ticker.history(period="max")

    ARP_algorithm(data, round(prob_slider.get()))
    canvas = FigureCanvasTkAgg(plt, master=S3_main_frame)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()


APR_run = ctk.CTkButton(master=S2_APR_settings_frame, text="RUN", command=APR_run_func).pack(pady=20)
ctk.CTkLabel(S2_APR_settings_frame, text=APR_settings_desc).pack()

# TF settings pane
S2_TF_settings_frame = ctk.CTkFrame(master=S2_main_frame)

# Sliders
# Slider update functions
def epoch_update_value(value):
    epoch_slider.set(value)
    epoch_label.configure(text=f"Epochs: {round(epoch_slider.get())}")  # Format to 2 decimal places

def pred_days_update_value(value):
    pred_days_slider.set(value)
    pred_days_label.configure(text=f"Prediction Days: {round(pred_days_slider.get())}")  # Format to 2 decimal places

def batch_size_update_value(value):
    batch_size_slider.set(value)
    batch_size_label.configure(text=f"Batch Size: {round(batch_size_slider.get())}")  # Format to 2 decimal places

def future_days_update_value(value):
    future_days_slider.set(value)
    future_days_label.configure(text=f"Future Days: {round(future_days_slider.get())}")  # Format to 2 decimal places


epoch_slider = ctk.CTkSlider(S2_TF_settings_frame, from_=1, to=50, command=epoch_update_value)
epoch_slider.set(25)
epoch_label = ctk.CTkLabel(S2_TF_settings_frame, text=f"Epochs: {epoch_slider.get()}")
epoch_label.pack(pady=20)
epoch_slider.pack(pady=20)

pred_days_slider = ctk.CTkSlider(S2_TF_settings_frame, from_=10, to=90, command=pred_days_update_value)
pred_days_slider.set(60)
pred_days_label = ctk.CTkLabel(S2_TF_settings_frame, text=f"Prediction Days: {pred_days_slider.get()}")
pred_days_label.pack(pady=20)
pred_days_slider.pack(pady=20)

batch_size_slider = ctk.CTkSlider(S2_TF_settings_frame, from_=5, to=80, command=batch_size_update_value)
batch_size_slider.set(32)
batch_size_label = ctk.CTkLabel(S2_TF_settings_frame, text=f"Batch Size: {batch_size_slider.get()}")
batch_size_label.pack(pady=20)
batch_size_slider.pack(pady=20)

future_days_slider = ctk.CTkSlider(S2_TF_settings_frame, from_=1, to=100, command=future_days_update_value)
future_days_slider.set(50)
future_days_label = ctk.CTkLabel(S2_TF_settings_frame, text=f"Future Days: {future_days_slider.get()}")
future_days_label.pack(pady=20)
future_days_slider.pack(pady=20)

# TF run button
def TF_run_func():
    import yfinance as yf
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
    global source
    global data
    if source == "Online":
        ticker = yf.Ticker(company.get())
        data = ticker.history(period="max")

    TensorFlow(data,
                 round(pred_days_slider.get()),
                 round(epoch_slider.get()),
                 round(batch_size_slider.get()),
                 round(future_days_slider.get()),
                 company.get())


TF_run = ctk.CTkButton(master=S2_TF_settings_frame, text="RUN", command=TF_run_func).pack(pady=20)
ctk.CTkLabel(S2_TF_settings_frame, text=TF_settings_desc).pack()


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
ctk.CTkLabel(S2_main_left_frame, font=title, text="APR").grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(S2_main_left_frame, text="", image=APR_img).grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
ctk.CTkLabel(master=S2_main_left_frame, text=APR_desc).grid(row = 3, column = 0, sticky="nesw", columnspan = 3)
STAGE2_L_button = ctk.CTkButton(master=S2_main_left_frame, text="APR", command=STAGE2_L_button_action).grid(row = 4, column = 1, sticky="ew", columnspan = 1)

# right column content
ctk.CTkLabel(master=S2_main_right_frame, font=title, text="TensorFlow").grid(row = 0, column = 0, sticky="nesw", columnspan = 3)
ctk.CTkLabel(master=S2_main_right_frame, image=TF_img, text="").grid(row = 1, column = 0, sticky="nesw", columnspan = 3, rowspan=2)
ctk.CTkLabel(master=S2_main_right_frame, text=TF_desc).grid(row = 3, column = 0, sticky="nesw", columnspan = 3)
STAGE2_R_button = ctk.CTkButton(master=S2_main_right_frame, text="TensorFlow", command=STAGE2_R_button_action).grid(row = 4, column = 1, sticky="ew", columnspan = 1)

# CONTENT FOR STAGE3
# main layout widgets
S3_banner_frame = ctk.CTkFrame(master=Stage3)
S3_main_frame = ctk.CTkFrame(master=Stage3)

# place layout
S3_banner_frame.place(x = 0, y = 0, relwidth = 1, relheight = 0.2)
ctk.CTkLabel(S3_banner_frame, bg_color = "#2c5881", font=header, text=Stage3_header).pack(expand = True, fill = "both")

S3_main_frame.place(x = 0, rely = 0.2, relwidth = 1, relheight = 0.8)

# Plot button
def STAGE3_plot_func():
    plt.show()

STAGE3_plot_button = ctk.CTkButton(master=S3_main_frame, text="PLOT", command=STAGE3_plot_func).place(relx=0.5, rely=0.5, anchor="center")

#########
# START #
#########
# run
window.mainloop()