import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# Compute the Moving Average Convergence Divergence (MACD) of the data
def compute_MACD(data, short_window=12, long_window=26, signal_window=9):
    EMA_short = data['High'].ewm(span=short_window).mean()
    EMA_long = data['High'].ewm(span=long_window).mean()
    MACD_line = EMA_short - EMA_long
    signal_line = MACD_line.ewm(span=signal_window).mean()
    histogram = MACD_line - signal_line
    return MACD_line, signal_line, histogram



# Define start and end dates
start_date = "1992-01-02"
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

# Download the historical data for Dow Jones using yfinance
dow_jones = yf.download("^DJI", start=start_date, end=end_date)

# Download the historical data for 13 Week Treasury Bill using yfinance
irx_data = yf.download("^IRX", start=start_date, end=end_date)
# Add 'Open' price data from IRX to dow_jones DataFrame
dow_jones['IRX_Open'] = irx_data['Open']

# Download the historical data for Treasury Yield 5 Years using yfinance
fvx_data = yf.download("^FVX", start=start_date, end=end_date)
# Add 'Open' price data from FVX to dow_jones DataFrame
dow_jones['FVX_Open'] = fvx_data['Open']

# Download the historical data for CBOE Volatility Index using yfinance
vix_data = yf.download("^VIX", start=start_date, end=end_date)
# Add 'Open', 'High', 'Low' and 'Close' price data from VIX to dow_jones DataFrame
dow_jones['VIX_Open'] = vix_data['Open']
dow_jones['VIX_High'] = vix_data['High']
dow_jones['VIX_Low'] = vix_data['Low']
dow_jones['VIX_Close'] = vix_data['Close']

# Continue with the rest of your code...



# Add MACD, signal line, and histogram to the DataFrame
dow_jones['MACD'], dow_jones['Signal'], dow_jones['Histogram'] = compute_MACD(dow_jones)

# Prepare the data for modeling using different-day moving averages
for window in [1, 1, 2, 3, 5, 8, 13, 21]:
    dow_jones[f'MA{window}_High'] = dow_jones['High'].shift(1).rolling(window=window).mean()
    dow_jones[f'MA{window}_Low'] = dow_jones['Low'].shift(1).rolling(window=window).mean()
    dow_jones[f'MA{window}_Close'] = dow_jones['Close'].shift(1).rolling(window=window).mean()
    dow_jones[f'MA{window}_Open'] = dow_jones['Open'].shift(1).rolling(window=window).mean()
    dow_jones[f'MA{window}_Volume'] = dow_jones['Volume'].shift(1).rolling(window=window).mean()

# Add rate of change features
dow_jones['Price_RoC'] = dow_jones['Close'].pct_change()
dow_jones['Volume_RoC'] = dow_jones['Volume'].pct_change()
dow_jones['MACD_RoC'] = dow_jones['MACD'].pct_change()
dow_jones['MA21_RoC'] = dow_jones['MA21_Close'].pct_change()
dow_jones['High_RoC'] = dow_jones['High'].pct_change()
dow_jones['Low_RoC'] = dow_jones['Low'].pct_change()
dow_jones['Open_RoC'] = dow_jones['Open'].pct_change()
dow_jones['VIX_Open_RoC'] = dow_jones['VIX_Open'].pct_change()
dow_jones['VIX_High_RoC'] = dow_jones['VIX_High'].pct_change()
dow_jones['VIX_Low_RoC'] = dow_jones['VIX_Low'].pct_change()
dow_jones['VIX_Close_RoC'] = dow_jones['VIX_Close'].pct_change()

# Apply the filter to 'Open' prices
kf_open = KalmanFilter(dim_x=1, dim_z=1)
kf_open.x = np.array([dow_jones['Open'].values[0]])
kf_open.P *= 1000.
kf_open.R = 5
kf_open.F = np.array([[1.]])
kf_open.H = np.array([[1.]])

state_estimates_open = []
for price in dow_jones['Open'].values[1:]:
    kf_open.predict()
    state_estimates_open.append(kf_open.x[0])
    kf_open.update([price])

# Repeat the process for 'Close' prices
kf_close = KalmanFilter(dim_x=1, dim_z=1)
kf_close.x = np.array([dow_jones['Close'].values[0]])
kf_close.P *= 1000.
kf_close.R = 5
kf_close.F = np.array([[1.]])
kf_close.H = np.array([[1.]])

state_estimates_close = []
for price in dow_jones['Close'].values[1:]:
    kf_close.predict()
    state_estimates_close.append(kf_close.x[0])
    kf_close.update([price])

# Repeat the process for 'Volume' prices
kf_volume = KalmanFilter(dim_x=1, dim_z=1)
kf_volume.x = np.array([dow_jones['Volume'].values[0]])
kf_volume.P *= 1000.
kf_volume.R = 5
kf_volume.F = np.array([[1.]])
kf_volume.H = np.array([[1.]])

state_estimates_volume = []
for price in dow_jones['Volume'].values[1:]:
    kf_volume.predict()
    state_estimates_volume.append(kf_volume.x[0])
    kf_volume.update([price])

# And for 'High' prices
kf_high = KalmanFilter(dim_x=1, dim_z=1)
kf_high.x = np.array([dow_jones['High'].values[0]])
kf_high.P *= 1000.
kf_high.R = 5
kf_high.F = np.array([[1.]])
kf_high.H = np.array([[1.]])

state_estimates_high = []
for price in dow_jones['High'].values[1:]:
    kf_high.predict()
    state_estimates_high.append(kf_high.x[0])
    kf_high.update([price])

# Add the filtered estimates to the dataframe
dow_jones = dow_jones.iloc[1:]  
dow_jones['Kalman_Filter_Open'] = state_estimates_open
dow_jones['Kalman_Filter_Close'] = state_estimates_close
dow_jones['Kalman_Filter_Volume'] = state_estimates_volume
dow_jones['Kalman_Filter_High'] = state_estimates_high


# Drop NA values
dow_jones = dow_jones.dropna()

# Prepare the dataset for training
X = dow_jones.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
y_highs = dow_jones["High"]

# Split the data into training and testing sets
X_train, X_test, y_train_highs, y_test_highs = train_test_split(X, y_highs, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest regressor for highs with scaled data
rf_highs = RandomForestRegressor(n_estimators=99, random_state=42)
rf_highs.fit(X_train_scaled, y_train_highs)

# Train the Linear Regression model for highs with scaled data
lr_highs = LinearRegression()
lr_highs.fit(X_train_scaled, y_train_highs)

# Prepare for GridSearch for the XGBoost model
tscv = TimeSeriesSplit(n_splits=5)
xgb_model = XGBRegressor()
xgb_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
xgb_cv = GridSearchCV(xgb_model, xgb_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_cv.fit(X_train_scaled, y_train_highs)

# Select the best XGBoost model
best_xgb_model = xgb_cv.best_estimator_

# Predict the high price for tomorrow using the trained models
last_MA_values = X.iloc[-1:]  # Assuming the last row of the dataset to predict the high price for the next day
last_MA_values_scaled = scaler.transform(last_MA_values)

prediction_rf_highs = rf_highs.predict(last_MA_values_scaled)
prediction_lr_highs = lr_highs.predict(last_MA_values_scaled)
prediction_xgb_highs = best_xgb_model.predict(last_MA_values_scaled)

# Print the predictions for tomorrow's high price
print("Random Forest prediction for tomorrow's high price: {:.2f}".format(prediction_rf_highs[0]))
print("Linear Regression prediction for tomorrow's high price: {:.2f}".format(prediction_lr_highs[0]))
print("XGBoost prediction for tomorrow's high price: {:.2f}".format(prediction_xgb_highs[0]))


# calculate and print confidence interval for xgbooster 
import numpy as np

num_iterations = 100  # Define number of iterations

def bootstrap_prediction(model, sample_data, num_iterations, scaler):
    predictions = []
    for _ in range(num_iterations):
        sample = sample_data.sample(frac=1, replace=True)
        X_sample = sample.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
        y_sample = sample["High"]  # Change target to 'High'
        X_sample_scaled = scaler.transform(X_sample)  # Scale the X_sample data
        model.fit(X_sample_scaled, y_sample)
        prediction = model.predict(last_MA_values_scaled)
        predictions.append(prediction)
    return predictions

# Get bootstrap predictions
bootstrap_preds_highs = bootstrap_prediction(best_xgb_model, dow_jones, num_iterations, scaler)  # This line will provide predictions for 'High'

# Calculate the 66.8% and 99.5% confidence intervals
lower_66_highs = np.percentile(bootstrap_preds_highs, 16.6)
upper_66_highs = np.percentile(bootstrap_preds_highs, 83.4)
lower_995_highs = np.percentile(bootstrap_preds_highs, 0.25)
upper_995_highs = np.percentile(bootstrap_preds_highs, 99.75)

# Print the confidence intervals
print("XGBoost 66.8% CI for tomorrow's high price: ({:.2f}, {:.2f})".format(lower_66_highs, upper_66_highs))
print("XGBoost 99.5% CI for tomorrow's high price: ({:.2f}, {:.2f})".format(lower_995_highs, upper_995_highs))



import numpy as np
import scipy.stats as stats

# Add this function to calculate the confidence interval
def confidence_interval(prediction, std_error, confidence_level, sample_size):
    t_score = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1)
    margin_of_error = t_score * std_error
    lower_bound = prediction - margin_of_error
    upper_bound = prediction + margin_of_error
    return lower_bound, upper_bound

# Calculate the standard error for the linear regression model
std_error_lr_highs = np.sqrt(np.sum((y_test_highs - lr_highs.predict(X_test_scaled))**2) / (len(X_test) - 2))

# Calculate the residuals for the linear regression model
residuals_lr_highs = y_train_highs - lr_highs.predict(X_train_scaled)

# Calculate the standard error for the linear regression model
std_error_lr_highs = residuals_lr_highs.std()


# Calculate the sample size
sample_size = len(X_train)

# Calculate the confidence intervals
ci_lr_highs_668 = confidence_interval(prediction_lr_highs[0], std_error_lr_highs, 0.668, sample_size)
ci_lr_highs_995 = confidence_interval(prediction_lr_highs[0], std_error_lr_highs, 0.995, sample_size)

# Print the confidence intervals
print("Linear Regression 66.8% CI for tomorrow's high price: ({:.2f}, {:.2f})".format(*ci_lr_highs_668))
print("Linear Regression 99.5% CI for tomorrow's high price: ({:.2f}, {:.2f})".format(*ci_lr_highs_995))



from sklearn.metrics import mean_squared_error, mean_absolute_error

# Make predictions for the test set
y_pred_rf_highs = rf_highs.predict(X_test_scaled)
y_pred_lr_highs = lr_highs.predict(X_test_scaled)
y_pred_xgb_highs = best_xgb_model.predict(X_test_scaled)

# Calculate and print MSE and MAE for each model
mse_rf_highs = mean_squared_error(y_test_highs, y_pred_rf_highs)
mse_lr_highs = mean_squared_error(y_test_highs, y_pred_lr_highs)
mse_xgb_highs = mean_squared_error(y_test_highs, y_pred_xgb_highs)

mae_rf_highs = mean_absolute_error(y_test_highs, y_pred_rf_highs)
mae_lr_highs = mean_absolute_error(y_test_highs, y_pred_lr_highs)
mae_xgb_highs = mean_absolute_error(y_test_highs, y_pred_xgb_highs)

print("\nRandom Forest:")
print(f"MSE: {mse_rf_highs:.2f}")
print(f"MAE: {mae_rf_highs:.2f}")

print("\nLinear Regression:")
print(f"MSE: {mse_lr_highs:.2f}")
print(f"MAE: {mae_lr_highs:.2f}")

print("\nXGBoost:")
print(f"MSE: {mse_xgb_highs:.2f}")
print(f"MAE: {mae_xgb_highs:.2f}")

