import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

# --- 1. Data Loading ---
# Load traffic data from CSV
print("--- Loading Traffic Data from traffic.csv ---")
try:
    df = pd.read_csv('traffic.csv')
    print("Traffic.csv loaded successfully!")
    print("Data Head:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
except Exception as e:
    print(f"Error loading traffic.csv: {e}")
    print("Ensure 'traffic.csv' is in the same directory.")
    exit()

# --- 2. Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Convert DateTime to datetime objects and engineer features
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek # Monday=0, Sunday=6
df['Month'] = df['DateTime'].dt.month
df['Year'] = df['DateTime'].dt.year
df['Day'] = df['DateTime'].dt.day

# Drop original columns not used as features
df = df.drop(['DateTime', 'ID'], axis=1)

# Check for missing values (uncomment if you want to see this in output)
print("Missing values before handling:")
print(df.isnull().sum())

# Define features (X) and target (y)
X = df.drop('Vehicles', axis=1)
y = df['Vehicles']

# Scale numerical features
numerical_features = ['Junction', 'Hour', 'DayOfWeek', 'Month', 'Year', 'Day']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\nFeatures after scaling (first 5 rows of numerical features):")
print(X[numerical_features].head())

# --- 3. Splitting Data into Training and Testing Sets ---
print("\n--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 4. Model Training ---
print("\n--- Model Training ---")

# Train Linear Regression model
print("\nTraining Linear Regression Model...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print("Linear Regression Model Trained.")

# Train Random Forest Regressor model
print("\nTraining Random Forest Regressor Model...")
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
random_forest_model.fit(X_train, y_train)
print("Random Forest Regressor Model Trained.")

# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")

# Function to evaluate model performance and generate plot
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {model_name} Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title(f"{model_name}: Actual vs. Predicted")
    plt.grid(True)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Display the plot
    plt.show() 
    plt.close() # Close the plot to free memory

    return buf.getvalue()

# Evaluate both trained models
lr_plot_data = evaluate_model(linear_model, X_test, y_test, "Linear Regression")
rf_plot_data = evaluate_model(random_forest_model, X_test, y_test, "Random Forest Regressor")

# --- 6. Making Predictions (Example) ---
print("\n--- Making Predictions (Example) ---")

# Example prediction for a new data point
new_data_point_raw = pd.DataFrame({
    'Junction': [1],
    'Hour': [17],
    'DayOfWeek': [4],
    'Month': [7],
    'Year': [2017],
    'Day': [15]
})

# Scale the new data point using the trained scaler
new_data_point_scaled = new_data_point_raw.copy()
new_data_point_scaled[numerical_features] = scaler.transform(new_data_point_raw[numerical_features])

# Make predictions
lr_prediction = linear_model.predict(new_data_point_scaled)
print(f"\nPredicted Traffic Volume (Linear Regression) for new data: {int(lr_prediction[0])}")

rf_prediction = random_forest_model.predict(new_data_point_scaled)
print(f"Predicted Traffic Volume (Random Forest) for new data: {int(rf_prediction[0])}")

print("\nProject execution complete.")
