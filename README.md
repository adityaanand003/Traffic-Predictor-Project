# Traffic-Predictor-Project

This project builds a machine learning model to predict traffic volume using historical data. It covers data handling, model training, and evaluation with Python.

## Project Overview

The project involves:

* **Data Loading & Preprocessing**: Reading `traffic.csv`, extracting time-based features (`Hour`, `DayOfWeek`, `Month`, `Year`, `Day`), and scaling data.

* **Model Training**: Training **Linear Regression** and **Random Forest Regressor** models.

* **Model Evaluation**: Assessing model performance using MAE, MSE, RMSE, and $R^2$.

* **Prediction**: Demonstrating traffic volume prediction for new data.

## Data

Uses `traffic.csv` with `DateTime`, `Junction`, `Vehicles`, and `ID` columns.

## Setup and Installation 🛠️

1.  **Get Files**: Clone the repository or download `traffic_predictor.py` and `traffic.csv` into a folder (e.g., `TrafficPredictionProject`).

    ```
    git clone https://github.com/adityaanand003/Traffic-Predictor-Project
    cd Traffic-Prediction-Project


    ```

2.  **Install Python**: Ensure Python 3.7+ is installed from [python.org](https://www.python.org/downloads/).

3.  **Install Libraries**: In your project directory, run:

    ```
    pip install pandas numpy scikit-learn matplotlib


    ```

## How to Run the Project 🚀

1.  **Open Terminal**: Navigate to your project folder.

    ```
    cd C:\Users\YourUsername\Desktop\TrafficPredictionProject


    ```

2.  **Execute Script**: Run the Python file:

    ```
    python traffic_predictor.py


    ```

    The script will print outputs and display plots locally.
