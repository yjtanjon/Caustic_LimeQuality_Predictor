# Caustic_LimeQuality_Predictor

This project aims to analyze and model the caustic process data using machine learning techniques. The dataset contains various parameters related to the caustic process, and the goal is to predict specific chemical properties based on this data.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Data Visualization](#data-visualization)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Model Export](#model-export)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The caustic process involves several chemical reactions, and the dataset contains measurements and parameters related to this process. The main objectives of this project are as follows:

1. **Data Preprocessing**: The dataset is preprocessed to handle missing values and outliers, ensuring that the data is suitable for machine learning.

2. **Data Visualization**: Data visualization is performed to gain insights into the caustic process and its various parameters.

3. **Modeling**: Machine learning models, including LSTM (Long Short-Term Memory), are used to predict specific chemical properties based on the process data.

4. **Evaluation**: The model's performance is evaluated using various metrics, such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R-squared (R2) Score.

5. **Model Export**: The trained model is exported for future predictions.

## Dataset

The dataset used in this project is stored in the `JanJune_CausticData.csv` file. It contains the following columns:

- `timestamp`: Timestamps of data collection.
- Various parameters related to the caustic process, such as temperature, flow rates, and residence times.
- Chemical properties to be predicted, including EA , AA , TTA , and CE .

## Preprocessing

The preprocessing steps performed on the dataset include:

- Handling missing values.
- Applying the Hampel filter to remove outliers.
- Calculating lime flow and residence times.
- Splitting the dataset into Caustic1 and Caustic5 datasets.
- Shifting timestamps based on residence time.

## Data Visualization

Data visualization is an essential step to understand the caustic process. The following visualizations are included:

- Line plots for chemical properties (EA, AA, TTA, CE) for both Caustic1 and Caustic5 datasets.

## Modeling

A Long Short-Term Memory (LSTM) neural network model is used for predicting chemical properties. The model architecture includes LSTM layers followed by a dense output layer.

## Evaluation

The model is evaluated using the following metrics:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R2) Score

These metrics assess the model's accuracy in predicting chemical properties.

## Model Export

The trained LSTM model is exported and saved as `lime_screw_model.pkl. Additionally, feature scaling parameters are saved to `input_scaler_data.txt` and 'output_scaler_data.txt'.

## Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/caustic-process-analysis.git



