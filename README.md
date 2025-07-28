# Landslide Fatality Prediction Model

This project implements a machine learning pipeline to predict the likelihood of fatalities in landslide events. It uses the Global Landslide Catalog dataset and integrates real-time geospatial and weather data to make predictions for specific locations and dates. The model is built using an XGBoost classifier and includes feature importance analysis with SHAP.

## 🌟 Features

* **Data-Driven Prediction:** Utilizes the NASA Global Landslide Catalog to train a robust prediction model.
* **Comprehensive Feature Engineering:** Incorporates a wide range of features, including:
    * **Geospatial:** Latitude and longitude.
    * **Temporal:** Month and season extracted from event dates.
    * **Weather-Related:** Identifies heavy rain as a key trigger.
    * **Land Use:** Flags events occurring in mining areas.
    * **Historical Patterns:** Uses the reported size of previous landslides.
    * **Human Impact:** Considers the population of the administrative division.
* **Real-Time Data Integration:** Includes a prediction function that fetches live data for:
    * **Geocoding:** Converts location names (country, city) into coordinates using `geopy`.
    * **Precipitation:** Retrieves rainfall data from the NOAA API.
    * **Terrain Analysis:** Gathers elevation and slope data from the Open-Elevation API.
* **Advanced Modeling:** Employs an XGBoost Classifier within a scikit-learn pipeline for robust and scalable model training.
* **Hyperparameter Tuning:** Uses `GridSearchCV` to find the optimal parameters for the XGBoost model, maximizing its predictive performance.
* **Model Interpretability:** Leverages **SHAP (SHapley Additive exPlanations)** to provide insights into the model's predictions and understand the importance of each feature.

## ⚙️ Methodology

1.  **Data Preprocessing:** The raw data is cleaned, and temporal features like `month` and `season` are extracted from the `event_date`. A binary target variable, `fatality_occurred`, is created.
2.  **Feature Engineering:** New features such as `heavy_rain` and `mining_area` are engineered from existing columns to improve model accuracy.
3.  **Pipeline Construction:** A scikit-learn `Pipeline` is used to streamline the workflow. It includes:
    * `StandardScaler` for numeric features.
    * `OneHotEncoder` for categorical features.
4.  **Model Training:** An XGBoost Classifier is trained on the preprocessed data. The model is optimized for handling class imbalance by using `scale_pos_weight`.
5.  **Hyperparameter Tuning:** `GridSearchCV` is employed to systematically search for the best combination of hyperparameters, using F1-score as the evaluation metric.
6.  **Evaluation & Interpretation:** The model's performance is evaluated using a classification report and confusion matrix. SHAP is used to visualize feature importance and explain the model's decision-making process.
