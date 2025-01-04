# ML-Model-Builder-app




This project is a user-friendly **Machine Learning Model Builder** developed with **Streamlit**. It allows users to upload datasets, clean data, preprocess features, and build, tune, and test various regression models interactively.

## Features
- Upload datasets in **CSV** or **Excel** format.
- Perform data cleaning (remove missing values and duplicates).
- Select and preprocess features (categorical encoding, feature scaling).
- Choose from multiple regression algorithms:
  - Linear Regression
  - Ridge, Lasso, ElasticNet Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Regression (SVR)
- Hyperparameter tuning for models.
- Evaluate model performance with **R-squared** on training and test data.
- Test the trained model with user input.
- Save and download the trained model for future use.

## Requirements
The required libraries are listed in the `requirements.txt` file. Install them with:
```bash
pip install -r requirements.txt
