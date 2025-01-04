import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# Title
st.title("Regression Model Builder with Data Cleaning, Hyperparameter Tuning, and Testing")

# Upload file
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Determine file type
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Dataset Preview:", df.head())

    # Data Cleaning
    st.subheader("Data Cleaning")
    if st.checkbox("Remove Missing Values"):
        df.dropna(inplace=True)
    
    if st.checkbox("Drop Duplicates"):
        df.drop_duplicates(inplace=True)

    st.write("Cleaned Dataset Preview:", df.head())

    # Select target variable
    target_column = st.selectbox("Select Target Column", df.columns)

    # Split data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encoding categorical features with OneHotEncoder
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns

    if len(categorical_columns) > 0:
        st.write("Encoding categorical columns:", list(categorical_columns))
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        transformer = ColumnTransformer(
            transformers=[('cat', encoder, categorical_columns)],
            remainder='passthrough'
        )

        # Apply transformer to the dataset
        X = transformer.fit_transform(X)

        # Convert to DataFrame (optional)
        X = pd.DataFrame(X, columns=transformer.get_feature_names_out())

    # Feature Scaling
    st.subheader("Feature Scaling")
    scaling_option = st.selectbox("Choose Feature Scaling Method", ["None", "Standard Scaling", "Min-Max Scaling"])

    scaler = None

    if scaling_option == "Standard Scaling":
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif scaling_option == "Min-Max Scaling":
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    st.write("Preprocessed Features (Numeric):", X.head())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select Regression Algorithm
    st.subheader("Select Regression Algorithm and Tune Hyperparameters")
    algorithm = st.selectbox(
        "Choose Algorithm",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "ElasticNet Regression",
            "K-Nearest Neighbors (KNN)",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "SVR",
        ]
    )

    # Initialize the selected model with hyperparameter tuning
    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Ridge Regression":
        alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
        model = Ridge(alpha=alpha)
    elif algorithm == "Lasso Regression":
        alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
        model = Lasso(alpha=alpha)
    elif algorithm == "ElasticNet Regression":
        alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elif algorithm == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.slider("Number of Neighbors", 1, 50, 5)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif algorithm == "Decision Tree":
        max_depth = st.slider("Maximum Depth", 1, 50, 10)
        min_samples_split = st.slider("Minimum Samples Split", 2, 20, 2)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    elif algorithm == "Random Forest":
        n_estimators = st.slider("Number of Estimators", 10, 200, 100)
        max_depth = st.slider("Maximum Depth", 1, 50, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif algorithm == "Gradient Boosting":
        n_estimators = st.slider("Number of Estimators", 10, 200, 100)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
        max_depth = st.slider("Maximum Depth", 1, 50, 3)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    elif algorithm == "XGBoost":
        n_estimators = st.slider("Number of Estimators", 10, 200, 100)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
        max_depth = st.slider("Maximum Depth", 1, 50, 3)
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, objective='reg:squarederror')
    else:  # SVR
        C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = SVR(C=C, kernel=kernel)

    # Train the model
    st.subheader(f"Training {algorithm} Model")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred) * 100
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred) * 100

    st.write(f"Training Accuracy (R-squared): {train_r2:.2f}%")
    st.write(f"Test Accuracy (R-squared): {r2:.2f}%")

    # Test the model with user data
    st.subheader("Test Model with New Data")

    # Collect user inputs
    user_data = {}
    for col in df.drop(columns=[target_column]).columns:
        if col in categorical_columns:
            user_data[col] = st.selectbox(f"Enter value for {col}", df[col].unique())
        else:
            user_data[col] = st.text_input(f"Enter value for {col} (numeric)")

    # Ensure all inputs are provided before prediction
    if st.button("Test with User Input"):
        try:
            # Create a DataFrame from user input
            user_data_df = pd.DataFrame([user_data])

            # Apply encoding to categorical features
            if len(categorical_columns) > 0:
                user_data_df = transformer.transform(user_data_df)

            # Apply scaling to numeric features
            if scaler is not None:
                user_data_df = scaler.transform(user_data_df)

            # Predict with the trained model
            prediction = model.predict(user_data_df)
            st.success(f"Predicted Value: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error in processing input or prediction: {e}")

    # Save and Download Model and Preprocessor
    st.subheader("Save and Download Model and Preprocessor")
    if st.button("Save Model and Preprocessor"):
        try:
            joblib.dump(model, "model.pkl")
            if transformer:
                joblib.dump(transformer, "transformer.pkl")
            if scaler:
                joblib.dump(scaler, "scaler.pkl")
            st.success("Model and preprocessor saved successfully!")
            with open("model.pkl", "rb") as f:
                st.download_button("Download Model", f, file_name="model.pkl")
            if transformer:
                with open("transformer.pkl", "rb") as f:
                    st.download_button("Download Transformer", f, file_name="transformer.pkl")
            if scaler:
                with open("scaler.pkl", "rb") as f:
                    st.download_button("Download Scaler", f, file_name="scaler.pkl")
        except Exception as e:
            st.error(f"Error in saving or downloading: {e}")
