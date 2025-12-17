
import joblib
import pandas as pd

def preprocess_input(df_raw, training=False):
    df = df_raw.copy()

    # Load all preprocessors and column lists
    scaler = joblib.load('scaler.joblib')
    num_imputer = joblib.load('num_imputer.joblib')

    # The features.joblib saved in nOkF82Xv2mKV is X.columns.tolist() which are the FINAL OHE features for the UI-safe model.
    final_model_features = joblib.load('features.joblib') # Renaming this for clarity

    # --- Manual definition of numerical columns for the UI-safe model ---
    # These are the columns the num_imputer and scaler were actually fitted on in nOkF82Xv2mKV
    num_cols_ui_safe = ['age_building', 'count_floors_pre_eq']

    # --- Step 1: Align input column names ---
    if 'age' in df.columns:
        df.rename(columns={'age': 'age_building'}, inplace=True)

    # --- Step 2: Extract and preprocess numerical features ---
    # Create a temporary DataFrame for numerical features from the UI input
    # Initialize with 0s and then fill if available in df
    df_numerical_subset = pd.DataFrame(0, index=df.index, columns=num_cols_ui_safe)
    for col in num_cols_ui_safe:
        if col in df.columns:
            df_numerical_subset[col] = df[col]

    # Apply numerical imputation and scaling to this subset
    df_numerical_subset[num_cols_ui_safe] = num_imputer.transform(df_numerical_subset[num_cols_ui_safe])
    df_numerical_subset[num_cols_ui_safe] = scaler.transform(df_numerical_subset[num_cols_ui_safe])

    # --- Step 3: Handle Categorical Features (One-Hot Encoding) ---
    # Identify categorical columns from the raw input that are NOT numerical
    # This assumes `df` contains the raw categorical inputs like 'foundation_type', 'roof_type' etc.
    categorical_raw_cols_from_ui = [col for col in df.columns if col not in num_cols_ui_safe and df[col].dtype == 'object']
    df_processed_categorical = pd.get_dummies(df[categorical_raw_cols_from_ui], drop_first=True)

    # --- Step 4: Combine numerical and categorical features ---
    # Combine the processed numerical and categorical features
    processed_features = pd.concat([df_numerical_subset, df_processed_categorical], axis=1)

    # --- Step 5: Final column alignment with the model's training features ---
    # Create a final DataFrame with all columns from final_model_features, filled with 0 if missing
    final_df = pd.DataFrame(0, index=processed_features.index, columns=final_model_features)
    for col in final_model_features:
        if col in processed_features.columns:
            final_df[col] = processed_features[col]

    return final_df

def predict_damage(input_df):
    model = joblib.load('lgb_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')

    processed = preprocess_input(input_df)
    pred = model.predict(processed)
    prob = model.predict_proba(processed)

    damage_label = label_encoder.inverse_transform(pred)[0]
    confidence = prob.max()

    return damage_label, confidence
