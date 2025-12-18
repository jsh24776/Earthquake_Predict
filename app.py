
import streamlit as st
import pandas as pd
from prediction import predict_damage

st.set_page_config(
    page_title="Earthquake Damage Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Constants ----
MODEL_ACCURACY = 0.76  # 76%

def confidence_to_certainty(conf):
    if conf < 0.35:
        return "Low"
    elif conf < 0.50:
        return "Moderate"
    elif conf < 0.65:
        return "High"
    else:
        return "Very High"

# ---- Sidebar ----
st.sidebar.header("Input Parameters")
st.sidebar.write("Provide the building details for prediction.")

with st.sidebar.form("input_form"):
    st.subheader("Structural Features")
    age = st.number_input("Building Age (years)", 0, 200, 20)
    floors = st.number_input("Number of Floors", 1, 10, 2)

    st.subheader("Construction Details")
    foundation = st.selectbox("Foundation Type", ['mud', 'cement', 'other'])
    roof = st.selectbox("Roof Type", ['bamboo', 'metal', 'concrete', 'other'])
    ground = st.selectbox("Ground Floor Type", ['mud', 'cement', 'other'])

    st.subheader("Position & Land")
    position = st.selectbox("Building Position", ['attached', 'not_attached'])
    land_surface = st.selectbox("Land Surface Condition", ['flat', 'slope', 'other'])

    submit = st.form_submit_button("Predict")

# ---- Main Panel ----
st.title("Earthquake Building Damage Prediction")
st.write("Predict potential building damage based on structural characteristics.")

if submit:
    input_df = pd.DataFrame([{
        'age': age,
        'count_floors_pre_eq': floors,
        'foundation_type': foundation,
        'roof_type': roof,
        'ground_floor_type': ground,
        'position': position,
        'land_surface_condition': land_surface
    }])

    label, confidence = predict_damage(input_df)
    certainty = confidence_to_certainty(confidence)

    st.markdown("---")
    st.header("Prediction Result")

    # Display main prediction
    st.success(f"Predicted Damage Level: **{label.upper()}**")

    # Additional information in columns for clarity
    col1, col2 = st.columns(2)
    col1.write(f"**Model Certainty:** {certainty}")
    col2.write(f"**Overall Model Accuracy:** {int(MODEL_ACCURACY * 100)}%")

    # Optional technical details
    with st.expander("See Technical Details"):
        st.write(f"Raw prediction confidence: **{confidence:.2%}**")
        st.caption(
            "Note: In multi-class damage prediction, moderate confidence is expected due to overlapping damage categories."
        )

# ---- Footer ----
st.markdown("---")
st.caption("All predictions are based on model estimation. Use as a guide, not a definitive assessment.")
