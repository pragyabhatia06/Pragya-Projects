import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Symptom Risk Classification Demo",
    page_icon="🩺",
    layout="centered",
)

MODEL_FILENAME = "Linear_Regression_Model_MP.pkl"


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "model" / MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def build_feature_row(
    hiv_infection: str,
    rectal_pain: str,
    sexually_transmitted_infection: str,
    systemic_illness: str,
    penile_oedema: str,
    sore_throat: str,
    solitary_lesion: str,
    swollen_tonsils: str,
) -> pd.DataFrame:
    yes_no_map = {"No": 0, "Yes": 1}
    illness_map = {
        "None": 0,
        "Fever": 1,
        "Muscle Aches and Pain": 2,
        "Swollen Lymph Nodes": 3,
    }

    # Use EXACT feature names expected by model
    row = {
        "HIV Infection": yes_no_map[hiv_infection],
        "Rectal Pain": yes_no_map[rectal_pain],
        "Sexually Transmitted Infection": yes_no_map[sexually_transmitted_infection],
        "Encoded Systemic Illness": illness_map[systemic_illness],
        "Penile Oedema": yes_no_map[penile_oedema],
        "Sore Throat": yes_no_map[sore_throat],
        "Solitary Lesion": yes_no_map[solitary_lesion],
        "Swollen Tonsils": yes_no_map[swollen_tonsils],
    }

    # Keep exact order too
    feature_order = [
        "HIV Infection",
        "Rectal Pain",
        "Sexually Transmitted Infection",
        "Encoded Systemic Illness",
        "Penile Oedema",
        "Sore Throat",
        "Solitary Lesion",
        "Swollen Tonsils",
    ]

    return pd.DataFrame([row])[feature_order]


def predict_risk(features_df: pd.DataFrame):
    model = load_model()
    prediction = model.predict(features_df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features_df)[0][1])

    return prediction, probability


st.title("🩺 Symptom Risk Classification Demo")
st.caption("Demo ML app for symptom-based classification. This is not medical advice or diagnosis.")

st.write("Select the symptom information below and run the model prediction.")

systemic_illness = st.selectbox(
    "Systemic Illness",
    ["None", "Fever", "Swollen Lymph Nodes", "Muscle Aches and Pain"],
)

col1, col2, col3 = st.columns(3)
with col1:
    sore_throat = st.selectbox("Sore Throat", ["No", "Yes"])
with col2:
    swollen_tonsils = st.selectbox("Swollen Tonsils", ["No", "Yes"])
with col3:
    hiv_infection = st.selectbox("HIV Infection", ["No", "Yes"])

col4, col5 = st.columns(2)
with col4:
    rectal_pain = st.selectbox("Rectal Pain", ["No", "Yes"])
with col5:
    sexually_transmitted_infection = st.selectbox("Sexually Transmitted Infection", ["No", "Yes"])

col6, col7 = st.columns(2)
with col6:
    penile_oedema = st.selectbox("Penile Oedema", ["No", "Yes"])
with col7:
    solitary_lesion = st.selectbox("Solitary Lesion", ["No", "Yes"])

if st.button("Run Prediction", use_container_width=True):
    try:
        input_df = build_feature_row(
            hiv_infection=hiv_infection,
            rectal_pain=rectal_pain,
            sexually_transmitted_infection=sexually_transmitted_infection,
            systemic_illness=systemic_illness,
            penile_oedema=penile_oedema,
            sore_throat=sore_throat,
            solitary_lesion=solitary_lesion,
            swollen_tonsils=swollen_tonsils,
        )

        prediction, probability = predict_risk(input_df)

        st.subheader("Prediction Result")
        if int(prediction) == 1:
            st.error("Model Output: Positive Class")
        else:
            st.success("Model Output: Negative Class")

        if probability is not None:
            st.metric("Predicted Positive Probability", f"{probability:.2%}")

        with st.expander("Input Features Sent to Model"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")