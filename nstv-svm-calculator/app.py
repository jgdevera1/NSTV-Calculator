from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(
    page_title="NTSV Adverse Perinatal Outcome Calculator",
    page_icon="🩺",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Cesarean delivery": APP_DIR / "svm_cesarean_delivery.joblib",
    "Maternal morbidity": APP_DIR / "svm_maternal_morbidity.joblib",
    "Neonatal morbidity": APP_DIR / "svm_neonatal_morbidity.joblib",
}

FALLBACK_PREDICTORS = [
    "age",
    "bmi",
    "gestational.age",
    "ob.cmi",
    "bishop",
    "membrane",
    "spontaneous.labor",
    "induction.labor",
    "augmentation.labor",
]

GESTATIONAL_AGE_OPTIONS = {
    "37–37 6/7 weeks": 0,
    "38–38 6/7 weeks": 1,
    "39–39 6/7 weeks": 2,
    "40–40 6/7 weeks": 3,
    "41–41 6/7 weeks": 4,
}

OB_CMI_OPTIONS = {
    "0–3 (low risk)": 0,
    "≥4 (high risk)": 1,
}

MEMBRANE_OPTIONS = {
    "Intact": 0,
    "Spontaneous rupture": 1,
}

YES_NO_OPTIONS = {
    "No": 0,
    "Yes": 1,
}


@st.cache_resource
def load_three_models():
    missing_files = [
        path.name
        for path in MODEL_FILES.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing_files)
        )

    loaded = {}

    for expected_outcome, path in MODEL_FILES.items():
        saved_object = load(path)

        if isinstance(saved_object, dict):
            if "model" not in saved_object:
                raise ValueError(
                    f"{path.name} is a dictionary but has no 'model' entry."
                )

            payload = dict(saved_object)
            payload.setdefault("outcome", expected_outcome)
            payload.setdefault("features", FALLBACK_PREDICTORS)
            payload.setdefault("threshold", 0.50)
            payload.setdefault("calibration_method", "Not recorded")
        else:
            payload = {
                "model": saved_object,
                "outcome": expected_outcome,
                "features": FALLBACK_PREDICTORS,
                "threshold": 0.50,
                "calibration_method": "Not recorded",
            }

        loaded[expected_outcome] = payload

    return loaded


def determine_predictor_order(model_payloads):
    feature_lists = [
        list(payload.get("features", []))
        for payload in model_payloads.values()
        if payload.get("features")
    ]

    if not feature_lists:
        return FALLBACK_PREDICTORS

    reference = feature_lists[0]

    for current in feature_lists[1:]:
        if current != reference:
            raise ValueError(
                "The three model files do not contain the same predictor "
                "names and order."
            )

    return reference


def create_patient_frame(predictor_order):
    st.subheader("Patient inputs")

    age = st.number_input(
        "Maternal age at admission (years)",
        min_value=18.0,
        max_value=60.0,
        value=30.0,
        step=1.0,
    )

    bmi = st.number_input(
        "Pre-pregnancy BMI (kg/m²)",
        min_value=10.0,
        max_value=70.0,
        value=24.0,
        step=0.1,
    )

    gestational_age = GESTATIONAL_AGE_OPTIONS[
        st.selectbox(
            "Gestational age category",
            list(GESTATIONAL_AGE_OPTIONS.keys()),
            index=2,
        )
    ]

    ob_cmi = OB_CMI_OPTIONS[
        st.selectbox(
            "OB-CMI risk category",
            list(OB_CMI_OPTIONS.keys()),
            index=0,
        )
    ]

    bishop = st.number_input(
        "Bishop score on admission",
        min_value=0.0,
        max_value=13.0,
        value=5.0,
        step=1.0,
    )

    membrane = MEMBRANE_OPTIONS[
        st.selectbox(
            "Membrane status on admission",
            list(MEMBRANE_OPTIONS.keys()),
            index=0,
        )
    ]

    spontaneous_labor = YES_NO_OPTIONS[
        st.selectbox(
            "Spontaneous labor",
            list(YES_NO_OPTIONS.keys()),
            index=1,
        )
    ]

    induction_labor = YES_NO_OPTIONS[
        st.selectbox(
            "Induction of labor",
            list(YES_NO_OPTIONS.keys()),
            index=0,
        )
    ]

    augmentation_labor = YES_NO_OPTIONS[
        st.selectbox(
            "Augmentation of labor",
            list(YES_NO_OPTIONS.keys()),
            index=0,
        )
    ]

    encoded = {
        "age": float(age),
        "bmi": float(bmi),
        "gestational.age": int(gestational_age),
        "ob.cmi": int(ob_cmi),
        "bishop": float(bishop),
        "membrane": int(membrane),
        "spontaneous.labor": int(spontaneous_labor),
        "induction.labor": int(induction_labor),
        "augmentation.labor": int(augmentation_labor),
    }

    for predictor in predictor_order:
        if predictor not in encoded:
            encoded[predictor] = np.nan

    return pd.DataFrame(
        [{name: encoded[name] for name in predictor_order}]
    )


def calculate_predictions(model_payloads, patient_frame):
    output = []
    probabilities = []

    for expected_outcome, payload in model_payloads.items():
        model = payload["model"]
        probability = float(
            model.predict_proba(patient_frame)[:, 1][0]
        )
        threshold = float(payload.get("threshold", 0.50))
        probabilities.append(probability)

        output.append({
            "Outcome": expected_outcome,
            "Predicted probability": probability,
            "Classification threshold": threshold,
            "Above selected threshold": probability >= threshold,
            "Calibration method": payload.get(
                "calibration_method",
                "Not recorded",
            ),
        })

    composite_score = float(np.mean(probabilities))

    return pd.DataFrame(output), composite_score


st.title("NTSV Adverse Perinatal Outcome Calculator")
st.caption(
    "Proof-of-concept calculator using three separately saved, "
    "calibrated support vector machine models."
)

try:
    model_payloads = load_three_models()
    predictor_order = determine_predictor_order(model_payloads)
except Exception as error:
    st.error(
        "The application could not load all three models. Place the "
        "following files in the same folder as app.py:"
    )
    st.code(
        "svm_cesarean_delivery.joblib\n"
        "svm_maternal_morbidity.joblib\n"
        "svm_neonatal_morbidity.joblib"
    )
    st.exception(error)
    st.stop()

with st.expander("Loaded model information"):
    model_information = pd.DataFrame([
        {
            "Outcome": outcome,
            "File": MODEL_FILES[outcome].name,
            "Threshold": payload.get("threshold", 0.50),
            "Calibration": payload.get(
                "calibration_method",
                "Not recorded",
            ),
            "Predictors": ", ".join(payload.get("features", [])),
        }
        for outcome, payload in model_payloads.items()
    ])
    st.dataframe(
        model_information,
        hide_index=True,
        use_container_width=True,
    )

patient_frame = create_patient_frame(predictor_order)

with st.expander("Encoded input sent to the models"):
    st.dataframe(
        patient_frame,
        hide_index=True,
        use_container_width=True,
    )

if st.button("Calculate predicted risks", type="primary"):
    try:
        results, composite_score = calculate_predictions(
            model_payloads,
            patient_frame,
        )
    except Exception as error:
        st.error(
            "Prediction failed. Confirm that the app is using the same "
            "scikit-learn version used when the models were saved."
        )
        st.exception(error)
        st.stop()

    st.subheader("Predicted outcomes")

    display_results = results.copy()
    display_results["Predicted probability"] = (
        display_results["Predicted probability"]
        .map(lambda value: f"{value:.1%}")
    )
    display_results["Classification threshold"] = (
        display_results["Classification threshold"]
        .map(lambda value: f"{value:.1%}")
    )
    display_results["Above selected threshold"] = (
        display_results["Above selected threshold"]
        .map({True: "Yes", False: "No"})
    )

    st.dataframe(
        display_results,
        hide_index=True,
        use_container_width=True,
    )

    st.metric(
        "Composite perinatal risk score",
        f"{composite_score:.1%}",
    )

    st.caption(
        "The composite score is the arithmetic mean of the three "
        "calibrated predicted probabilities. This proof of concept is "
        "for research and validation only and must not be used as the "
        "sole basis for clinical decisions."
    )

    export = results.copy()
    export.loc[len(export)] = {
        "Outcome": "Composite perinatal risk score",
        "Predicted probability": composite_score,
        "Classification threshold": np.nan,
        "Above selected threshold": np.nan,
        "Calibration method": (
            "Arithmetic mean of calibrated probabilities"
        ),
    }

    st.download_button(
        "Download result as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="ntsv_svm_risk_calculation.csv",
        mime="text/csv",
    )
