import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# === Load Model and Preprocessing Objects ===
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_stage_fear.pkl", "rb") as f:
    label_encoder_stage_fear = pickle.load(f)

with open("label_encoder_drained_after_socializing.pkl", "rb") as f:
    label_encoder_drained = pickle.load(f)

with open("standardscaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Streamlit UI ===
st.set_page_config(page_title="Personality Predictor", page_icon="🌟")
st.title("🧠 Personality Type Predictor")
st.markdown(
    """
Welcome to the Personality Prediction App!
Fill out the information below and discover whether your personality leans more toward an **Introvert** or **Extrovert**.
"""
)

# === Collect Input from User ===
time_spent_alone = st.slider("🕐 Time Spent Alone (hours per day)", 0.0, 24.0, 4.0)
stage_fear = st.selectbox("🎤 Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider(
    "🎉 Social Event Attendance (events per month)", 0.0, 30.0, 2.0
)
going_outside = st.slider("🚶 Going Outside (times per week)", 0.0, 14.0, 3.0)
drained_after_socializing = st.selectbox(
    "😓 Do you feel drained after socializing?", ["Yes", "No"]
)
friends_circle_size = st.slider("👥 Number of Close Friends", 0.0, 50.0, 5.0)
post_frequency = st.slider(
    "📱 Social Media Post Frequency (posts per week)", 0.0, 20.0, 3.0
)

# === Predict Button ===
if st.button("🔍 Predict Personality"):
    input_data = {
        "Time_spent_Alone": time_spent_alone,
        "Stage_fear": stage_fear,
        "Social_event_attendance": social_event_attendance,
        "Going_outside": going_outside,
        "Drained_after_socializing": drained_after_socializing,
        "Friends_circle_size": friends_circle_size,
        "Post_frequency": post_frequency,
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode
    input_df["Stage_fear"] = label_encoder_stage_fear.transform(input_df["Stage_fear"])
    input_df["Drained_after_socializing"] = label_encoder_drained.transform(
        input_df["Drained_after_socializing"]
    )

    # Reorder Columns
    ordered_cols = [
        "Time_spent_Alone",
        "Stage_fear",
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency",
    ]
    input_df = input_df[ordered_cols]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)
    prediction_proba = prediction[0][0]
    personality = "Introvert" if prediction_proba > 0.5 else "Extrovert"

    st.success(f"🧬 Predicted Personality: **{personality}**")
    st.info(f"Model Confidence: {prediction_proba:.2%}")
