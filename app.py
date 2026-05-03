import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
   return joblib.load("model/penguin_model.joblib")


pipeline = load_model()


st.title("🐧 Penguin Predictor")


species = st.selectbox("Species", ["Adelie", "Chinstrap", "Gentoo"])
island  = st.selectbox("Island",  ["Biscoe", "Dream", "Torgersen"])
sex     = st.radio("Sex", ["Male", "Female"], horizontal=True)
bill_length_mm    = st.number_input("Bill length (mm)",    30.0, 60.0, 44.0)
bill_depth_mm     = st.number_input("Bill depth (mm)",     13.0, 22.0, 17.0)
flipper_length_mm = st.number_input("Flipper length (mm)", 170.0, 235.0, 200.0)


if st.button("Predict body mass", type="primary", use_container_width=True):


   # Build a one-row DataFrame from the user inputs.
   # The column names MUST match the names the pipeline was trained on.
   user_input = pd.DataFrame([{
       "species": species,
       "island": island,
       "sex": sex,
       "bill_length_mm": bill_length_mm,
       "bill_depth_mm": bill_depth_mm,
       "flipper_length_mm": flipper_length_mm,
   }])


   # Pass the raw input to the pipeline. The pipeline handles encoding
   # and scaling internally, then runs the model.
   prediction = pipeline.predict(user_input)[0]


   # Show the result in a friendly format
   st.success(f"### Predicted body mass: **{prediction:,.0f} grams**")
   st.caption(f"That's about {prediction / 1000:.2f} kg — roughly the weight of a small bag of rice.")


   # Show what the user typed in (helps with trust)
   with st.expander("See the values used for this prediction"):
       st.dataframe(user_input, hide_index=True)

