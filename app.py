
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from pathlib import Path
import gzip
from sklearn.preprocessing import StandardScaler,OneHotEncoder, MinMaxScaler

# === PAGE SETUP ===
st.set_page_config(layout="wide")

# === BACKGROUND STYLE ===
page_bg_img = f'''
<style>
.stApp {{
background-image: url("https://your-image-url-here.com");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# === BUTTONS NAVIGATION ===
st.markdown("""
    <div style='text-align: center;'>
        <a href="?page=overview" style="margin: 0 10px;">
            <button style='padding:15px 30px; font-size:20px; border-radius:10px;'>Overview</button>
        </a>
        <a href="?page=evaluation" style="margin: 0 10px;">
            <button style='padding:15px 30px; font-size:20px; border-radius:10px;'>Evaluation</button>
        </a>
        <a href="?page=calculator" style="margin: 0 10px;">
            <button style='padding:15px 30px; font-size:20px; border-radius:10px;'>Calculator</button>
        </a>
    </div>
""", unsafe_allow_html=True)



# === GET PAGE FROM URL ===
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["overview"])[0]


# === PAGE CONTENT ===
if current_page == "overview":
    st.markdown("# 📚 Overview")
    st.markdown("""
### Overview

For training our model, we worked with a dataset containing detailed information about Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) registered with the Washington State Department of Licensing (DOL).Here is a description of the columns af this dataset :



| Feature | Description |
|:--------|:------------|
| VIN (1-10) | The first 10 characters of each vehicle’s Vehicle Identification Number (VIN). |
| County | The county where the registered owner resides. |
| City | The city where the registered owner resides. |
| State | The state of residence for the registered owner. |
| ZIP Code | The 5-digit postal code for the owner’s residence. |
| Model Year | Determined by decoding the vehicle's VIN, indicating the year of manufacture. |
| Make | The vehicle manufacturer, decoded from the VIN. |
| Model | The specific model of the vehicle, also derived from the VIN. |
| Electric Vehicle Type | Classification as either a fully electric or a plug-in hybrid vehicle. |
| Clean Alternative Fuel Vehicle (CAFV) Eligibility | An indicator showing whether the vehicle qualifies as a Clean Alternative Fuel Vehicle, based on its fuel type and electric-only range. |
| Electric Range | The distance a vehicle can travel purely on its electric battery. |
| Base MSRP | The lowest Manufacturer’s Suggested Retail Price for any version of the vehicle model. |
| Legislative District | The political district within Washington State where the registered owner lives. |
| DOL Vehicle ID | A unique identifier assigned to each vehicle by the Department of Licensing. |
| Vehicle Location | The geographic center of the registered vehicle’s ZIP code. |
| Electric Utility | The power service provider for the registered address. |
| Expected Price | The estimated resale value of the vehicle, which we aim to predict. |
    """)

elif current_page == "evaluation":
    st.markdown("# 📊 Evaluation")
    st.markdown("""
    🔍 Model Evaluation Overview
We trained and fine-tuned a machine learning model called Support Vector Regression (SVR) to predict the expected resale price of a car based on features such as the make, model, fuel type, battery capacity, and more.
To find the best version of this model, we tested different settings (called hyperparameters) and compared their performance. Below is a summary of the results from our tuning process:
| SVR Settings         | R² Score | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) |
|:---------------------|:--------:|:------------------------:|:------------------------------:|
| C=7000, ε=2.3         | 0.9855   | 7.19                     | 2.68                           |
| C=14000, ε=2.3        | 0.9870   | 6.43                     | 2.54                           |
| ✅ C=28000, ε=2.3     | 0.9882   | 5.87                     | 2.42                           |





✅ Final Model Performance
After tuning, the best performing model had the following settings:
•	C = 28000
•	Epsilon (ε) = 2.3
•	Gamma = 'scale' (default scaling method for SVR)
With these parameters, the model achieved:
•	R² Score: 0.9882
•	RMSE: 2.42
________________________________________
📈 What does this mean?
•	R² Score (0.9882): This is a measure of how well the model's predictions match the actual prices. A perfect score would be 1.0, and anything above 0.95 is considered excellent. In our case, the model explains more than 98% of the variation in car prices, which indicates a very accurate prediction capability.
•	RMSE (2.42): This tells us the average error in price predictions. On average, the model's predictions are within about ±2.42 units of the actual price (2420$). A lower RMSE means better accuracy.
________________________________________
💡 TL;DR – Summary for Everyone
Our model is very accurate at predicting the resale price of a car based on its characteristics. It has been fine-tuned and tested for performance, and the final version predicts car prices with over 98% accuracy, and an average error of only 2.42.
Whether you're buying or selling, this tool gives you trusted price estimates based on real data and advanced modeling.

    """)

elif current_page == "calculator":
    st.markdown("# 🧮 Calculator")

    # === Load Data and Model ===
    df = pd.read_csv('Electric_cars_dataset.csv')
    
     
    
    
    
    model = joblib.load('svr_model.joblib')


    df.columns = df.columns.str.replace(' ', '_')

    drop_cols = ['ID', 'State', 'VIN_(1-10)', 'ZIP_Code', 'DOL_Vehicle_ID', 'Vehicle_Location', 'Base_MSRP']
    freq_cols = ['County', 'Electric_Utility', 'Legislative_District', 'City']
    onehot_cols = ['Make', 'Model', 'Electric_Vehicle_Type', 'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility']
    scale_cols = ['Model_Year', 'Electric_Range']

    freq_uniques = {col: df[col].dropna().unique().tolist() for col in freq_cols}
    onehot_uniques = {col: df[col].dropna().unique().tolist() for col in onehot_cols}
    scale_uniques = {col: df[col].dropna().unique().tolist() for col in scale_cols}

    # === User Input Form ===
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://www.neodrift.in/cdn/shop/articles/best-resale-cars-featured.jpg?v=1722222661", use_column_width=True)

    with col2:
        st.subheader("Vehicle Information")

        make = st.selectbox("🚘 Make", onehot_uniques['Make'])
        model_car = st.selectbox("📦 Model", onehot_uniques['Model'])
        model_year = st.slider("📅 Model Year", int(min(scale_uniques['Model_Year'])), int(max(scale_uniques['Model_Year'])))
        ev_type = st.selectbox("⚡ Electric Vehicle Type", onehot_uniques['Electric_Vehicle_Type'])
        cafv = st.selectbox("♻️ CAFV Eligibility", onehot_uniques['Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility'])
        electric_range = st.slider("🔋 Electric Range (miles)", int(min(scale_uniques['Electric_Range'])), int(max(scale_uniques['Electric_Range'])))
        county = st.selectbox("🏙️ County", freq_uniques['County'])
        utility = st.selectbox("🏢 Electric Utility", freq_uniques['Electric_Utility'])
        district = st.selectbox("🏛️ Legislative District", freq_uniques['Legislative_District'])
        city = st.selectbox("📍 City", freq_uniques['City'])

        # Create raw input DataFrame
        input_df = pd.DataFrame({
            'Make': [make], 'Model': [model_car], 'Model_Year': [model_year],
            'Electric_Vehicle_Type': [ev_type],
            'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility': [cafv],
            'Electric_Range': [electric_range], 'County': [county],
            'Electric_Utility': [utility], 'Legislative_District': [district], 'City': [city]
        })

        # === Frequency Encoding ===
        for col in freq_cols:
            mapping = df[col].value_counts().to_dict()
            input_df[col + '_freq'] = input_df[col].map(mapping).fillna(0)

        freq_scaled = MinMaxScaler()
        input_df[[col + '_freq' for col in freq_cols]] = freq_scaled.fit_transform(input_df[[col + '_freq' for col in freq_cols]])
        input_df.drop(columns=freq_cols, inplace=True)

        # === One-Hot Encoding ===
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit(df[onehot_cols]).transform(input_df[onehot_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(onehot_cols))

        input_df = input_df.drop(columns=onehot_cols).reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        input_df = pd.concat([input_df, encoded_df], axis=1)

        # === Scaling ===
        scaler = StandardScaler()
        input_df[scale_cols] = scaler.fit(df[scale_cols]).transform(input_df[scale_cols])

        # === Reorder Columns ===
        correct_column_order = ['Model_Year','Electric_Range','County_freq','Electric_Utility_freq','Legislative_District_freq','City_freq','Make_AUDI','Make_AZURE DYNAMICS',
    'Make_BENTLEY',
    'Make_BMW',
    'Make_CADILLAC',
    'Make_CHEVROLET',
    'Make_CHRYSLER',
    'Make_DODGE',
    'Make_FIAT',
    'Make_FISKER',
    'Make_FORD',
    'Make_HONDA',
    'Make_HYUNDAI',
    'Make_JAGUAR',
    'Make_JEEP',
    'Make_KIA',
    'Make_LAND ROVER',
    'Make_LINCOLN',
    'Make_MERCEDES-BENZ',
    'Make_MINI',
    'Make_MITSUBISHI',
    'Make_NISSAN',
    'Make_POLESTAR',
    'Make_PORSCHE',
    'Make_SMART',
    'Make_SUBARU',
    'Make_TESLA',
    'Make_TH!NK',
    'Make_TOYOTA',
    'Make_VOLKSWAGEN',
    'Make_VOLVO',
    'Make_WHEEGO ELECTRIC CARS',
    'Model_$16.36K',
    'Model_330E',
    'Model_500',
    'Model_530E',
    'Model_530E XDRIVE',
    'Model_740E XDRIVE',
    'Model_745E',
    'Model_918 SPYDER',
    'Model_A3',
    'Model_A7',
    'Model_A8 E',
    'Model_ACCORD',
    'Model_AVIATOR',
    'Model_B-CLASS',
    'Model_BENTAYGA',
    'Model_BOLT EV',
    'Model_C-CLASS',
    'Model_C-MAX',
    'Model_CARAVAN',
    'Model_CAYENNE',
    'Model_CITY',
    'Model_CLARITY',
    'Model_CORSAIR',
    'Model_COUNTRYMAN',
    'Model_CROSSTREK HYBRID AWD',
    'Model_CT6',
    'Model_E-GOLF',
    'Model_E-TRON',
    'Model_E-TRON SPORTBACK',
    'Model_ELR',
    'Model_EQ FORTWO',
    'Model_ESCAPE',
    'Model_FOCUS',
    'Model_FORTWO',
    'Model_FORTWO ELECTRIC DRIVE',
    'Model_FUSION',
    'Model_GLC-CLASS',
    'Model_GLE-CLASS',
    'Model_HARDTOP',
    'Model_I-MIEV',
    'Model_I-PACE',
    'Model_I3',
    'Model_I8',
    'Model_IONIQ',
    'Model_KARMA',
    'Model_KONA',
    'Model_LEAF',
    'Model_LIFE',
    'Model_MODEL 3',
    'Model_MODEL S',
    'Model_MODEL X',
    'Model_MODEL Y',
    'Model_NIRO',
    'Model_NIRO ELECTRIC',
    'Model_NIRO PLUG-IN HYBRID',
    'Model_OPTIMA',
    'Model_OPTIMA PLUG-IN HYBRID',
    'Model_OUTLANDER',
    'Model_PACIFICA',
    'Model_PANAMERA',
    'Model_PRIUS PLUG-IN',
    'Model_PRIUS PLUG-IN HYBRID',
    'Model_PRIUS PRIME',
    'Model_PS2',
    'Model_Q5',
    'Model_Q5 E',
    'Model_RANGE ROVER',
    'Model_RANGE ROVER SPORT',
    'Model_RANGER',
    'Model_RAV4',
    'Model_RAV4 PRIME',
    'Model_ROADSTER',
    'Model_S-CLASS',
    'Model_S60',
    'Model_S90',
    'Model_SANTA FE',
    'Model_SONATA',
    'Model_SONATA PLUG-IN HYBRID',
    'Model_SORENTO',
    'Model_SOUL',
    'Model_SOUL EV',
    'Model_SPARK',
    'Model_TAYCAN',
    'Model_TRANSIT CONNECT ELECTRIC',
    'Model_TUCSON',
    'Model_VOLT',
    'Model_WRANGLER',
    'Model_X3',
    'Model_X5',
    'Model_XC60',
    'Model_XC60 AWD',
    'Model_XC60 AWD PHEV',
    'Model_XC90',
    'Model_XC90 AWD',
    'Model_XC90 AWD PHEV',
    'Electric_Vehicle_Type_Battery Electric Vehicle (BEV)',
    'Electric_Vehicle_Type_Plug-in Hybrid Electric Vehicle (PHEV)',
    'Electric_Vehicle_Type_nan',
    'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility_Clean Alternative Fuel Vehicle Eligible',
    'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility_Not eligible due to low battery range',
    'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility_nan']  # Replace with your full list as shown earlier

        for col in correct_column_order:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[correct_column_order]

# === Predict Price ===
if current_page == "calculator":
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("Estimate"):
            predicted_price = model.predict(input_df)[0]
            st.subheader("💰 Estimated Price:")
            st.success(f"${predicted_price * 1000:,.2f}")
