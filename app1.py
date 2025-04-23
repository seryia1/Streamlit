import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# === PAGE SETUP ===
st.set_page_config(layout="wide", page_title="Car Price Analysis & Prediction App")
# === GET PAGE FROM URL ===
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["overview"])[0]
# === CUSTOM CSS FOR DARK THEME ===
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #333;
    }
    
    /* Card-like containers */
    .card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Tabs styling */
    .tab-nav {
        display: flex;
        border-bottom: 1px solid #333;
        margin-bottom: 20px;
        padding-bottom: 10px;
    }
    
    .tab {
        padding: 8px 16px;
        margin-right: 10px;
        cursor: pointer;
        border-radius: 5px;
        font-weight: 500;
    }
    
    .tab-active {
        border-bottom: 2px solid #e11d48;
        color: #e11d48;
    }
    
    /* Button styling */
    .custom-button {
        background-color: #e11d48;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s;
        width: 100%;
        text-align: center;
    }
    
    .custom-button:hover {
        background-color: #be123c;
    }
    
    /* Dropdown styling */
    .stSelectbox label, .stSlider label {
        color: #e5e5e5 !important;
        font-weight: 500;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: white;
    }
    
    /* Red accent color for highlights */
    .accent {
        color: #e11d48;
    }
    
    /* Car icon */
    .car-icon {
        width: 100%;
        max-width: 150px;
        margin-bottom: 20px;
    }
    
    /* Prediction result */
    .prediction-result {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 4px solid #e11d48;
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Custom select box */
    .custom-select {
        background-color: #2a2a2a;
        border: 1px solid #333;
        border-radius: 5px;
        color: white;
        padding: 10px;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background-color: #333;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# === CAR ICON SVG ===
car_icon = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="#e11d48" class="car-icon">
  <path d="M85,40c-0.6-1.5-1.6-2.9-2.9-4c-1.2-1-2.7-1.7-4.2-2c-0.4-0.1-0.8-0.1-1.2-0.1h-4.3l-7.1-11.2c-1.1-1.7-2.7-3-4.5-3.9
    c-1.8-0.8-3.9-1.3-5.9-1.3H35.1c-2,0-4.1,0.4-5.9,1.3c-1.8,0.8-3.4,2.2-4.5,3.9L17.6,34h-4.3c-0.4,0-0.8,0-1.2,0.1
    c-1.5,0.3-3,1-4.2,2c-1.3,1.1-2.3,2.5-2.9,4c-0.6,1.5-0.8,3.2-0.5,4.8l2.3,12.5c0.2,1.1,0.8,2.1,1.6,2.9c0.8,0.8,1.9,1.3,3,1.3h2.1
    v7.8c0,1.5,0.6,2.9,1.7,4c1.1,1.1,2.5,1.7,4,1.7h5.2c1.5,0,2.9-0.6,4-1.7c1.1-1.1,1.7-2.5,1.7-4v-7.8h36.8v7.8c0,1.5,0.6,2.9,1.7,4
    c1.1,1.1,2.5,1.7,4,1.7h5.2c1.5,0,2.9-0.6,4-1.7c1.1-1.1,1.7-2.5,1.7-4v-7.8h2.1c1.1,0,2.2-0.5,3-1.3c0.8-0.8,1.4-1.8,1.6-2.9
    l2.3-12.5C85.8,43.2,85.6,41.5,85,40z M30.4,27.2c0.5-0.8,1.2-1.4,2-1.8c0.8-0.4,1.8-0.6,2.7-0.6h29.8c0.9,0,1.9,0.2,2.7,0.6
    c0.8,0.4,1.5,1,2,1.8L75,38H25L30.4,27.2z M25.5,55.5c-1.5,0-3-0.6-4-1.7c-1.1-1.1-1.7-2.5-1.7-4c0-1.5,0.6-2.9,1.7-4
    c1.1-1.1,2.5-1.7,4-1.7c1.5,0,3,0.6,4,1.7c1.1,1.1,1.7,2.5,1.7,4c0,1.5-0.6,2.9-1.7,4C28.5,54.9,27,55.5,25.5,55.5z M74.5,55.5
    c-1.5,0-3-0.6-4-1.7c-1.1-1.1-1.7-2.5-1.7-4c0-1.5,0.6-2.9,1.7-4c1.1-1.1,2.5-1.7,4-1.7c1.5,0,3,0.6,4,1.7c1.1,1.1,1.7,2.5,1.7,4
    c0,1.5-0.6,2.9-1.7,4C77.5,54.9,76,55.5,74.5,55.5z"/>
</svg>
"""

# === LOAD DATA AND MODEL ===
try:
    df = pd.read_csv('Electric_cars_dataset.csv')
    model = joblib.load("svr_model.joblib")
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.warning("Please update the file paths in the code to match your environment.")
    # Create sample data for demonstration
    df = pd.DataFrame({
        'Make': ['TESLA', 'BMW', 'NISSAN'],
        'Model': ['MODEL 3', 'I3', 'LEAF'],
        'Model_Year': [2020, 2019, 2018],
        'Electric_Vehicle_Type': ['Battery Electric Vehicle (BEV)', 'Battery Electric Vehicle (BEV)', 'Battery Electric Vehicle (BEV)'],
        'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility': ['Clean Alternative Fuel Vehicle Eligible', 'Clean Alternative Fuel Vehicle Eligible', 'Clean Alternative Fuel Vehicle Eligible'],
        'Electric_Range': [300, 150, 200],
        'County': ['King', 'Pierce', 'Snohomish'],
        'City': ['Seattle', 'Tacoma', 'Everett'],
        'Electric_Utility': ['SEATTLE CITY LIGHT', 'TACOMA POWER', 'SNOHOMISH COUNTY PUD'],
        'Legislative_District': ['43', '27', '38']
    })
    model = None

# Clean column names
df.columns = df.columns.str.replace(' ', '_')

# Define columns for processing
drop_cols = ['ID', 'State', 'VIN_(1-10)', 'ZIP_Code', 'DOL_Vehicle_ID', 'Vehicle_Location', 'Base_MSRP']
freq_cols = ['County', 'Electric_Utility', 'Legislative_District', 'City']
onehot_cols = ['Make', 'Model', 'Electric_Vehicle_Type', 'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility']
scale_cols = ['Model_Year', 'Electric_Range']

# Get unique values for each column
freq_uniques = {col: df[col].dropna().unique().tolist() for col in freq_cols}
onehot_uniques = {col: df[col].dropna().unique().tolist() for col in onehot_cols}
scale_uniques = {col: df[col].dropna().unique().tolist() for col in scale_cols}

# === APP LAYOUT ===
# Main container
main_container = st.container()

with main_container:
    # App header with logo and title
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.markdown(car_icon, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>Car Price Analysis & Prediction App</h1>", unsafe_allow_html=True)
    
    # Custom tabs
    st.markdown("""
    <div class="tab-nav">
        <div class="tab tab-active" id="tab-home" onclick="switchTab('home')">
            <span style="color: #e11d48;">●</span> Home
        </div>
        <div class="tab" id="tab-analysis" onclick="switchTab('analysis')">
            Analysis
        </div>
        <div class="tab" id="tab-prediction" onclick="switchTab('prediction')">
            Prediction
        </div>
    </div>
    
    <script>
    function switchTab(tabName) {
        // This would normally handle tab switching, but we'll use Streamlit's own tab system
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Create tabs using Streamlit
    tab1, tab2, tab3 = st.tabs(["Home", "Analysis", "Prediction"])
    
    # === HOME TAB ===
    with tab1:
        st.markdown("""
        <div class="card">
            <h2>Are you curious about the potential market price of a car?</h2>
            <p>This app allows you to predict the resale price of vehicles using machine learning! Simply input the car's details like its model, year, engine size, and more. Our algorithm will provide an accurate price estimate based on historical data. Whether you are buying or selling, this tool can help you make informed decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Car silhouette image
        st.markdown("""
        <div style="text-align: center; margin: 40px 0;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400" fill="#e11d48" style="max-width: 500px; width: 100%;">
              <path d="M700,200c-10-30-30-60-60-80c-30-20-60-30-100-30c-30,0-60,5-90,15c-30,10-60,25-90,45c-20,15-40,30-60,40
                c-20,10-40,15-60,15c-30,0-60-10-80-30c-20-20-30-40-30-70c0-20,5-40,15-60c10-20,25-40,45-60c-30,10-60,25-80,45
                c-20,20-40,40-50,60c-10,20-15,40-15,60c0,30,10,60,30,80c20,20,50,30,80,30c20,0,40-5,60-15c20-10,40-25,60-45
                c30-30,60-50,90-60c30-10,60-15,90-15c30,0,60,5,90,15c30,10,60,30,80,50c20,20,30,40,30,60c0,15-5,30-15,40
                c-10,10-25,15-45,15c-15,0-30-5-45-15c-15-10-30-25-45-45c-10,30-5,60,15,80c20,20,50,30,90,30c30,0,60-10,80-30
                c20-20,30-40,30-70C720,240,710,220,700,200z"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset overview
        st.markdown("""
        <div class="card">
            <h2>Dataset Overview</h2>
            <p>For training our model, we worked with a dataset containing detailed information about Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) registered with the Washington State Department of Licensing (DOL).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature description table
        st.markdown("""
        <div class="card">
            <h3>Dataset Features</h3>
            <table style="width:100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #333;">Feature</th>
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #333;">Description</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">VIN (1-10)</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">The first 10 characters of each vehicle's Vehicle Identification Number (VIN).</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">County</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">The county where the registered owner resides.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Model Year</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Determined by decoding the vehicle's VIN, indicating the year of manufacture.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Make</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">The vehicle manufacturer, decoded from the VIN.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Model</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">The specific model of the vehicle, also derived from the VIN.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Electric Vehicle Type</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Classification as either a fully electric or a plug-in hybrid vehicle.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Electric Range</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">The distance a vehicle can travel purely on its electric battery.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">Expected Price</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">The estimated resale value of the vehicle, which we aim to predict.</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # === ANALYSIS TAB ===
    with tab2:
        st.markdown("""
        <div class="card">
            <h2>🔍 Model Evaluation Overview</h2>
            <p>We trained and fine-tuned a machine learning model called Support Vector Regression (SVR) to predict the expected resale price of a car based on features such as the make, model, fuel type, battery capacity, and more.</p>
            <p>To find the best version of this model, we tested different settings (called hyperparameters) and compared their performance. Below is a summary of the results from our tuning process:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model performance table
        st.markdown("""
        <div class="card">
            <table style="width:100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #333;">SVR Settings</th>
                    <th style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">R² Score</th>
                    <th style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">Mean Squared Error (MSE)</th>
                    <th style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">Root Mean Squared Error (RMSE)</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">C=7000, ε=2.3</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">0.9855</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">7.19</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">2.68</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">C=14000, ε=2.3</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">0.9870</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">6.43</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">2.54</td>
                </tr>
                <tr style="background-color: rgba(225, 29, 72, 0.1);">
                    <td style="padding: 8px; border-bottom: 1px solid #333;">✅ C=28000, ε=2.3</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">0.9882</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">5.87</td>
                    <td style="text-align: center; padding: 8px; border-bottom: 1px solid #333;">2.42</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Final model performance
        st.markdown("""
        <div class="card">
            <h3>✅ Final Model Performance</h3>
            <p>After tuning, the best performing model had the following settings:</p>
            <ul>
                <li>C = 28000</li>
                <li>Epsilon (ε) = 2.3</li>
                <li>Gamma = 'scale' (default scaling method for SVR)</li>
            </ul>
            <p>With these parameters, the model achieved:</p>
            <ul>
                <li>R² Score: 0.9882</li>
                <li>RMSE: 2.42</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # What does this mean
        st.markdown("""
        <div class="card">
            <h3>📈 What does this mean?</h3>
            <ul>
                <li><strong>R² Score (0.9882):</strong> This is a measure of how well the model's predictions match the actual prices. A perfect score would be 1.0, and anything above 0.95 is considered excellent. In our case, the model explains more than 98% of the variation in car prices, which indicates a very accurate prediction capability.</li>
                <li><strong>RMSE (2.42):</strong> This tells us the average error in price predictions. On average, the model's predictions are within about ±2.42 units of the actual price (2420$). A lower RMSE means better accuracy.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary
        st.markdown("""
        <div class="card">
            <h3>💡 TL;DR – Summary</h3>
            <p>Our model is very accurate at predicting the resale price of a car based on its characteristics. It has been fine-tuned and tested for performance, and the final version predicts car prices with over 98% accuracy, and an average error of only 2.42.</p>
            <p>Whether you're buying or selling, this tool gives you trusted price estimates based on real data and advanced modeling.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # === PREDICTION TAB ===
    with tab3:
        # Main layout with sidebar-like left panel and content area
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Select Vehicle Details</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Vehicle filters
            make = st.selectbox("🚘 Make", onehot_uniques['Make'])
            model_car = st.selectbox("📦 Model", onehot_uniques['Model'])
            
            # Year slider
            min_year = int(min(scale_uniques['Model_Year'])) if scale_uniques['Model_Year'] else 2010
            max_year = int(max(scale_uniques['Model_Year'])) if scale_uniques['Model_Year'] else 2023
            model_year = st.slider("📅 Model Year", min_year, max_year, value=min_year + (max_year - min_year) // 2)
            
            # EV type
            ev_type = st.selectbox("⚡ Electric Vehicle Type", onehot_uniques['Electric_Vehicle_Type'])
            
            # CAFV eligibility
            cafv = st.selectbox("♻️ CAFV Eligibility", onehot_uniques['Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility'])
            
            # Electric range
            min_range = int(min(scale_uniques['Electric_Range'])) if scale_uniques['Electric_Range'] else 50
            max_range = int(max(scale_uniques['Electric_Range'])) if scale_uniques['Electric_Range'] else 400
            electric_range = st.slider("🔋 Electric Range (miles)", min_range, max_range, value=min_range + (max_range - min_range) // 2)
            
            # Location details
            county = st.selectbox("🏙️ County", freq_uniques['County'])
            city = st.selectbox("📍 City", freq_uniques['City'])
            utility = st.selectbox("🏢 Electric Utility", freq_uniques['Electric_Utility'])
            district = st.selectbox("🏛️ Legislative District", freq_uniques['Legislative_District'])
        
        with col2:
            st.markdown("""
            <div class="card">
                <h2>Car Price Analysis & Prediction</h2>
                <p>Are you curious about the potential market price of a car? This app allows you to predict the resale price of vehicles using machine learning!</p>
                <p>Simply input the car's details like its model, year, engine size, and more. Our algorithm will provide an accurate price estimate based on historical data. Whether you are buying or selling, this tool can help you make informed decisions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Car image
            st.image("https://www.neodrift.in/cdn/shop/articles/best-resale-cars-featured.jpg?v=1722222661", use_column_width=True)
            
            # Create input DataFrame for prediction
            if st.button("Estimate Price", type="primary"):
                try:
                    # Create raw input DataFrame
                    input_df = pd.DataFrame({
                        'Make': [make],
                        'Model': [model_car],
                        'Model_Year': [model_year],
                        'Electric_Vehicle_Type': [ev_type],
                        'Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility': [cafv],
                        'Electric_Range': [electric_range],
                        'County': [county],
                        'Electric_Utility': [utility],
                        'Legislative_District': [district],
                        'City': [city]
                    })

                    # === Frequency Encoding ===
                    for col in freq_cols:
                        mapping = df[col].value_counts().to_dict()
                        input_df[col + '_freq'] = input_df[col].map(mapping).fillna(0)

                    freq_scaled = MinMaxScaler()
                    input_df[[col + '_freq' for col in freq_cols]] = freq_scaled.fit_transform(
                        input_df[[col + '_freq' for col in freq_cols]]
                    )
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
                    correct_column_order = ['Model_Year','Electric_Range','County_freq','Electric_Utility_freq','Legislative_District_freq','City_freq','Make_AUDI','Make_AZURE DYNAMICS','Make_BENTLEY','Make_BMW','Make_CADILLAC','Make_CHEVROLET','Make_CHRYSLER','Make_DODGE','Make_FIAT','Make_FISKER','Make_FORD','Make_HONDA','Make_HYUNDAI','Make_JAGUAR','Make_JEEP','Make_KIA','Make_LAND ROVER','Make_LINCOLN','Make_MERCEDES-BENZ','Make_MINI','Make_MITSUBISHI','Make_NISSAN','Make_POLESTAR','Make_PORSCHE','Make_SMART','Make_SUBARU','Make_TESLA','Make_TH!NK','Make_TOYOTA','Make_VOLKSWAGEN','Make_VOLVO','Make_WHEEGO ELECTRIC CARS','Model_$16.36K','Model_330E','Model_500','Model_530E','Model_530E XDRIVE','Model_740E XDRIVE','Model_745E','Model_918 SPYDER','Model_A3','Model_A7','Model_A8 E','Model_ACCORD','Model_AVIATOR','Model_B-CLASS','Model_BENTAYGA','Model_BOLT EV','Model_C-CLASS','Model_C-MAX','Model_CARAVAN','Model_CAYENNE','Model_CITY','Model_CLARITY','Model_CORSAIR','Model_COUNTRYMAN','Model_CROSSTREK HYBRID AWD','Model_CT6','Model_E-GOLF','Model_E-TRON','Model_E-TRON SPORTBACK','Model_ELR','Model_EQ FORTWO','Model_ESCAPE','Model_FOCUS','Model_FORTWO','Model_FORTWO ELECTRIC DRIVE','Model_FUSION','Model_GLC-CLASS','Model_GLE-CLASS','Model_HARDTOP','Model_I-MIEV','Model_I-PACE','Model_I3','Model_I8','Model_IONIQ','Model_KARMA','Model_KONA','Model_LEAF','Model_LIFE','Model_MODEL 3','Model_MODEL S','Model_MODEL X','Model_MODEL Y','Model_NIRO','Model_NIRO ELECTRIC','Model_NIRO PLUG-IN HYBRID','Model_OPTIMA','Model_OPTIMA PLUG-IN HYBRID','Model_OUTLANDER','Model_PACIFICA','Model_PANAMERA','Model_PRIUS PLUG-IN','Model_PRIUS PLUG-IN HYBRID','Model_PRIUS PRIME','Model_PS2','Model_Q5','Model_Q5 E','Model_RANGE ROVER','Model_RANGE ROVER SPORT','Model_RANGER','Model_RAV4','Model_RAV4 PRIME','Model_ROADSTER','Model_S-CLASS','Model_S60','Model_S90','Model_SANTA FE','Model_SONATA','Model_SONATA PLUG-IN HYBRID','Model_SORENTO','Model_SOUL','Model_SOUL EV','Model_SPARK','Model_TAYCAN','Model_TRANSIT CONNECT ELECTRIC','Model_TUCSON','Model_VOLT','Model_WRANGLER','Model_X3','Model_X5','Model_XC60','Model_XC60 AWD','Model_XC60 AWD PHEV','Model_XC90','Model_XC90 AWD','Model_XC90 AWD PHEV','Electric_Vehicle_Type_Battery Electric Vehicle (BEV)','Electric_Vehicle_Type_Plug-in Hybrid Electric Vehicle (PHEV)','Electric_Vehicle_Type_nan','Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility_Clean Alternative Fuel Vehicle Eligible','Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility_Not eligible due to low battery range','Clean_Alternative_Fuel_Vehicle_(CAFV)_Eligibility_nan']

                    
                    # Add one-hot encoded columns
                    for col in encoded_df.columns:
                        correct_column_order.append(col)

                    # Ensure all columns exist
                    for col in correct_column_order:
                        if col not in input_df.columns:
                            input_df[col] = 0
                            input_df = input_df[correct_column_order]

                    

                    # === Predict Price ===
                    if model is not None:
                        predicted_price = model.predict(input_df)[0]
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h3>💰 Estimated Price:</h3>
                            <p style="font-size: 32px; color: #e11d48;">${predicted_price * 1000:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Model not loaded. This is a demonstration of the UI only.")
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h3>💰 Estimated Price (Demo):</h3>
                            <p style="font-size: 32px; color: #e11d48;">$35,750.00</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("Please check your input data and try again.")

# Add footer
st.markdown("""
<div style="background-color: #1a1a1a; padding: 20px; text-align: center; margin-top: 30px; border-top: 1px solid #333;">
    <p>Car Price Analysis & Prediction App © 2025</p>
    <p style="font-size: 12px; color: #888;">Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
