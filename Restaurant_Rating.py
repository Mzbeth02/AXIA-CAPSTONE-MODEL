# Import Libraries
import streamlit as st
import os
import numpy as np
import joblib  
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#loading models
def load_scaler(): 
 scaler = joblib.load('projectscaler.joblib')
 return scaler

def load_encoder(): 
 encoder = joblib.load('projectencoder.joblib')
 return encoder

def load_model():
 model = joblib.load('projectmodelcompress.joblib')
 return model

st.sidebar.title("Welcome to DamDee Dining Rating Predictor" )
#st.sidebar.info("Predict! Improve! Thrive!")
st.sidebar.markdown("### **✅ Predict! Improve! Thrive!**")
st.image('https://raw.githubusercontent.com/Mzbeth02/axia-capstone-model/main/Restaurant.jpg', width = 1500)
st.title('🍴Dining Rating Predictor')
st.info("##### **This app is designed for restaurants to predict their overall rating**")

name = st.text_input('Restaurant Name')
category = st.selectbox("Restaurant Category", ['Economy', 'Mid-range', 'Premium', 'Luxury'])
location_df = pd.DataFrame(
    {
        'City': [
            'Makati City', 'Makati City', 'Mandaluyong City', 'Mandaluyong City',
            'Pasay City', 'Pasay City', 'Pasig City', 'Quezon City',
            'San Juan City', 'San Juan City', 'New Delhi', 'New Delhi',
            'New Delhi', 'New Delhi', 'New Delhi', 'New Delhi', 'New Delhi',
            'New Delhi', 'New Delhi', 'New Delhi', 'New Delhi', 'New Delhi',
            'New Delhi', 'New Delhi', 'New Delhi', 'New Delhi', 'New Delhi',
            'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul',
            'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul',
            'Singapore', 'Singapore', 'Singapore', 'Singapore', 'Singapore',
            'Singapore', 'Singapore', 'Abu Dhabi', 'Abu Dhabi', 'Abu Dhabi',
            'Abu Dhabi', 'Abu Dhabi', 'Abu Dhabi', 'Abu Dhabi', 'Abu Dhabi',
            'Abu Dhabi', 'Abu Dhabi', 'Abu Dhabi', 'Abu Dhabi', 'Abu Dhabi',
            'Abu Dhabi', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai',
            'Birmingham', 'Birmingham', 'Birmingham', 'Birmingham', 'Birmingham',
            'Birmingham', 'Birmingham', 'Edinburgh', 'Edinburgh', 'Edinburgh',
            'Edinburgh', 'Edinburgh', 'Edinburgh', 'London', 'London',
            'London', 'London'
        ],
        'Locality': [
            'Century City Mall, Poblacion, Makati City',
            'Little Tokyo, Legaspi Village, Makati City',
            'Edsa Shangri-La, Ortigas, Mandaluyong City',
            'SM Megamall, Ortigas, Mandaluyong City',
            'SM by the Bay, Mall of Asia Complex, Pasay City',
            'Sofitel Philippine Plaza Manila, Pasay City',
            'Kapitolyo',
            'UP Town Center, Diliman, Quezon City',
            'Addition Hills', 'Little Baguio',
            'Aaya Nagar', 'Adchini', 'Aditya Mega Mall, Karkardooma', 'Aerocity',
            'Aggarwal City Mall, Pitampura', 'Aggarwal City Plaza, Rohini',
            'Alaknanda', 'Ambience Mall, Vasant Kunj', 'Anand Lok', 'Anand Vihar',
            'Andaz Delhi, Aerocity', 'Ansal Plaza Mall, Khel Gaon Marg',
            'ARSS Mall, Paschim Vihar', 'Asaf Ali Road', 'Ashok Vihar Phase 1',
            'Ashok Vihar Phase 2', 'Ashok Vihar Phase 3', 'Asmalmescit', 'Bebek',
            'Beikta Merkez', 'Caddebostan', 'Emirgn', 'Kadky Merkez', 'Karaky',
            'Kouyolu', 'Kurueme', 'Moda', 'Bayfront Avenue, Downtown Core',
            'Bayfront Subzone, Downtown Core', 'Cantonment Road, Outram',
            'Chinatown, Outram', 'City Hall, Downtown Core', 'Duxton Hill, Outram',
            'Haji Lane, Rochor', 'Abu Dhabi Mall, Tourist Club Area  Al Zahiyah',
            'Al Dhafrah', 'Al Mushrif', 'Al Wahda Mall, Al Wahda',
            'Crowne Plaza Abu Dhabi, Al Markaziya', 'Dalma Mall, Mussafah Sanaiya',
            'Madinat Zayed', 'Madinat Zayed Shopping Centre, Madinat Zayed',
            'Mushrif Mall, Al Mushrif', 'Mussafah Sanaiya', 'Najda',
            'Venetian Village, Al Maqtaa', 'World Trade Center Mall, Al Markaziya',
            'Yas Mall, Yas Island', 'Al Barari', 'Al Karama', 'Barsha 2',
            'CITY WALK, Al Safa', 'Deira City Centre Area', 'DIFC', 'Harborne',
            'Jewellery Quarter', 'Moseley', 'Small Heath', 'Smethwick',
            'Sparkhill', 'The Mailbox, Broad Street', 'Fountainbridge', 'Leith',
            'New Town', 'Old Town', 'Tollcross', 'Twelve Picardy Place, New Town',
            'Albemarle Street, Mayfair', 'Archer Street, Soho', 'Beak Street, Soho',
            'Bishopsgate, City Of London'
        ]
    }
)

# Select City
city_options = location_df ['City'].unique()
city = st.selectbox("City", city_options)

# Step 2: Filter locality based on selected city

filtered_locality = location_df[location_df['City'] == city]['Locality'].unique()
locality = st.selectbox("Locality", filtered_locality)
table_booking = st.radio('Has Table Booking?', ['No', 'Yes'])
online_delivery = st.radio('Has Online Delivery?', ['No', 'Yes'])

cuisine = st.selectbox('Cuisines', ['Japanese','Desserts','Seafood','Asian','Filipino','Indian','Sushi',
'Korean','Mediterranean','Fast Food','Brazilian','Arabian','Bar Food','Grill','International','Cantonese',
'Dim Sum','Western','Finger Food','British','Deli','Indonesian','North Indian','Mughlai','Biryani','Taiwanese',
'Fish and Chips','Contemporary','Scottish','Curry','Patisserie','Kebab','Turkish Pizza','Sandwich','Steak'])

currency = st.selectbox('Currency',['Botswana Pula(P)','Brazilian Real(R$)','Dollar($)',
                         'Emirati Diram(AED)','Indian Rupees(Rs.)','Indonesian Rupiah(IDR)',
                         'NewZealand($)','Pounds(£)','Qatari Rial(QR)','Rand(R)','Sri Lankan Rupee(LKR)',
                         'Turkish Lira(TL)'])
average_cost = st.number_input('Average cost for two', min_value=0, max_value=800000, help= "Average cost for 2 ranges from 0 to 800,000")
st.markdown("###### **Note: Number of ratings lesser than 4 will result into 0 aggregate rating**")
votes = st.number_input('Number of Ratings', min_value=0, max_value=10934, help= 'Number of rating ranges from 0 to 10,934')
pred = st.button('Predict Rating')

num_col = (average_cost, votes)
n_feature = ['Average Cost for two', 'Votes']

cat_col = (category, city, locality, cuisine, currency, table_booking, online_delivery)
c_feature = ['Restaurant Category', 'City', 'Locality', 'Cuisines', 'Currency', 'Has Table booking',
       'Has Online delivery']

num_col_values = pd.DataFrame([num_col], columns=n_feature)
cat_col_values = pd.DataFrame([cat_col], columns=c_feature)

encoder = load_encoder()
scaler = load_scaler()
model = load_model()

encoded_data = encoder.transform(cat_col_values).toarray()
scaled_data = scaler.transform(num_col_values)
processed_data = np.hstack((encoded_data, scaled_data))
prediction = model.predict(processed_data)[0]

if pred:
    if not name or not category or not city or not locality or not table_booking or not online_delivery or not currency  or not cuisine:
        st.error('Please fill in all required fields before submitting.')
    else:
        st.write(round(prediction,4))
        if round(prediction,4) < 2.5:
            st.write('Poor 🙁')
        elif round(prediction,4) >= 2.5 and round(prediction,4) < 3.5:
            st.write('Average 😐')
        elif round(prediction,4) >= 3.5 and round(prediction,4) <4.0:
            st.write('Good 👍🏼')      
        elif round(prediction,4) >= 4.0 and round(prediction,4) <= 4.4:
            st.write('Very Good 👏🏼')
        else:
            st.write('Excellent 💯')         
