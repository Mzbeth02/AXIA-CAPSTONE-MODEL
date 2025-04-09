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

def load_scaler(): 
 encoder = joblib.load('projectencoder.joblib')
 return scaler

def load_model():
 model = joblib.load('projectmodelcompress.joblib')
 return model

st.sidebar.title("Welcome to Jolly's Dining Rating Predictor" )
#st.sidebar.info("Predict! Improve! Thrive!")
st.sidebar.markdown("### **✅ Predict! Improve! Thrive!**")
st.image('https://raw.githubusercontent.com/Mzbeth02/axia-capstone-model/main/Restaurant.jpg', width = 1500)
st.title('🍴Dining Rating Predictor')
st.info("This form is to be completed by restaurant owners to predict their cuisines' rating")

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
        ],
        'Restaurant ID': [
            6317637, 6304287, 6300002, 6318506, 6300781, 6300010, 6314987, 6318433,
            6310470, 6314605, 18287358, 18216944, 18334443, 18258757, 310491, 3370,
            7580, 7249, 5565, 6258, 18429396, 179, 310281, 18207831, 18258778,
            310802, 1192, 5904116, 5901782, 5902117, 5927248, 5905215, 5926979,
            5916085, 5908749, 5915807, 5927402, 18483372, 18484349, 18496057,
            18483389, 18483222, 18483085, 18484423, 18212135, 5701978, 5704168,
            18277098, 5700052, 5702418, 5700386, 5704202, 5701052, 5704118, 5701548,
            5703500, 18253896, 5702574, 202321, 206488, 18340881, 18233284,
            18269368, 18254160, 6901051, 6900388, 18273002, 6901394, 6900843,
            6900992, 6900069, 7600803, 7600217, 7602204, 7601106, 7601177, 7600921,
            6118140, 6103683, 6114338, 6114829
        ],
        'Restaurant Name': [
            'Le Petit Souffle','Izakaya Kikufuji','Heat - Edsa Shangri-La','Ooma','Buffet 101','Spiral - Sofitel Philippine Plaza Manila',
              'Locavore','Silantro Fil-Mex','Guevarras','Sodam Korean Restaurant','Food Cloud','Burger.in',
              'BarShala', 'Bella Italia', '4 on 44 Restaurant  Bar', 'BTW', 'Aggarwals Sweets Paradise',
              'Alaturka', 'Mithapur', 'Aggarwal Sweet India','Juniper Bar - Andaz Delhi','McDonalds',
              'Haldirams','Jux Pux - The Dreamers Cafe','34, Chowringhee Lane','Kanwarjis','Apni Rasoi',
              'Jadore Chocolatier', 'Starbucks', 'Valonia','Draft Gastro Pub','Emirgan Sti','Leman Kltr',
              'Dem Karaky','Ceviz Aac','Huqqa', 'Walters Coffee Roastery','Sky On 57','Cut By Wolfgang Puck',
              'Restaurant Andre','Potato Head Folk','Jaan', 'Rhubarb Le Restaurant','Alfrank Cookies',
              'Dennys', 'Pizza Di Rocco', 'Salt','Genghis Grill', 'Cho Gao - Crowne Plaza Abu Dhabi',
              'Gazebo', 'Sangeetha Vegetarian Restaurant', 'Hot Palayok', 'Applebees',  'Tikka Tonight',
              'Bait El Khetyar', 'Punjab Grill','Tamba','The Cheesecake Factory','The Farm','Maharaja Bhog',
              'Barbeque Nation','Farzi Cafe','ABs Absolute Barbecues', 'Carnival By Tresind','The Plough',
              'Lasan Restaurant','Damascena Coffee House', 'Jamjar','Chennai Dosa','Mughal E Azam','Bar Estilo',
              'Loudons Cafe  Bakery','La Favorita','El Cartel', '10 To 10 In Delhi','Tuk Tuk Indian Street Food',
              'Steak','Gymkhana','Bocca Di Lupo','Flat Iron', 'Duck  Waffle',
]
    }
)

# Select id
id_options = location_df ['Restaurant ID'].unique()
id = st.selectbox("Restaurant ID", id_options)

# Step 2: Filter city and locality based on selected id
filtered_name = location_df[location_df['Restaurant ID'] == id]['Restaurant Name'].unique()
name = st.selectbox("Restaurant Name", filtered_name)


filtered_city = location_df[location_df['Restaurant ID'] == id]['City'].unique()
city = st.selectbox("City", filtered_city)


filtered_locality = location_df[location_df['Restaurant ID'] == id]['Locality'].unique()
locality = st.selectbox("Locality", filtered_locality)

#id = st.number_input('Restaurant ID', 0)
#city = st.text_input('City')
#city = st.selectbox("City", ['Makati City','Mandaluyong City','Pasay City','Pasig City','Quezon City','San Juan City',
#'Santa Rosa','Tagaytay City','Taguig City','Rio de Janeiro','Albany','Armidale','Athens','Augusta','Balingup',
#'Beechworth','Boise','Cedar Rapids/Iowa City','Chatham-Kent','Clatskanie','Cochrane','Columbus','Consort',
#'Dalton','Davenport','Des Moines','Dicky Beach','Dubuque','East Ballina','Fernley','Flaxton','Forrest',
#'Gainesville','Hepburn Springs','Huskisson','Inverloch','Lakes Entrance','Lakeview','Lincoln','Lorn',
#'Macedon','Macon','Mayfield','Mc Millan','Middleton Beach','Monroe','Montville','Ojo Caliente','Orlando',
#'Palm Cove','Paynesville','Penola','Pensacola','Phillip Island','Pocatello','Potrero','Princeton',
#'Rest of Hawaii','Savannah','Singapore','Sioux City','Tampa Bay','Tanunda','Trentham East','Valdosta',
#'Vernonia','Victor Harbor','Vineland Station','Waterloo','Weirton','Winchester Bay','Yorkton','Abu Dhabi',
#'Dubai','Sharjah','Agra','Ahmedabad','Allahabad','Amritsar','Aurangabad','Bangalore','Bhopal','Bhubaneshwar',
#'Chandigarh','Chennai','Coimbatore','Dehradun','Faridabad','Ghaziabad','Goa','Gurgaon','Guwahati',
#'Hyderabad','Indore','Jaipur','Kanpur','Kochi','Kolkata','Lucknow','Ludhiana','Mangalore','Mohali','Mumbai',
#'Mysore','Nagpur','Nashik','New Delhi','Noida','Panchkula','Patna','Puducherry','Pune','Ranchi','Secunderabad',
#'Surat','Vadodara','Varanasi','Vizag','Bandung','Bogor','Jakarta','Tangerang','Auckland','Wellington City',
#'Birmingham','Edinburgh','London','Manchester','Doha','Cape Town','Inner City','Johannesburg','Pretoria',
#'Randburg','Sandton','Colombo','Ankara']
#)
#locality = st.text_input('Locality')
#st.text_input('Address')
#st.caption('Locality speaks to the neighborhood')
table_booking = st.radio('Has Table Booking?', ['No', 'Yes'])
online_delivery = st.radio('Has Online Delivery?', ['No', 'Yes'])
#cuisine = st.text_input('Cuisine', placeholder = "e.g. Dessert, Sunda, French, Italian etc")
cuisine = st.selectbox('Cuisines', ['Japanese','Desserts','Seafood','Asian','Filipino','Indian','Sushi',
'Korean','Mediterranean','Fast Food','Brazilian','Arabian','Bar Food','Grill','International','Cantonese',
'Dim Sum','Western','Finger Food','British','Deli','Indonesian','North Indian','Mughlai','Biryani','Taiwanese',
'Fish and Chips','Contemporary','Scottish','Curry','Patisserie','Kebab','Turkish Pizza','Sandwich','Steak'])

currency = st.selectbox('Currency',['Botswana Pula(P)','Brazilian Real(R$)','Dollar($)',
                         'Emirati Diram(AED)','Indian Rupees(Rs.)','Indonesian Rupiah(IDR)',
                         'NewZealand($)','Pounds(£)','Qatari Rial(QR)','Rand(R)','Sri Lankan Rupee(LKR)',
                         'Turkish Lira(TL)'])
price = st.number_input("Price range", min_value=0, max_value=4, help="Enter number between 1 and 4")
average_cost = st.number_input('Average cost for two', min_value=0)
votes = st.number_input('Number of Votes', 0)
pred = st.button('Predict Aggregate Rating')

num_col = (id, average_cost, price, votes)
n_feature = ['Restaurant ID', 'Average Cost for two', 'Price range', 'Votes']

cat_col = (city, locality, cuisine, currency, table_booking, online_delivery)
c_feature = ['City', 'Locality', 'Cuisines', 'Currency', 'Has Table booking',
       'Has Online delivery']
num_col_values = pd.DataFrame([num_col], columns=n_feature)
cat_col_values = pd.DataFrame([cat_col], columns=c_feature)

encoded_data = encoder.transform(cat_col_values).toarray()
scaled_data = scaler.transform(num_col_values)
processed_data = np.hstack((encoded_data, scaled_data))
prediction = model.predict(processed_data)[0]

#st.write(encoded_data).shape
#st.write(scaled_data).shape
if pred:
    st.write(round(prediction,4))
