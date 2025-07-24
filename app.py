import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from utils import load_and_merge_data, cluster_attack_locations, load_risk_model, prepare_features

# Load and process data
df, indicators_df = load_and_merge_data()
df = cluster_attack_locations(df)

st.title("🌊 Pirate Attack Risk Dashboard")

# Sidebar filters
available_countries = df['Nearest_Country'].dropna().unique().tolist()
selected_country = st.sidebar.selectbox("Select Country", available_countries)

if selected_country:
    selected_df = df[df['Nearest_Country'] == selected_country]

    st.subheader(f"Pirate Attacks in {selected_country}")
    if not selected_df.empty:
        st.dataframe(selected_df[['Date', 'Latitude', 'Longitude', 'Cluster']])
    else:
        st.warning("No data available for this country.")

    # Show on Map
    st.subheader("Map of Attacks")
    m = folium.Map(location=[selected_df['Latitude'].mean(), selected_df['Longitude'].mean()], zoom_start=4)
    for _, row in selected_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"Cluster: {row['Cluster']}",
            color='red' if row['Cluster'] != -1 else 'blue',
            fill=True
        ).add_to(m)
    folium_static(m)
