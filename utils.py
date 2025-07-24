import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN

def load_and_merge_data():
    attacks = pd.read_csv('data/pirate_attacks.csv')
    codes = pd.read_csv('data/country_codes.csv')
    indicators = pd.read_csv('data/country_indicators.csv')

    # Normalize column names for safety
    attacks.columns = [col.strip().title() for col in attacks.columns]
    codes.columns = [col.strip() for col in codes.columns]
    indicators.columns = [col.strip() for col in indicators.columns]

    # Merge data
    merged = attacks.merge(codes, how='left', left_on='Nearest_Country', right_on='country_name')
    merged = merged.merge(indicators, how='left', on='country')

    # Fix date column
    if 'Date' in merged.columns:
        merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
    else:
        merged['Date'] = pd.NaT

    # Ensure Latitude & Longitude
    if 'Latitude' not in merged.columns or 'Longitude' not in merged.columns:
        raise KeyError("Latitude or Longitude columns are missing.")

    merged = merged.dropna(subset=['Latitude', 'Longitude'])

    return merged, indicators

def train_risk_model(attacks_df, indicators_df, save_path='models/risk_model.pkl'):
    if 'Country' not in attacks_df.columns and 'Nearest_Country' in attacks_df.columns:
        attacks_df['Country'] = attacks_df['Nearest_Country']

    attack_counts = attacks_df['Country'].value_counts().reset_index()
    attack_counts.columns = ['Country Name', 'Attack Count']

    df = attack_counts.merge(indicators_df, left_on='Country Name', right_on='country', how='left')
    threshold = df['Attack Count'].quantile(0.75)
    df['High Risk'] = (df['Attack Count'] >= threshold).astype(int)
    df.fillna(0, inplace=True)

    X = df.drop(columns=['Country Name', 'High Risk', 'country'])
    y = df['High Risk']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, save_path)

    return model

def load_risk_model(path='models/risk_model.pkl'):
    return joblib.load(path)

def prepare_features(country_name, indicators_df):
    row = indicators_df[indicators_df['country'] == country_name]
    if row.empty:
        return None
    return row.drop(columns=['country']).fillna(0).values

def cluster_attack_locations(df, eps=1.0, min_samples=5):
    # Clean up column names
    df.columns = [col.strip().title() for col in df.columns]
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        raise KeyError("Latitude or Longitude columns are missing.")

    locs = df[['Latitude', 'Longitude']].dropna().astype(float)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(locs)
    df['Cluster'] = labels
    return df
