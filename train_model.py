import pandas as pd
import numpy as np
import re
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 65)
print("  BANGALORE HOUSE PRICE PREDICTION - Model Training Pipeline")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------------
DATA_FILE = "Bengaluru_House_Data.csv"

def download_dataset():
    if os.path.exists(DATA_FILE):
        print(f"Dataset already exists: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)
    try:
        import kagglehub
        path = kagglehub.dataset_download("amitabhajoy/bengaluru-house-price-data")
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    full = os.path.join(root, f)
                    df = pd.read_csv(full)
                    df.to_csv(DATA_FILE, index=False)
                    print(f"Downloaded dataset from Kaggle: {df.shape[0]} rows")
                    return df
    except Exception as e:
        print(f"Kaggle download failed ({e}), generating comprehensive dataset...")

    return generate_dataset()


def generate_dataset():
    np.random.seed(42)
    locations = [
        'Whitefield', 'Sarjapur Road', 'Electronic City', 'Bannerghatta Road',
        'Thanisandra', 'Yelahanka', 'Hebbal', 'KR Puram', 'Marathahalli',
        'Raja Rajeshwari Nagar', 'Peenya', 'Banashankari', 'Jayanagar',
        'JP Nagar', 'Koramangala', 'HSR Layout', 'BTM Layout', 'Indira Nagar',
        'Malleshwaram', 'Rajaji Nagar', 'Bommanahalli', 'Bellandur',
        'Varthur', 'Hosa Road', 'Haralur Road', 'Kanakapura Road',
        'Hennur Road', 'Old Airport Road', 'Yeshwanthpur', 'Nagarbhavi',
        'Uttarahalli', 'Sahakara Nagar', 'Ramamurthy Nagar', 'Kadugodi',
        'Hoodi', 'Akshaya Nagar', 'Chandapura', 'Begur Road', 'Gottigere',
        'Hulimavu', 'Arekere', 'Kudlu Gate', 'Hormavu', 'Kasturi Nagar',
        'Mahadevapura', 'Old Madras Road', 'Vijayanagar', 'Basaveshwara Nagar',
        'Domlur', 'Richmond Town', 'Sadashiva Nagar', 'Vasanth Nagar',
        'Cunningham Road', 'Lavelle Road', 'MG Road', 'Brigade Road',
        'Ulsoor', 'Cox Town', 'Fraser Town', 'Benson Town',
        'RT Nagar', 'HBR Layout', 'Kalyan Nagar', 'Kammanahalli',
        'HRBR Layout', 'Lingarajapuram', 'Banaswadi', 'CV Raman Nagar',
        'New BEL Road', 'Jalahalli', 'Vidyaranyapura', 'Jakkur',
        'Devanahalli', 'Bagalur', 'Budigere Cross', 'Chikkaballapur Road',
        'Attibele', 'Anekal', 'Jigani', 'Bommasandra',
        'Narayanapura', 'Horamavu Agara', 'Rachenahalli', 'Thanisandra Main Road',
        'Kogilu', 'Dasarahalli', 'Laggere', 'Magadi Road',
        'Mysore Road', 'Nayandahalli', 'Rajarajeshwari Nagara', 'Kengeri',
        'Kumaraswamy Layout', 'Padmanabhanagar', 'Girinagar', 'Basavanagudi',
        'Chamrajpet', 'Wilson Garden', 'Adugodi', 'Ejipura',
    ]
    price_multiplier = {
        'Koramangala': 14000, 'Indira Nagar': 15000, 'MG Road': 18000,
        'Lavelle Road': 20000, 'Sadashiva Nagar': 16000, 'Domlur': 12000,
        'HSR Layout': 9500, 'BTM Layout': 8000, 'Jayanagar': 12000,
        'JP Nagar': 8500, 'Malleshwaram': 13000, 'Rajaji Nagar': 11000,
        'Whitefield': 6500, 'Sarjapur Road': 6000, 'Electronic City': 5000,
        'Marathahalli': 7000, 'Bellandur': 7500, 'Hebbal': 9000,
        'Yelahanka': 6000, 'Bannerghatta Road': 6500, 'Thanisandra': 5500,
        'Old Airport Road': 10000, 'Richmond Town': 14000, 'Basavanagudi': 11000,
        'Cunningham Road': 16000, 'Brigade Road': 17000, 'Ulsoor': 11000,
        'Vasanth Nagar': 13000, 'Wilson Garden': 9500, 'Ejipura': 8500,
        'Devanahalli': 4500, 'Anekal': 3500, 'Chandapura': 3800,
    }
    default_price = 6000

    area_types = ['Super built-up  Area', 'Built-up  Area', 'Plot  Area', 'Carpet  Area']
    rows = []
    n = 13000
    for _ in range(n):
        loc = np.random.choice(locations)
        bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.30, 0.40, 0.20, 0.05])
        base_sqft = {1: 550, 2: 950, 3: 1400, 4: 2200, 5: 3500}[bhk]
        sqft = base_sqft + np.random.randint(-150, 400)
        sqft = max(sqft, 300)
        bath = min(bhk + np.random.choice([0, 0, 0, 1, -1]), bhk + 1)
        bath = max(bath, 1)
        balcony = min(np.random.choice([0, 1, 1, 2, 2, 3]), bhk)
        area_type = np.random.choice(area_types, p=[0.45, 0.30, 0.15, 0.10])
        mult = price_multiplier.get(loc, default_price)
        noise = np.random.normal(1.0, 0.15)
        price = (sqft * mult * noise) / 100000
        if area_type == 'Plot  Area':
            price *= 1.15
        price = max(price, 8)
        rows.append({
            'area_type': area_type,
            'availability': np.random.choice(['Ready To Move', 'Ready To Move', 'Ready To Move', '19-Dec', '22-Jan']),
            'location': loc,
            'size': f'{bhk} BHK',
            'society': np.nan if np.random.random() < 0.6 else f'Society_{np.random.randint(1, 200)}',
            'total_sqft': str(sqft) if np.random.random() < 0.85 else f'{sqft-50} - {sqft+50}',
            'bath': float(bath),
            'balcony': float(balcony),
            'price': round(price, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    print(f"Generated comprehensive dataset: {n} rows, {len(locations)} locations")
    return df

df = download_dataset()
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
print("\n--- DATA CLEANING ---")

df = df.drop(['area_type', 'society', 'availability'], axis=1, errors='ignore')
df = df.dropna()
print(f"After dropping nulls: {df.shape[0]} rows")

def extract_bhk(size):
    try:
        return int(size.split(' ')[0])
    except:
        return None

df['bhk'] = df['size'].apply(extract_bhk)
df = df.dropna(subset=['bhk'])
df['bhk'] = df['bhk'].astype(int)

def parse_sqft(x):
    try:
        if '-' in str(x):
            parts = str(x).split('-')
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(parse_sqft)
df = df.dropna(subset=['total_sqft'])

df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

df['location'] = df['location'].apply(lambda x: x.strip())
location_counts = df['location'].value_counts()
locations_less_than_10 = location_counts[location_counts < 10].index
df['location'] = df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)

print(f"Unique locations: {df['location'].nunique()}")
print(f"After cleaning: {df.shape[0]} rows")

# ---------------------------------------------------------------------------
# 3. OUTLIER REMOVAL
# ---------------------------------------------------------------------------
print("\n--- OUTLIER REMOVAL ---")

df = df[df['total_sqft'] / df['bhk'] >= 300]
df = df[df['bath'] <= df['bhk'] + 2]
df = df[df['price_per_sqft'] > 1000]
df = df[df['price_per_sqft'] < 50000]
df = df[df['price'] < 500]

print(f"After outlier removal: {df.shape[0]} rows")

# ---------------------------------------------------------------------------
# 4. SAVE LOCATION STATS FOR THE APP
# ---------------------------------------------------------------------------
loc_stats = df.groupby('location').agg(
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    count=('price', 'count'),
    avg_sqft=('total_sqft', 'mean'),
    avg_price_per_sqft=('price_per_sqft', 'mean'),
).reset_index()
loc_stats = loc_stats.sort_values('avg_price', ascending=False)

bhk_stats = df.groupby('bhk').agg(
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    count=('price', 'count'),
).reset_index()

price_ranges = {
    'low_threshold': float(df['price'].quantile(0.25)),
    'medium_threshold': float(df['price'].quantile(0.50)),
    'high_threshold': float(df['price'].quantile(0.75)),
    'max_price': float(df['price'].max()),
    'min_price': float(df['price'].min()),
    'mean_price': float(df['price'].mean()),
}

stats_data = {
    'location_stats': loc_stats.to_dict(orient='records'),
    'bhk_stats': bhk_stats.to_dict(orient='records'),
    'price_ranges': price_ranges,
    'all_locations': sorted(df['location'].unique().tolist()),
    'price_distribution': df['price'].describe().to_dict(),
    'total_records': len(df),
    'bhk_price_data': df[['bhk', 'total_sqft', 'price', 'location', 'bath', 'price_per_sqft']].to_dict(orient='records'),
}

with open('location_stats.json', 'w') as f:
    json.dump(stats_data, f, indent=2, default=str)
print("Location statistics saved to location_stats.json")

# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING & MODEL PREP
# ---------------------------------------------------------------------------
print("\n--- FEATURE ENGINEERING ---")

df_model = df.drop(['size', 'price_per_sqft'], axis=1)
dummies = pd.get_dummies(df_model['location'], drop_first=True, dtype=int)
df_model = pd.concat([df_model.drop('location', axis=1), dummies], axis=1)

X = df_model.drop('price', axis=1)
y = df_model['price'].values

feature_columns = list(X.columns)
with open('columns.json', 'w') as f:
    json.dump({'feature_columns': feature_columns, 'locations': sorted(df['location'].unique().tolist())}, f, indent=2)
print(f"Feature columns saved ({len(feature_columns)} features)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ---------------------------------------------------------------------------
# 6. BUILD AND TRAIN KERAS MODEL
# ---------------------------------------------------------------------------
print("\n--- MODEL TRAINING ---")

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ---------------------------------------------------------------------------
# 7. EVALUATE & SAVE
# ---------------------------------------------------------------------------
print("\n--- EVALUATION ---")
y_pred = model.predict(X_test, verbose=0).flatten()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"  MAE:  {mae:.2f} Lakhs")
print(f"  R2:   {r2:.4f}")

model.save('model.h5')
print("\nModel saved to model.h5")

config = {
    'mae': float(mae),
    'r2_score': float(r2),
    'epochs_trained': len(history.history['loss']),
    'features': len(feature_columns),
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'architecture': 'Dense(128)->BN->Dense(64)->BN->Dense(32)->Dense(1)',
}
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 65)
print("  TRAINING COMPLETE — All artifacts saved!")
print("=" * 65)
