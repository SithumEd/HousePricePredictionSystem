import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_excel('dataset.xlsx')

label_encoder_location = LabelEncoder()
label_encoder_amenities = LabelEncoder()
df['Location'] = label_encoder_location.fit_transform(df['Location'])
df['Amenities'] = label_encoder_amenities.fit_transform(df['Amenities'])

X = df[['Acres', 'Rooms', 'Location', 'Amenities']]

y = df['Predicted Price in 6-12 Months (in $)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder_location, 'label_encoder_location.pkl')
joblib.dump(label_encoder_amenities, 'label_encoder_amenities.pkl')
