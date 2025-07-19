import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv('c2_intent_dataset.csv')

df = df.drop(columns=['src_ip', 'dst_ip'], errors='ignore')

if df['protocol'].dtype == object:
    le = LabelEncoder()
    df['protocol'] = le.fit_transform(df['protocol'])
    joblib.dump(le, 'protocol_encoder.pkl')  

X = df.drop(columns=['intent'])
y = df['intent']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

y_pred = clf.predict(X_test)

print("\n Model, scaler, and encoder saved successfully!")