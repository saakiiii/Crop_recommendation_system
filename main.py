import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
# data = pd.read_csv("Crop_recommendation.csv")

# # Preprocessing steps
# # Check for missing values
# if data.isnull().sum().sum() > 0:
#     imputer = SimpleImputer(strategy='mean')  # You can change this as necessary
#     data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

# # Scale the features
# scaler = StandardScaler()
# data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# # Split features and target variable
# X = data.drop(columns=['label'])
# y = data['label']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# # Predictions on the test set
# y_pred = rf_classifier.predict(X_test)

# # Evaluate the model
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Save the model and scaler
# joblib.dump(rf_classifier, 'rf_model.joblib')
# joblib.dump(scaler, 'scaler.joblib')

# To load the model and scaler later
loaded_rf_model = joblib.load('rf_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# Example prediction using loaded model
new_data = pd.DataFrame({'N': [93], 'P': [85], 'K': [49], 'temperature': [27.97], 'humidity': [79.29], 'ph': [5.7], 'rainfall': [119.48]})
new_data = loaded_scaler.transform(new_data)  # Scale new data
prediction = loaded_rf_model.predict(new_data)
print("Predicted Label:", prediction)
