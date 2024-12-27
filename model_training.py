import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import time, argparse, joblib
# Load the dataset
parser = argparse.ArgumentParser(description='Train a predictive maintenance model set.')
parser.add_argument('-dp','--dataset_path', type=str, required=True, help='Input CSV file path')
parser.add_argument('-mp','--model_path', type=str, required=True, help='Output model file path')
args = parser.parse_args()
data = pd.read_csv(args.dataset_path)  # Adjust path if needed

# Define features (X) and targets (Y)
X = data[['equipment_type', 'temperature_avg', 'humidity_avg',
          'last_maintenance_days', 'uptime_hours', 'sensor_1',
          'sensor_2', 'vibration', 'voltage_fluctuation', 'pressure']]
Y = data[['failure_probability', 'failure_reason', 'cost_implications', 'updated_uptime']]

# Encode categorical target column 'failure_reason'
Y = Y.copy()  # Avoid SettingWithCopyWarning
label_encoder = LabelEncoder()
Y['failure_reason'] = label_encoder.fit_transform(Y['failure_reason'])

# Define preprocessing steps for features
categorical_features = ['equipment_type']
numerical_features = ['temperature_avg', 'humidity_avg', 'last_maintenance_days',
                      'uptime_hours', 'sensor_1', 'sensor_2',
                      'vibration', 'voltage_fluctuation', 'pressure']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Separate targets
Y_fp = Y_train['failure_probability']
Y_ci = Y_train['cost_implications']
Y_uu = Y_train['updated_uptime']
Y_fr = Y_train['failure_reason']  # Classification task

# Define pipelines for separate targets
# 1. Model for failure_probability
fp_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# 2. Model for cost_implications
ci_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# 3. Model for updated_uptime
uu_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# 4. Model for failure_reason (classification)
fr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Train models
s = time.time()
fp_model.fit(X_train, Y_fp)
ci_model.fit(X_train, Y_ci)
uu_model.fit(X_train, Y_uu)
fr_model.fit(X_train, Y_fr)
print('Training Time: ',time.time()-s,"s")

# Evaluate models
s = time.time()

# Regression tasks
fp_pred = fp_model.predict(X_test)
ci_pred = ci_model.predict(X_test)
uu_pred = uu_model.predict(X_test)

# Classification task
fr_pred = fr_model.predict(X_test)

print('Inference Time: ',(time.time()-s))

# Metric calculation
fp_mse = mean_squared_error(Y_test['failure_probability'], fp_pred)
ci_mse = mean_squared_error(Y_test['cost_implications'], ci_pred)
uu_mse = mean_squared_error(Y_test['updated_uptime'], uu_pred)
fr_accuracy = accuracy_score(Y_test['failure_reason'], fr_pred)

# Results
print(f"Failure Probability MSE: {fp_mse}")
print(f"Cost Implications MSE: {ci_mse}")
print(f"Updated Uptime MSE: {uu_mse}")
print(f"Failure Reason Classification Accuracy: {fr_accuracy}")


# Save the models
joblib.dump(fp_model, args.model_path+"1.pkl")
joblib.dump(ci_model, args.model_path+"2.pkl")
joblib.dump(uu_model, args.model_path+"3.pkl")
joblib.dump(fr_model, args.model_path+"4.pkl")
print(f"Models saved to {args.model_path}")