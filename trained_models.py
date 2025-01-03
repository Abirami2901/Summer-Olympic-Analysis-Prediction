import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from prettytable import PrettyTable
import pickle

# Load the cleaned dataset
file_path = 'D:/streamlit-olympic/cleaned_summer_olympics_data.csv'
olympics_data = pd.read_csv(file_path)

olympics_data = olympics_data.rename(columns={'Team': 'Country'})


# Create the 'Medal_Won' column based on the 'Medal' column
olympics_data['Medal_Won'] = olympics_data['Medal'].apply(lambda x: 1 if x in ['Gold', 'Silver', 'Bronze'] else 0)

# Select features and target variable
features = ['Age', 'Height', 'Weight', 'Country', 'Year', 'Sport', 'GDP', 'Population']
target = 'Medal_Won'

# Split data into features (X) and target (y)
X = olympics_data[features]
y = olympics_data[target]

# Define a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), ['Age', 'Height', 'Weight', 'GDP', 'Population']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country', 'Sport'])
    ])

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    # Combine preprocessing and model in a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save the trained pipeline model to a file (in Colab)
    with open('D:/streamlit-olympic/random_forest_model_new.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

    # Predict on the test data
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Store the results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'ROC AUC': roc_auc
    }

# Display results in a table format
table = PrettyTable()
table.field_names = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "ROC AUC (%)"]

for model_name, metrics in results.items():
    table.add_row([
        model_name,
        f"{metrics['Accuracy'] * 100:.2f}",
        f"{metrics['Precision'] * 100:.2f}",
        f"{metrics['Recall'] * 100:.2f}",
        f"{metrics['ROC AUC'] * 100:.2f}"
    ])

# Print the table
print(table)
