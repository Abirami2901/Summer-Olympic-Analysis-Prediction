import pandas as pd
import pickle

# Load the cleaned dataset
file_path = 'D:/streamlit-olympic/cleaned_summer_olympics_data.csv'
olympics_data = pd.read_csv(file_path)
olympics_data = olympics_data.rename(columns={'Team': 'Country'})  # Ensure 'Country' column is present

# Function to predict medal probabilities by country
def predict_future_medals_by_country(countries):
    # Load the trained model (Random Forest) from file
    with open('random_forest_model_new.pkl', 'rb') as file:
        pipeline = pickle.load(file)

    # Prepare the data: calculate historical averages for numeric features
    historical_averages = olympics_data.groupby('Country')[['Age', 'Height', 'Weight', 'GDP', 'Population']].mean().reset_index()

    # Filter for the specified countries
    future_data = historical_averages[historical_averages['Country'].isin(countries)].copy()

    # Add placeholder columns for Year and Sport for future predictions
    future_data['Year'] = 2028  # Set the year to 2028 for predictions
    future_data['Sport'] = 'Unknown'  # Placeholder for Sport

    # Apply the preprocessing pipeline to prepare the data
    X_preprocessed = pipeline.named_steps['preprocessor'].transform(
        future_data[['Age', 'Height', 'Weight', 'GDP', 'Population', 'Year', 'Sport', 'Country']]
    )

    # Predict medal probabilities for the future data
    future_data['Medal_Prob'] = pipeline.named_steps['classifier'].predict_proba(X_preprocessed)[:, 1]

    # Aggregate predicted probabilities by country
    country_probabilities = future_data[['Country', 'Medal_Prob']].groupby('Country').sum().reset_index()
    country_probabilities = country_probabilities.sort_values(by='Medal_Prob', ascending=False)

    return country_probabilities

# Function to calculate the number of athletes in the most recent year (e.g., 2016)
def get_athlete_count_by_country():
    # Filter data for the most recent year (e.g., 2016)
    latest_year_data = olympics_data[olympics_data['Year'] == 2016]

    # Count unique athletes per country in 2016
    athlete_counts = latest_year_data.groupby('Country')['ID'].nunique().reset_index()
    athlete_counts.rename(columns={'ID': 'Athlete_Count'}, inplace=True)
    return athlete_counts

# Get all unique countries from the dataset
all_countries = olympics_data['Country'].unique()

# Predict future medals for all countries
future_predictions = predict_future_medals_by_country(all_countries)

# Get athlete count by country for the year 2016
athlete_counts = get_athlete_count_by_country()

# Merge the predicted probabilities with athlete counts
future_predictions_with_medals = pd.merge(future_predictions, athlete_counts, on='Country', how='left')

# Calculate expected medals based on predicted probabilities and athlete counts
future_predictions_with_medals['Expected_Medals'] = (
    future_predictions_with_medals['Medal_Prob'] * future_predictions_with_medals['Athlete_Count']
)

# Save the predictions to a CSV file
output_file_path = 'future_medal_predictions_with_total.csv'
future_predictions_with_medals.to_csv(output_file_path, index=False)

print(f"Future medal predictions with total medals saved to {output_file_path}")

# Optionally, print the first few rows to verify the output
print(future_predictions_with_medals.head())
