# Summer-Olympic-Analysis-Prediction
 ~cleaned_summer_olympic_data-comp- The source dataset cleaned, modified and transformed to my preferences from Kaggle.Identified missing values-imputed-removed inconsistencies.
 
 ~app.py- The source code for all three analysis-Descriptive-Diagnostic-Predictive and a description about the dataset in 'about data' under the streamlit framework.

DESCRIPTIVE ANALYSIS:
Under descriptive analysis analysed athlete's consistency, medal analysis by country and year,ebvent-wise medal analysis, country's consitency and designed it in a chart to gain actionable insights.

DIAGNOSTIC ANALYSIS:
Under the diagnostic analysis, we identified the top 10 dominant countries across all years and events, visualizing their dominance through a chart. We then compared their GDP and population using a scatter plot to explore potential relationships.

To further investigate, we performed a correlation analysis using a correlation matrix and a heatmap to determine whether a country’s Olympic dominance is influenced by its GDP or population.
   KEY INSIGHT:
   This analysis highlights the need to explore other factors that might drive Olympic dominance, such as sports infrastructure, government investment, training facilities, and cultural emphasis on sports.

PREDICTIVE ANALYSIS:
   
   ~trained_models.py- The Random Forest model was employed for training and testing the dataset to predict future Olympic performance. Random Forest was chosen for its robust ability to handle complex patterns and its effectiveness in predictive tasks involving structured data.[85% accuracy]
   
   ~predictions.py -After training the model, predictions were saved to a CSV file for further analysis.[~future_medal_predictions_with_total.csv]
The predictions were then visualized alongside historical performance data, creating a comprehensive chart to show trends and future expectations.
