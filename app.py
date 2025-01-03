import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib  # Add this import
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go


# Streamlit app title
st.title("🏅 Olympic Data Analysis & Prediction")

# File upload
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format!")
        st.stop()

# Sidebar for navigation
selected_options = st.sidebar.radio(
    "Choose Analysis",
    ["About Data", "Descriptive Analysis", "Diagnostic Analysis", "Predictive Analysis"]
)

# Conditional logic for options
if selected_options == "About Data":
    st.title("About the Data")
    st.write("Display data details here.")
    

    # Convert Medal column to a binary value (1 if medal won, 0 if no medal)
    df['Medal_Won'] = np.where(df['Medal'] == 'No Medal', 0, 1)

    # Rename column 'Team' to 'Country'
    df = df.rename(columns={'Team': 'Country'})

    st.write(df.head())
    st.write("""
    This dataset contains detailed information about Olympic athletes, covering multiple **Summer Olympic Games**. 
    It includes athlete demographics, event participation, and medal achievements. 
    The dataset has been cleaned and structured to facilitate analysis.

    ### **Key Features:**
    - **Athlete Information:** ID, Name, Sex, Age, Height, Weight.
    - **Event Details:** Sport, Event, Games, Year, Season, City.
    - **Performance Metrics:** Medal status (Gold, Silver, Bronze, No Medal).
    - **Country-Level Data:** National Olympic Committee (NOC), Team, Country Code.
    - **Socioeconomic Indicators:** GDP and Population of the respective countries.

    This dataset enables diverse analyses, such as identifying dominant countries in specific sports, 
    examining athlete characteristics, and exploring socioeconomic factors influencing Olympic performance.
    """)

elif selected_options == "Descriptive Analysis":
    st.title("Descriptive Analysis")
    st.write("Explore trends and statistics about athletes, countries, and events in the Olympic dataset.")

    # Convert Medal column to a binary value (1 if medal won, 0 if no medal)
    df['Medal_Won'] = np.where(df['Medal'] == 'No Medal', 0, 1)

    # Rename column 'Team' to 'Country'
    df = df.rename(columns={'Team': 'Country'})

    # Step 1: Analyze consistency of athletes
    st.subheader("Athlete Consistency")
    athlete_medals_per_year = df.groupby(['Year', 'Name', 'Sport', 'Country'])['Medal_Won'].sum().reset_index()
    athlete_consistency = (
        athlete_medals_per_year[athlete_medals_per_year['Medal_Won'] > 0]
        .groupby(['Name', 'Sport', 'Country'])
        .agg(
            Years_Medaled=('Year', 'nunique'),
            Years_Won=('Year', list)
        )
        .reset_index()
    )

    # Dropdown to filter athletes by number of years medaled
    unique_years = list(athlete_consistency['Years_Medaled'].unique())
    unique_years.sort()
    unique_years.append("6 or more")
    selected_years = st.multiselect(
        "Filter Athletes by Years Medaled:",
        options=unique_years,
        default=[4, 5]
    )

    # Filter athlete consistency based on selection
    if "6 or more" in selected_years:
        filtered_athlete_consistency = athlete_consistency[
            (athlete_consistency['Years_Medaled'].isin(selected_years[:-1])) |
            (athlete_consistency['Years_Medaled'] >= 6)
        ]
    else:
        filtered_athlete_consistency = athlete_consistency[
            athlete_consistency['Years_Medaled'].isin(selected_years)
        ]

    # Display filtered athletes
    st.write("Athletes Matching Selected Criteria", filtered_athlete_consistency.head(60))

    # Step 2: Analyze total medals by country and year
    st.subheader("Medal Analysis by Country and Year")
    unique_countries = sorted(df['Country'].unique())
    countries_of_interest = st.multiselect(
        "Select Countries:",
        options=unique_countries,
        default=["India", "Sri Lanka", "Spain"]
    )

    unique_years = sorted(df['Year'].unique())
    year_of_interest = st.selectbox(
        "Select Year:",
        options=unique_years,
        index=len(unique_years) - 1
    )

    # Filter data based on country and year selection
    filtered_data = df[
        (df['Country'].isin(countries_of_interest)) &
        (df['Year'] == year_of_interest) &
        (df['Medal_Won'] == 1)
    ]

    # Medal Counts by Country
    medal_counts = filtered_data.groupby('Country').size().reset_index(name='Total_Medals')
    st.write(f"Total Number of Medals in {year_of_interest}", medal_counts)

    # Visualization 1: Bar Chart for Total Medals by Country
    st.subheader("Bar Chart: Total Medals by Country")
    if not medal_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(medal_counts['Country'], medal_counts['Total_Medals'], color=['blue', 'red', 'green'])
        ax.set_title(f"Total Medals for Countries in {year_of_interest}")
        ax.set_xlabel("Country")
        ax.set_ylabel("Total Medals")
        ax.set_xticks(range(len(medal_counts['Country'])))
        ax.set_xticklabels(medal_counts['Country'], rotation=45)
        st.pyplot(fig)
    else:
        st.write("No medal data available for the selected countries and year.")

    # Step 3: Heatmaps of medal counts by event and medal type
    st.subheader(f"Heatmaps: Medal Counts by Event and Medal Type in {year_of_interest}")
    filtered_data = filtered_data[filtered_data['Medal'] != "No Medal"]
    medal_counts_by_event = filtered_data.groupby(['Country', 'Event', 'Medal']).size().reset_index(name='Total_Medals')

    if not medal_counts_by_event.empty:
        fig, axes = plt.subplots(1, len(countries_of_interest), figsize=(16, 8))
        for i, country in enumerate(countries_of_interest):
            country_data = medal_counts_by_event[medal_counts_by_event['Country'] == country]
            country_pivot = country_data.pivot_table(index='Event', columns='Medal', values='Total_Medals', aggfunc='sum').fillna(0)
            country_pivot = country_pivot[country_pivot.sum(axis=1) > 0]
            if not country_pivot.empty:
                sns.heatmap(country_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=axes[i])
                axes[i].set_title(f"{country} Medal Counts")
                axes[i].set_xlabel("Medal Type")
                axes[i].set_ylabel("Event")
            else:
                axes[i].set_title(f"{country} Medal Counts (No Data)")
                axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No medal data available for the selected countries and year.")

    # Step 4: Visualization for specific events
    st.subheader("Event-Wise Medal Analysis")
    events_of_interest = sorted(df['Event'].unique())
    selected_events = st.multiselect(
        "Select Events",
        events_of_interest,
        default=["Basketball Men's Basketball", "Badminton Women's Singles"]
    )

    if selected_events:
        fig, axes = plt.subplots(len(selected_events), 2, figsize=(16, 8))
        for i, event_of_interest in enumerate(selected_events):
            event_data = df[(df['Event'] == event_of_interest) & (df['Medal'] != 'DNW') & (df['Medal'] != 'No Medal')]
            medal_counts = event_data.groupby(['Country', 'Medal']).size().unstack(fill_value=0).reset_index()
            medal_counts.set_index('Country').plot(kind='bar', stacked=True, color=['gold', 'silver', 'brown'], ax=axes[i, 0])
            axes[i, 0].set_title(f"Medal Counts by Country for {event_of_interest}")
            axes[i, 0].set_xlabel("Country")
            axes[i, 0].set_ylabel("Medal Count")
            axes[i, 0].set_xticklabels(medal_counts['Country'], rotation=45)
            axes[i, 0].legend(title="Medal Type", title_fontsize='13', loc='upper right')
            heatmap_data = medal_counts.set_index('Country').T
            sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=axes[i, 1])
            axes[i, 1].set_title(f"Heatmap of Medal Counts for {event_of_interest}")
            axes[i, 1].set_xlabel("Country")
            axes[i, 1].set_ylabel("Medal Type")
        plt.tight_layout()
        st.pyplot(fig)

        # Define countries and years available in the dataset for filtering
        # Define countries and years available in the dataset for filtering
    countries_of_interest = sorted(df['Country'].unique())
    years_of_interest = sorted(df['Year'].unique())

# Filters for country and year (directly on the page, not in the sidebar)
    st.subheader("Country Consistency over the years")
    selected_countries = st.multiselect(
      "Select Countries",
       countries_of_interest,
       default=['USA', 'India', 'China', 'Australia']
    )
    selected_years = st.multiselect(
      "Select Years",
       years_of_interest,
       default=[1972, 2000, 2016]
    )

# Filter the data based on selected countries and years
    filtered_data = df[
     (df['Country'].isin(selected_countries)) &
     (df['Year'].isin(selected_years)) &
     (df['Medal_Won'] == 1) &  # Only consider medals won
     (df['Medal'] != 'No Medal')  # Exclude "No Medal"
    ]

# Group the data by country, year, and medal type, and count the medals
    medal_counts = filtered_data.groupby(['Country', 'Year', 'Medal']).size().reset_index(name='Total_Medals')

# Create a figure for the plots
    fig, axes = plt.subplots(1, len(selected_years), figsize=(16, 6), sharey=True)  # Shared y-axis for better comparison

# Loop through the selected years and create a subplot for each
    for idx, year in enumerate(selected_years):
    # Filter the data for the current year
     year_data = medal_counts[medal_counts['Year'] == year]

    # Create a pivot table for easier plotting
     pivot_table = year_data.pivot_table(index='Country', columns='Medal', values='Total_Medals', fill_value=0)

    # Select the appropriate subplot
     ax = axes[idx] if len(selected_years) > 1 else axes

    # Create a scatter line plot for total medals by country
     for medal in pivot_table.columns:
        ax.plot(pivot_table.index, pivot_table[medal], marker='o', label=medal)

     ax.set_title(f'Total Medals for {year} by Country')
     ax.set_xlabel('Country')
     ax.set_ylabel('Total Medals' if idx == 0 else "")  # Add ylabel only for the first plot
     ax.legend(title='Medal Type', fontsize='small')
     ax.tick_params(axis='x', rotation=45)
     ax.grid()

# Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)  # Display the plot in Streamlit

   


elif selected_options == "Diagnostic Analysis":
    st.title("Diagnostic Analysis")
    st.write("Include diagnostic analysis visuals.")

    # Convert Medal column to a binary value (1 if medal won, 0 if no medal)
    df['Medal_Won'] = np.where(df['Medal'] == 'No Medal', 0, 1)

    # Rename column 'Team' to 'Country'
    df = df.rename(columns={'Team': 'Country'})

    medals_won_df = df.dropna(subset=['Medal'])

    # Error handling for missing or invalid GDP and Population data
    if 'GDP' not in medals_won_df.columns or medals_won_df['GDP'].isnull().all():
        st.error("GDP data is missing or invalid.")
    elif 'Population' not in medals_won_df.columns or medals_won_df['Population'].isnull().all():
        st.error("Population data is missing or invalid.")
    else:
        # Group by Country to count total medals won across all events
        country_total_medals = medals_won_df.groupby('Country').size().reset_index(name='Total_Medals')

        # Identify the top 10 dominant countries across all events
        top_dominant_countries_all = country_total_medals.nlargest(10, 'Total_Medals')

        # Visualize top dominant countries across all events (Bar Plot)
        fig_bar_all = px.bar(
            top_dominant_countries_all,
            x='Total_Medals',
            y='Country',
            orientation='h',  # Horizontal bar plot for better readability
            title='Top 10 Dominant Countries by Total Medals (All Events)',
            text='Total_Medals',
            color='Country',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar_all.update_layout(
            xaxis_title="Total Medals",
            yaxis_title="Country",
            legend_title="Country",
            font=dict(size=12)
        )
        fig_bar_all.update_traces(
            hovertemplate="<b>%{y}</b><br>Total Medals: %{x}<extra></extra>"
        )

        # Filter for top 10 dominant countries' data (for GDP and Population analysis)
        top_teams_all = top_dominant_countries_all['Country'].unique()
        top_countries_data_all = medals_won_df[medals_won_df['Country'].isin(top_teams_all)]

        # Calculate total medals, average GDP, and average population for these top countries
        aggregated_data_all = top_countries_data_all.groupby('Country').agg(
            Total_Medals=('Medal', 'count'),
            Avg_GDP=('GDP', 'mean'),
            Avg_Population=('Population', 'mean')
        ).reset_index()

        # Visualize GDP vs Total Medals for dominant countries (Scatter Plot)
        fig_scatter_all = px.scatter(
            aggregated_data_all,
            x='Avg_GDP',
            y='Total_Medals',
            size='Avg_Population',
            color='Country',
            hover_name='Country',
            log_x=True,  # Logarithmic scale for GDP
            size_max=60,
            title='Top Dominant Countries: Medals vs. GDP and Population (All Events)'
        )
        fig_scatter_all.update_layout(
            xaxis_title="Average GDP (Log Scale, in USD)",
            yaxis_title="Total Medals",
            legend_title="Country",
            font=dict(size=12)
        )
        fig_scatter_all.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>GDP: %{x}<br>Total Medals: %{y}<br>Population: %{marker.size}<extra></extra>"
        )

        # Display the visualizations
        st.subheader("Top 10 Dominant Countries by Total Medals")
        st.plotly_chart(fig_bar_all, use_container_width=True)

        st.subheader("GDP vs. Total Medals for Dominant Countries")
        st.plotly_chart(fig_scatter_all, use_container_width=True)

        # Correlation analysis
        st.subheader("Correlation Analysis")
        correlation_data = aggregated_data_all[['Total_Medals', 'Avg_GDP', 'Avg_Population']]
        correlation_matrix = correlation_data.corr()

        # Display correlation matrix
        st.write("Correlation Matrix:")
        st.dataframe(correlation_matrix)

        # Visualize the correlation matrix
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

        st.title("Conclusion of Diagnostic Analysis")

        st.write("""
        In this analysis, we identified the **top 10 dominant countries** across all years and events, visualizing their dominance through a **chart**. We then compared their **GDP and population** using a **scatter plot** to explore potential relationships.

        To further investigate, we performed a **correlation analysis** using a **correlation matrix** and a **heatmap** to determine whether a country’s Olympic dominance is influenced by its GDP or population.

        ### **Key Findings:**
        - Surprisingly, the results did not align with initial expectations.
        - There is **no significant correlation** between a country's Olympic dominance and its GDP or population.
        - The correlation values indicate weak or negative relationships, suggesting that factors beyond economic strength and population size contribute to Olympic success.

        This analysis highlights the need to explore **other factors** that might drive Olympic dominance, such as **sports infrastructure, government investment, training facilities, and cultural emphasis on sports**.
        """)



elif selected_options == "Predictive Analysis":
    st.subheader("Predictive Analysis")
    st.write("Predict medals for a selected country in 2028.")

    

# Step 1: Load the predictions data
    predictions_file_path = 'future_medal_predictions_with_total.csv'
    predictions_data = pd.read_csv(predictions_file_path)

# Ensure the file path for historical data is correct
    historical_data_path = 'cleaned_summer_olympics_data.csv'
    c_summer_olympics = pd.read_csv(historical_data_path)
    c_summer_olympics = c_summer_olympics.rename(columns={'Team': 'Country'})



    # Dropdown menu for country selection
    sorted_countries = predictions_data['Country'].sort_values().unique()
    selected_country = st.selectbox("Select a country", sorted_countries)

    # Filter historical data for the selected country
    historical_data = c_summer_olympics[
        (c_summer_olympics['Country'] == selected_country) & (c_summer_olympics['Medal'] != 'No Medal')
    ]
    historical_medals = historical_data.groupby('Year').size().reset_index(name='Total_Medals')

    # Filter predictions for the selected country
    prediction_data = predictions_data[predictions_data['Country'] == selected_country]
    if prediction_data.empty:
        st.write("No prediction data available for the selected country.")
    else:
        expected_medals_2028 = prediction_data['Expected_Medals'].values[0]
        medal_probability = prediction_data['Medal_Prob'].values[0]

        # Display the predictions in text
        st.write(f"### Predicted Medal Details for {selected_country} in 2028")
        st.write(f"- **Expected Total Medals**: {expected_medals_2028:.2f}")
        st.write(f"- **Predicted Probability of Winning Medals**: {medal_probability:.2f}")

        # Filter historical data for the selected country
        historical_data = c_summer_olympics[
            (c_summer_olympics['Country'] == selected_country) & (c_summer_olympics['Medal'] != 'No Medal')
        ]
        historical_medals = historical_data.groupby('Year').size().reset_index(name='Total_Medals')

        # Create a prediction DataFrame
        prediction_df = pd.DataFrame({'Year': [2028], 'Total_Medals': [expected_medals_2028]})

        # Concatenate historical and prediction data
        full_data = pd.concat([historical_medals, prediction_df], ignore_index=True)

        # Create Plotly Visualization
        fig = go.Figure()
        

        # Historical data (blue line and markers)
        fig.add_trace(go.Scatter(
            x=historical_medals['Year'],
            y=historical_medals['Total_Medals'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue'),
            marker=dict(color='blue', size=8)
        ))

        # 2028 prediction (red marker)
        fig.add_trace(go.Scatter(
            x=prediction_df['Year'],
            y=prediction_df['Total_Medals'],
            mode='markers',
            name='2028 Prediction',
            marker=dict(color='red', size=10, symbol='x')
        ))

        # Prediction line (dotted line from last historical year to 2028)
        last_historical_year = historical_medals['Year'].max()
        last_historical_medals = historical_medals.loc[historical_medals['Year'] == last_historical_year, 'Total_Medals'].values[0]

        fig.add_trace(go.Scatter(
            x=[last_historical_year, 2028],
            y=[last_historical_medals, expected_medals_2028],
            mode='lines',
            name='Prediction Line',
            line=dict(color='orange', dash='dot')
        ))

        # Step 3: Customize Plot Layout
        fig.update_layout(
            title=f"Historical Performance and 2028 Medal Prediction for {selected_country}",
            xaxis_title="Year",
            yaxis_title="Total Medals",
            xaxis=dict(
                tickmode='linear',
                tick0=historical_medals['Year'].min(),
                dtick=4
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )

        # Step 4: Display the Plot
        st.plotly_chart(fig, use_container_width=True)
        st.title("Prediction Workflow and Chart Explanation")

        st.write("""
### **Prediction Workflow and Chart Explanation**

#### **Model Used**  
The **Random Forest model** was employed for training and testing the dataset to predict future Olympic performance. Random Forest was chosen for its robust ability to handle complex patterns and its effectiveness in predictive tasks involving structured data.

#### **Data Handling**  
- After training the model, **predictions were saved to a CSV file** for further analysis.
- The predictions were then visualized alongside **historical performance data**, creating a comprehensive chart to show trends and future expectations.

#### **Chart Details**  
The chart above displays the **historical medal performance** of India in the Olympics and the **predicted medal count for 2028**.  
- **Blue Line:** Represents the historical performance trend.  
- **Red Marker:** Highlights the **2028 prediction**.  
- **Dashed Line:** Indicates the predicted trajectory beyond the latest available data.  

  

This predictive analysis enables a data-driven understanding of future expectations and highlights areas where further focus is needed to enhance performance.
""")


else:
    st.warning("Please upload a dataset to proceed.")
