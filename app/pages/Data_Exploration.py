import folium
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import pickle
import numpy as np
from streamlit_folium import folium_static
import plotly.graph_objects as go
import os

CURRENT_PATH = os.path.dirname(__file__)
st.set_page_config(
    page_title="Software Developer Salary Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
   
)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to top, #00008b, #4b0082);
        opacity: 0.7;
        color: white;
    }
    [data-testid="stSidebar"] .sidebar-content {
        
    }
    </style>
    """, unsafe_allow_html=True)

# Load the data
file_path = os.path.join(CURRENT_PATH, 'survey_results_public.csv')
df = pd.read_csv(file_path)
selected_columns = ['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']
df = df[selected_columns].rename(columns={'ConvertedCompYearly': 'Salary'})
df = df.dropna()
df = df[df['Employment'] == 'Employed, full-time']
df = df.drop(columns=['Employment'])

# Function to clean experience data
def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

# Function to clean education data
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)

# Sidebar for user input
st.sidebar.title("Customize Visualization")
cutoff_value = st.sidebar.slider("Choose Cutoff Value for Country Count", 0, 1000, (100), step=10)

df_counts = df['Country'].value_counts()
countries_below_cutoff = df_counts[df_counts < cutoff_value].index
df['Country'] = df['Country'].apply(lambda x: 'Other' if x in countries_below_cutoff else x)


salary_value = st.sidebar.slider("Choose Cutoff Value for Salary Range", 0, 250000, (10000, 250000), step=1000)
selected_countries = st.sidebar.multiselect("Select Countries to Include", df['Country'].unique())

df_counts = df['Country'].value_counts()
countries_below_cutoff = df_counts[df_counts < cutoff_value].index
df['Country'] = df['Country'].apply(lambda x: 'Other' if x in countries_below_cutoff else x)
df_filtered = df[(df['Salary'] >= salary_value[0]) & (df['Salary'] <= salary_value[1])]


if selected_countries:
    df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]

st.title("Exploratory Data Analysis")
st.header("Introduction")
######################################################################
#################### Visualizing the data ############################
######################################################################
st.header("A First Look at Our Data")
st.write('''We have access to an annual survey by Stack Overflow where we can find more than 67K software developers' answers. Part of the survey includes salary, experience, country, full/partial time employee, language, demographic information and so on. 
         After processing and cleaning the data, here we have 50 samples of what our final dataset looks like. In the following sections you will be able to deep dive into the specifics of this dataset, and you may be able to find insightful patterns.''')
st.dataframe(df.sample(50), use_container_width=True) 

######################################################################
#################### DISTRIBUTION PLOTS ##############################
######################################################################

# Visualization: Number of Records per Country
st.header("Distributions")
st.markdown("**The main insights to take from the collection of plots within this section are related to how all the different variables are distributed. That is, the counts of records of each particular variable. That can be also seen as the number of developers for each distinct value of a variable.**")


st.subheader("Developers Distribution per Country")
st.write("This first chart illustrates the distribution of developer counts by country. We can see that we have mostly, data of software developers in USA. The next most common country after that is shown to be 'Others' which are all the aggregated countries with lower count than the set cut-off value. (It can be changed on the side bar)")

counts_by_country = df['Country'].value_counts().reset_index()
counts_by_country.columns = ['Country', 'Count']
counts_by_country = counts_by_country.sort_values(by='Count', ascending=False)

chart = alt.Chart(counts_by_country).mark_bar().encode(
    alt.X("Country:N", title='Country', sort=alt.EncodingSortField(field="Count", order='descending')),
    alt.Y('Count:Q', title='Number of Developers'),
    tooltip=['Count:Q', alt.Tooltip('Country:N', title='Country')],
    color=alt.Color('Count:Q', scale=alt.Scale(scheme='viridis'), title='Number of Developers')
).properties(
    width=600,  
    height=400 
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='middle'
)
st.altair_chart(chart, use_container_width=True)


## Visualization 2
st.subheader("Developers Distribution per Experience")
st.write("This chart depicts the distribution of professional coding experience. It is interesting to see that most of the software developers in our dataset have between 1-10 years of experience, which it can be considered to be quite low. However, we must take into account that it is a relatively new profession, so it makes sense that there isn't a huge amount of very experienced software developrs.")

chart = alt.Chart(df_filtered).mark_bar().encode(
    alt.X("YearsCodePro:Q", bin=alt.Bin(maxbins=50), title='Years of Professional Coding Experience'),
    alt.Y('count():Q', title='Number of Developers'),
    tooltip=['count():Q', alt.Tooltip('YearsCodePro:Q', bin=alt.Bin(maxbins=50), title='Years of Experience')],
    color=alt.Color('count():Q', scale=alt.Scale(scheme='viridis'), title='Frequency')
).properties(
    width=600,  
    height=400  
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='middle'
)
st.altair_chart(chart, use_container_width=True)


# Visualization 3
# Visualization 3: Distribution of Education Level
st.subheader("Developers Distribution per Education Level")
st.write("This chart displays the distribution of education levels among developers. We can easily observe how the most common degree is a Bachelor's degree. It makes sense that there are less people with Master's and Post grad, since those usually come later, and it is common that after finishing a Bachelor's degree, the students want to work already. What is surprising though, is that there is a considerable amount of people without a Bachelor's degree that are working as softweare developers. This means, that there are a lot of self-taught people working in this field.")

chart_education = alt.Chart(df_filtered).mark_bar().encode(
    alt.X("EdLevel:N", title='Education Level'),
    alt.Y('count():Q', title='Number of Developers'),
    tooltip=['count():Q', alt.Tooltip('EdLevel:N', title='Education Level')],
    color=alt.Color('count():Q', scale=alt.Scale(scheme='viridis'), title='Number of Developers')
).properties(
    width=600,  
    height=400 
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='middle'
)
st.altair_chart(chart_education, use_container_width=True)

st.subheader("Developers Distribution per Salary Range")
st.write("This last chart of the section, displays the distribution of salaries accross developers in every country. We can observe that the amount of developers earning between 10k and 80k is more less the same, with a peak at around 60k. We see that the number developers earning more than that plumets fastly.")
# Define fewer salary bins
salary_bins = list(range(salary_value[0], salary_value[1], 1000))

# Create a histogram chart with fewer bins ordered by salary
chart_salary_counts = alt.Chart(df_filtered).mark_bar().encode(
    alt.X("Salary:O", title='Salary Range', bin=alt.Bin(step=2000), sort=salary_bins),
    alt.Y('count():Q', title='Number of Developers'),
    tooltip=['count():Q'],
    color=alt.Color('count():Q', scale=alt.Scale(scheme='viridis'), title='Number of Developers')
).properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='middle'
)

# Display the chart using Streamlit
st.altair_chart(chart_salary_counts, use_container_width=True)


######################################################################
#################### SALARY PLOTS ####################################
######################################################################


# Visualization: Salary Distribution per Country (Ordered by Mean Salary)
st.header('Insights by Salary')
st.markdown("**The main insights to take from the collection of plots within this section are related to how all the different variables are related to the salary distribution.**")

st.subheader("Salary Distribution per Country")
st.write("This box plot visualizes the salary distribution per country, ordered by mean salary.")
st.write("Note that in this boxplot, we can only see the distribution of salaries between the values set in the side bar. With salaries between 10k and 250k, we can get further insight. Firstly, we can see that most countries fall below the mean salary value over all countries. We still see a large amount of outliers in all the countries, which makes sense, because there will always be very valuable software developers in almost every country. We can see that countries with the top-earning software developers are USA, Israel, Switzerland and Canada, which makes sense because those countries are ahead in the race for technological advancement and invest a lot of capital in these sectors. On the other side, we observe that countries with the lowest paid salaries tend to be under developed countries, where there hasn't been a big digital revolution compared to the other countries. In some sense, this plot can also give insights about the state of digitalization in every country.")

mean_salary_by_country = df_filtered.groupby('Country')['Salary'].mean().sort_values(ascending=False).index
fig2 = px.box(df_filtered, x='Country', y='Salary', color='Country',
              
              category_orders={'Country': mean_salary_by_country})
mean_salary = df_filtered['Salary'].mean()
fig2.add_shape(
    dict(
        type='line',
        yref='y',
        y0=mean_salary,
        y1=mean_salary,
        x0=-0.5,
        x1=len(mean_salary_by_country) - 0.5,
        line=dict(color='red', width=2, dash='dash')
    )
)
fig2.update_layout(xaxis_title='Country', yaxis_title='Salary', width=1600, height=800, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig2, use_container_width=True)

# Visualization 6: Salary Distribution
st.subheader("Salary-Country Distribution Chart")
st.write("This bar chart displays the mean salary by country. The insights obtained are mostly the same as the plot above. This plot just serves as a complement to the boxplot.")
df_mean_salaries = df_filtered.groupby('Country')['Salary'].mean().reset_index()
df_mean_salaries = df_mean_salaries.sort_values('Salary', ascending=False)
sorted_countries = df_mean_salaries['Country'].tolist()

chart_country_salary = alt.Chart(df_filtered).mark_bar().encode(
    alt.X('Country:N', title='Country', sort=sorted_countries),
    alt.Y('mean(Salary):Q', title='Mean Salary'),
    tooltip=['mean(Salary):Q', alt.Tooltip('Country:N', title='Country')],
    color=alt.Color('mean(Salary):Q', scale=alt.Scale(scheme='viridis'), title='Mean Salary')
).properties(
    width=600,  # Adjust the width as needed
    height=400  # Adjust the height as needed
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='middle'
)
st.altair_chart(chart_country_salary, use_container_width=True)


# Visualization 6: Salary Distribution by Education Level
st.write("In this chart, we can see how much money do different software developers with different education levels are currently earning. Firstly, we observe that software developers witha post grad degree are the top earners. However, something that is interesting is that Software developers with just a Bachelor's degree are earning more than those with a Mater's degree. That might be because those students that started working with a Bachelir's degree, started earlier than those with a Master's degree and ended up having more experience. Perhaps, in this field, experinece is more invaluable than having an extra degree.")
st.subheader("Salary-Education Bar chart")
st.write("This heatmap shows the mean salary by education level.")
df_mean_salaries = df_filtered.groupby('EdLevel')['Salary'].mean().reset_index()
df_mean_salaries = df_mean_salaries.sort_values('Salary', ascending=False)
sorted_edlevels = df_mean_salaries['EdLevel'].tolist()

chart_education_salary = alt.Chart(df_filtered).mark_rect().encode(
    alt.X('EdLevel:N', title='Education Level', sort=sorted_edlevels),
    alt.Y('mean(Salary):Q', title='Mean Salary'),
    alt.Color('mean(Salary):Q', scale=alt.Scale(scheme='viridis'), title='Mean Salary'),
    alt.Size('mean(Salary):Q', title='Mean Salary')
).properties(
    width=400,  # Adjust the width as needed
    height=400  # Adjust the height as needed
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='middle'
)
st.altair_chart(chart_education_salary, use_container_width=True)


# Visualization: Mean Salary by Professional Coding Experience
st.subheader("Salary-Experience Scatter Plot")
st.write("This scatter plot illustrates the mean salary by professional coding experience. We can see a linear trend in the beggining, in regard to the first years of experience. However, we obnserve that after around 20 years of experience, salaries don't vary that much and seem to stabilize.")

chart_experience_salary = alt.Chart(df_filtered).mark_circle().encode(
    alt.X("YearsCodePro:Q", title='Years of Professional Coding Experience'),
    alt.Y('mean(Salary):Q', title='Mean Salary'),
    tooltip=['mean(Salary):Q', alt.Tooltip('YearsCodePro:Q', title='Experience')],
    size=alt.Size('count()', scale=alt.Scale(range=[50, 300]), title='Count'),  
    color=alt.Color('count()', scale=alt.Scale(scheme='oranges'), title='Count')
).properties(
    width=600,  
    height=400  
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='middle'
)
st.altair_chart(chart_experience_salary, use_container_width=True)


# Visualization MAP
st.subheader("Map of Mean Salaries by Country")
st.write("This map displays mean salaries by country. It serves as a more visual interpretation of salary distribution among different countries. For example we can see that software developers tend to earn more in Northern Europe than in Southern Europe.")
file_geo = os.path.join(CURRENT_PATH, 'geo_loc_df.csv')
df_geo_loc = pd.read_csv(file_geo)
map_center = [df_geo_loc['lat'].mean(), df_geo_loc['lon'].mean()]
my_map = folium.Map(location=map_center, zoom_start=2)
mean_salary_by_country = df_geo_loc.groupby('Country')['Salary'].mean().reset_index()
# Iterate through the DataFrame and add markers to the map
for index, row in mean_salary_by_country.iterrows():
    # Adjust the size of the marker based on the mean salary
    marker_size = row['Salary'] / mean_salary_by_country['Salary'].max() * 10  # Adjust the multiplier as needed
    
    folium.CircleMarker(
        location=[df_geo_loc.loc[df_geo_loc['Country'] == row['Country'], 'lat'].mean(),
                  df_geo_loc.loc[df_geo_loc['Country'] == row['Country'], 'lon'].mean()],
        radius=marker_size,
        color='red',
        fill=False,
        fill_color='red'
    ).add_to(my_map)

folium_static(my_map, width = 1400, height=600)

######################################################################
#################### CROSSED VARIABLES PLOTS #########################
######################################################################
st.header("Cross-variable Insights")
st.markdown("**The main insights to take from the collection of plots within this section are related to how all the different variables behave and are related to each other, without considering the independent variable used later in the predictive models  -the salary-**")
st.subheader("Country-Education Circle Plot")
st.write("This scatter plot visualizes the relationship between country and education level. That helps us see the amount of developers with different education levels in each country.")

chart = alt.Chart(df_filtered).mark_circle().encode(
    alt.X('Country:N', title='Country'),
    alt.Y('EdLevel:N', title='Education Level'),
    size=alt.Size('count()', scale=alt.Scale(range=[50, 300])),
    color=alt.Color('count():Q', scale=alt.Scale(scheme='oranges'), title='Number of Developers'),
    tooltip=['Country:N', 'EdLevel:N', 'count():Q']
).properties(
    width=600,
    height=400,
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Country-Experience Polar Chart")
st.write("This polar bubble plot represents the mean years of experience by country. It helps us see what countries tend to hire software developers with more years of experience against which countries tend to hire developers with less experience. We see that New Zeland is, for some reson, the country that has software developers with the highest mean salary, followed by Australia, which is the neighbour country. That can be interpreted that either there were a lot of software developers in that area a few years ago, and they stayed in their countries. Another worrying perspective could be that all the newly graduated software developers are either not finding a job in those countries or are migrating to other countries. It would definetely be interesting to do a further study to understand the situation")

# Group by country and calculate mean years of experience
mean_experience_by_country = df_filtered.groupby('Country')['YearsCodePro'].mean().reset_index()
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=mean_experience_by_country['YearsCodePro'],
    theta=mean_experience_by_country['Country'],
    mode='markers',
    marker=dict(
        size=mean_experience_by_country['YearsCodePro'] * 2.3, 
        color=mean_experience_by_country['YearsCodePro'],
        colorscale='viridis',
        showscale=True
    ),
    text=mean_experience_by_country['Country'],
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True),
    ),
    showlegend=False,
    width=1200 
)

st.plotly_chart(fig, use_container_width=True)


## Correlation Visual
file_path_model = os.path.join(CURRENT_PATH, 'saved_models.pkl')
with open(file_path_model, 'rb') as file:
    data = pickle.load(file)

le_country = data["le_country"]
le_education = data["le_education"]

df_filtered['Country'] = le_country.transform(df_filtered['Country'])
df_filtered['EdLevel'] = le_education.transform(df_filtered['EdLevel'])


# Visualization: Correlation Matrix
st.subheader("Correlation Matrix")
st.write("This heatmap displays the correlation matrix between variables. We can see that the highest correlation is between salary-country (38% of correlation), and salary-experience(37% of correlation). We can see then that the least correlated variable with salary is the education level, which can give us a lot of information about how salaries are stablished -education level is not as important as we might think-.")


corr_matrix = df_filtered.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

corr_flat = corr_matrix.stack().reset_index(name='correlation').rename(columns={'level_0': 'variable1', 'level_1': 'variable2'})
heatmap = alt.Chart(corr_flat).mark_rect().encode(
    alt.X('variable2:N', title=None),
    alt.Y('variable1:N', title=None),
    color=alt.Color('correlation:Q', scale=alt.Scale(scheme='tealblues'), title='Correlation'),
    tooltip=['variable1:N', 'variable2:N', 'correlation:Q']
).properties(
    width=400,  
    height=400  
).transform_filter(
    (alt.datum.variable1 != alt.datum.variable2) | (alt.datum.variable1 == alt.datum.variable2)
)
st.altair_chart(heatmap, use_container_width=True)
