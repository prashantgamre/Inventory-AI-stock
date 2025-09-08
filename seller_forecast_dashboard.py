import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import datetime
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from prophet import Prophet

load_dotenv()


st.logo(image="cropped-Vauch-Info-Logo-1-1-300x194.png", size="large", link="https://www.vauchinfotech.com", icon_image="cropped-Vauch-Info-Logo-1-1-300x194.png")

# Title and API Key Input
st.set_page_config(page_title="VAUCH Info Tech",page_icon="cropped-Vauch-Info-Logo-1-1-300x194.png", layout="wide")
st.title("📊 Chat with Your Data – AI Forecast Dashboard")

# API Key Input
api_key = st.sidebar.text_input("🔑 Enter Anthropic API Key", type="password")
if not api_key:
    st.warning("Please enter your Anthropic API key in the sidebar to enable AI features.")
    st.stop()

# Set the API key in environment
os.environ["ANTHROPIC_API_KEY"] = api_key

# Upload CSV
data_file = st.file_uploader("Upload 5-Year Sales CSV (2020–2024)", type=["csv"])
if not data_file:
    st.info("Please upload the sales data CSV file.")
    st.stop()

# Load and preprocess data
df = pd.read_csv(data_file)
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
products = df['Product Name'].unique()
regions = df['Region'].unique()
categories = df['Category'].unique()

# Filters
st.sidebar.header("🔍 Filters")
selected_products = st.sidebar.multiselect("Product", products, default=list(products))
selected_regions = st.sidebar.multiselect("Region", regions, default=list(regions))
selected_categories = st.sidebar.multiselect("Category", categories, default=list(categories))

# Filtered Data
filtered_df = df[(df['Product Name'].isin(selected_products)) &
                 (df['Region'].isin(selected_regions)) &
                 (df['Category'].isin(selected_categories))]

# KPIs
st.subheader("📈 Region & Category KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Total Units Sold", int(filtered_df['Units Sold'].sum()))
col2.metric("Total RFQs", int(filtered_df['RFQs Received'].sum()))
col3.metric("Distinct Products", len(filtered_df['Product Name'].unique()))

# Forecast per product
st.subheader("🔮 Forecast Insights (2025–2029)")
forecast_results = []
for product in selected_products:
    st.markdown(f"### ▶ {product}")
    prod_df = filtered_df[filtered_df['Product Name'] == product]
    monthly_df = prod_df.groupby('Month').agg({'Units Sold': 'sum'}).reset_index()
    monthly_df.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True)
    model.fit(monthly_df)
    future = model.make_future_dataframe(periods=60, freq='M')
    forecast = model.predict(future)
    forecast_only = forecast[forecast['ds'] > monthly_df['ds'].max()]
    actual_only = monthly_df[monthly_df['ds'] <= monthly_df['ds'].max()]

    # MAPE
    merged = forecast.merge(monthly_df, on='ds', how='left')
    mape = mean_absolute_percentage_error(merged.dropna()['y'], merged.dropna()['yhat'])
    
    # Actual Sales Chart
    st.markdown("#### 📊 Actual Sales (2020–2024)")
    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(x=actual_only['ds'], y=actual_only['y'], mode='lines+markers', name='Actual Sales'))
    fig_actual.update_layout(xaxis_title="Month", yaxis_title="Units Sold", height=300)
    st.plotly_chart(fig_actual, use_container_width=True)

    # Forecast Chart
    st.markdown("#### 🔮 Forecasted Sales (2025–2029)")
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast_only['ds'], y=forecast_only['yhat'], mode='lines+markers', name='Forecast'))
    fig_forecast.add_trace(go.Scatter(x=forecast_only['ds'], y=forecast_only['yhat_upper'], line=dict(width=0), mode='lines', showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=forecast_only['ds'], y=forecast_only['yhat_lower'], fill='tonexty', line=dict(width=0), mode='lines', showlegend=False))
    fig_forecast.update_layout(xaxis_title="Month", yaxis_title="Forecasted Units Sold", height=300)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown(f"**MAPE:** `{mape:.2%}`")
    st.info("This forecast uses Facebook Prophet to analyze trends from 2020–2024 and projects 60 months into the future based on seasonal patterns and historical demand.")

    forecast_summary = forecast_only[['ds', 'yhat']].rename(columns={'ds': 'Month', 'yhat': 'Forecasted Units Sold'})
    forecast_summary['Product Name'] = product
    forecast_results.append(forecast_summary)

    
# Initialize the model with error handling
try:
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.5)
    
    agent = create_pandas_dataframe_agent(
        model, 
        df=filtered_df, 
        verbose=True, 
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True
    )
except Exception as e:
    st.error(f"Error initializing the AI model: {str(e)}")
    st.stop()

CSV_PROMPT_PREFIX = """
First set the pandas display options to show all rows and columns,
get the column names, then answer the question.

Product.csv has : product details 
Inventory.csv has : inventory details such as how many product has in currently in stock
Sales.csv has : sales details such as how many product has been sold and at which date and time
Suppliers.csv has : supplier details

"""


CSV_PROMPT_SUFFIX = """
You are a helpful assistant that can answer questions about a CSV file.
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE**
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n"
In the explanation, mention the column names that you used to get
to the final answer."""

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

def set_button_clicked():
    st.session_state.button_clicked = True

with st.sidebar:
    st.write("### Question")
    question = st.text_input("Question:", "which product has the highest sales?")

# Button with a unique key
    if st.button("Ask", key="ask_button", on_click=set_button_clicked) or st.session_state.button_clicked:
        if question:  # Only process if there's a question
            with st.spinner('Analyzing...'):
                res = agent.invoke(CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX)
                st.write("### Answer")
                st.markdown(res["output"])
                st.session_state.button_clicked = False  # Reset the button state
        else:
            st.warning("Please enter a question.")

# Combined Forecast Table
    st.subheader("📋 Combined Forecast Table")
    forecast_df = pd.concat(forecast_results)
    forecast_df['Month'] = forecast_df['Month'].dt.strftime('%Y-%m')
    st.dataframe(forecast_df)
    st.download_button("⬇️ Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecast_summary.csv", mime="text/csv")



