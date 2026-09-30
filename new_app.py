
#  These libraries are imported for data handling, dashboard development, model training, and model evaluation.


import pandas as pd     # dataset/CSV ko read aur data ko handle karne ke liye use hota hai.
import streamlit as st   #  Streamlit se interactive web dashboard banate hain.

from sklearn.model_selection import train_test_split  # Dataset ko training aur testing data mein divide karne ke liye.
from sklearn.linear_model import LinearRegression   # # Linear Regression machine-learning model import karta hai, jo daily orders predict karega.

from sklearn.metrics import mean_absolute_error, r2_score   # Model ki performance check karne ke liye MAE aur R² Score import karta hai.

# PAGE SETTINGS

st.set_page_config(     # webpage ki settings set karta hai.
    page_title="Daily Orders Analytics",    # browser tab ka naam set karta hai.
    page_icon="🛒",   # shopping cart icon show kary ga
    layout="wide"   # screen ka lyout full wide hoga
)


# CUSTOM STYLE

# st.markdown() ke through hum CSS code Streamlit page mein apply kar rahe hain.

# <style> ke andar CSS styling likhi ja rahi hai.

st.markdown("""     
<style>
                                 
/* Main Background */      
.stApp {
    background: linear-gradient(
        135deg,
        #F5F7F2 0%,
        #EEF3EC 50%,
        #FFFFFF 100%
    );
}


/* Main Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #496A5A !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #385445 !important;
    color: white !important;
}


/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.92);
    border: 1px solid #DCE5DD;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(73, 106, 90, 0.08);
}


/* Metric Value */
div[data-testid="stMetricValue"] {
    color: #496A5A !important;
    font-weight: 800 !important;
}


/* Metric Label */
div[data-testid="stMetricLabel"] {
    color: #68736C !important;
}


/* Tabs */
button[data-baseweb="tab"] {
    color: #496A5A !important;
    font-weight: 600 !important;
}


/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F0F3EF;
}


/* Selectbox / Inputs */
div[data-baseweb="select"] {
    border-radius: 10px;
}


/* Success Message */
div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* Footer */
.footer-text {
    color: #7A857E;
    text-align: center;
    padding: 25px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# LOAD DATASET

df = pd.read_csv(
    "daraz_daily_orders_synthetic.csv"
)
# FEATURES AND TARGET


feature_columns = [
    "Website_Visitors",
    "Ad_Spend_PKR",
    "Discount_Percent",
    "Weekend"
]

X = df[feature_columns]

y = df["Daily_Orders"]


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# TRAIN LINEAR REGRESSION MODEL

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# TEST MODEL

test_predictions = model.predict(
    X_test
)

# MODEL EVALUATION

mae = mean_absolute_error(
    y_test,
    test_predictions
)

r2 = r2_score(
    y_test,
    test_predictions
)


# TEST RESULTS

test_results = X_test.copy()

test_results["Actual_Orders"] = y_test.values

test_results["Predicted_Orders"] = (
    test_predictions.round(1)
)

test_results["Residual"] = (
    test_results["Actual_Orders"]
    - test_results["Predicted_Orders"]
).round(1)


# COEFFICIENT TABLE


coefficient_table = pd.DataFrame({

    "Feature": feature_columns,

    "Coefficient": model.coef_.round(4)

})


# SIDEBAR


with st.sidebar:

    st.title("🛒 Orders Analytics")

    st.divider()

    st.subheader("Dashboard")

    st.info(
        "Explore your business data, uncover insights, "
        "and predict daily orders with AI."
    )

    st.subheader("Dataset")

    st.write(
        f"📁 Rows: **{len(df):,}**"
    )

    st.write(
        f"📊 Features: **{len(feature_columns)}**"
    )

    st.divider()

    st.caption(
        "AIcadamy 360 Classroom Project"
    )

# MAIN HEADER


st.caption(
    "Created by Sundas Waqas"
)

st.title(
    "🛒 Daily Orders Analytics"
)

st.write(
    "AI-powered business dashboard for daily order prediction"
)


# TOP KPI SECTION


avg_orders = df["Daily_Orders"].mean()

avg_visitors = df["Website_Visitors"].mean()

avg_ad_spend = df["Ad_Spend_PKR"].mean()


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        label="📦 Average Orders",
        value=f"{avg_orders:,.0f}"
    )


with k2:

    st.metric(
        label="👥 Average Visitors",
        value=f"{avg_visitors:,.0f}"
    )


with k3:

    st.metric(
        label="💰 Average Ad Spend",
        value=f"PKR {avg_ad_spend:,.0f}"
    )


with k4:

    st.metric(
        label="📈 Model R²",
        value=f"{r2:.3f}"
    )


st.write("")



# TABS

predict_tab, analytics_tab, insights_tab, data_tab = st.tabs(
    [
        "🔮 Predict Orders",
        "📊 Business Analytics",
        "🤖 Model Insights",
        "📁 Explore Data"
    ]
)

# TAB 1
# PREDICT ORDERS

with predict_tab:

    st.subheader(
        "🔮 Create a New Business Scenario"
    )

    st.write(
        "Enter today's business conditions and generate "
        "an estimated number of daily orders."
    )

    st.divider()

    # INPUT FORM


    with st.form("prediction_form"):

        left_col, right_col = st.columns(2)


        # LEFT SIDE

        with left_col:

            website_visitors = st.slider(
                "👥 Website Visitors",

                min_value=int(
                    df["Website_Visitors"].min()
                ),

                max_value=int(
                    df["Website_Visitors"].max()
                ),

                value=int(
                    df["Website_Visitors"].median()
                ),

                step=50
            )


            ad_spend = st.slider(
                "💰 Ad Spend (PKR)",

                min_value=int(
                    df["Ad_Spend_PKR"].min()
                ),

                max_value=int(
                    df["Ad_Spend_PKR"].max()
                ),

                value=int(
                    df["Ad_Spend_PKR"].median()
                ),

                step=500
            )


        # RIGHT SIDE
        
        with right_col:

            discount_percent = st.slider(
                "🏷️ Discount (%)",

                min_value=int(
                    df["Discount_Percent"].min()
                ),

                max_value=int(
                    df["Discount_Percent"].max()
                ),

                value=int(
                    df["Discount_Percent"].median()
                ),

                step=1
            )


            day_type = st.radio(
                "📅 Day Type",

                options=[
                    "Weekday",
                    "Weekend"
                ],

                horizontal=True
            )


        st.write("")


        # PREDICTION BUTTON
  
        predict_button = st.form_submit_button(
            "🌿 Generate Order Forecast",
            type="primary",
            use_container_width=True
        )

    # PREDICTION
 

    if predict_button:

        # Weekend = 1
        # Weekday = 0

        weekend = (
            1
            if day_type == "Weekend"
            else 0
        )


        # Create new input

        new_day = pd.DataFrame({

            "Website_Visitors": [
                website_visitors
            ],

            "Ad_Spend_PKR": [
                ad_spend
            ],

            "Discount_Percent": [
                discount_percent
            ],

            "Weekend": [
                weekend
            ]

        })


        # Generate prediction

        predicted_orders = model.predict(
            new_day
        )[0]


        st.divider()


        # PREDICTION RESULT
       

        st.success(
            "✅ Order forecast generated successfully."
        )


        st.subheader(
            "🛒 Estimated Daily Orders"
        )


        st.metric(
            label="Predicted Orders",

            value=f"{round(predicted_orders):,} orders"
        )


        st.caption(
            f"Raw model estimate: "
            f"{predicted_orders:.2f} orders"
        )

        # INPUT SUMMARY
       

        st.subheader(
            "📋 Business Scenario"
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "👥 Visitors",
            f"{website_visitors:,}"
        )


        c2.metric(
            "💰 Ad Spend",
            f"PKR {ad_spend:,}"
        )


        c3.metric(
            "🏷️ Discount",
            f"{discount_percent}%"
        )


        c4.metric(
            "📅 Day",
            day_type
        )

        # MODEL INPUT
       

        with st.expander(
            "🔍 View exact input sent to the model"
        ):

            st.dataframe(
                new_day,

                use_container_width=True,

                hide_index=True
            )


        st.warning(
            "This prediction is based on synthetic classroom "
            "data and should be treated as a model estimate, "
            "not a guaranteed future sales number."
        )



# TAB 2
# BUSINESS ANALYTICS


with analytics_tab:

    st.subheader(
        "📊 Business Analytics"
    )

    st.write(
        "Explore the relationship between business factors "
        "and daily orders."
    )

    st.divider()

    # VISITORS VS ORDERS
  

    chart_col1, chart_col2 = st.columns(2)


    with chart_col1:

        st.subheader(
            "👥 Visitors vs Daily Orders"
        )

        st.scatter_chart(
            df,

            x="Website_Visitors",

            y="Daily_Orders",

            use_container_width=True
        )

    # AD SPEND VS ORDERS
  

    with chart_col2:

        st.subheader(
            "💰 Ad Spend vs Daily Orders"
        )

        st.scatter_chart(
            df,

            x="Ad_Spend_PKR",

            y="Daily_Orders",

            use_container_width=True
        )


    # DISCOUNT VS ORDERS
  

    chart_col3, chart_col4 = st.columns(2)


    with chart_col3:

        st.subheader(
            "🏷️ Discount vs Daily Orders"
        )

        st.scatter_chart(
            df,

            x="Discount_Percent",

            y="Daily_Orders",

            use_container_width=True
        )

    # WEEKEND VS WEEKDAY
   
    with chart_col4:

        st.subheader(
            "📅 Weekend vs Weekday"
        )


        day_summary = (
            df.groupby("Weekend")[
                "Daily_Orders"
            ]
            .mean()
            .rename(
                index={
                    0: "Weekday",
                    1: "Weekend"
                }
            )
            .to_frame(
                "Average Orders"
            )
        )


        st.bar_chart(
            day_summary
        )


    st.divider()


    # QUICK BUSINESS STATISTICS
    

    st.subheader(
        "📌 Quick Business Statistics"
    )


    s1, s2, s3, s4 = st.columns(4)


    s1.metric(
        "Highest Orders",
        f"{df['Daily_Orders'].max():,.0f}"
    )


    s2.metric(
        "Lowest Orders",
        f"{df['Daily_Orders'].min():,.0f}"
    )


    s3.metric(
        "Max Visitors",
        f"{df['Website_Visitors'].max():,.0f}"
    )


    s4.metric(
        "Max Ad Spend",
        f"PKR {df['Ad_Spend_PKR'].max():,.0f}"
    )


# TAB 3
# MODEL INSIGHTS


with insights_tab:

    st.subheader(
        "🤖 Model Insights"
    )

    st.write(
        "Understand how the Linear Regression model "
        "performed and what it learned from the data."
    )

    st.divider()


    # EVALUATION METRICS
   

    score_1, score_2 = st.columns(2)


    score_1.metric(
        "Mean Absolute Error",
        f"{mae:.2f} orders"
    )


    score_2.metric(
        "R² Score",
        f"{r2:.3f}"
    )


    st.info(
        f"On the test data, the model's predictions "
        f"are approximately {mae:.2f} orders away "
        f"from the actual value on average."
    )


    st.write(
        f"R² indicates that approximately "
        f"{r2 * 100:.1f}% of the variation "
        f"in test orders is explained by the model."
    )


    st.divider()

    # INTERCEPT
   
    st.subheader(
        "🧠 What Did Linear Regression Learn?"
    )


    st.metric(
        "Intercept",
        f"{model.intercept_:.2f}"
    )


    # COEFFICIENT TABLE
  

    st.dataframe(
        coefficient_table,

        use_container_width=True,

        hide_index=True
    )


    # COEFFICIENT CHART
  

    st.subheader(
        "📊 Feature Coefficients"
    )


    st.bar_chart(
        coefficient_table.set_index(
            "Feature"
        ),

        use_container_width=True
    )


    st.caption(
        "A coefficient describes the fitted relationship "
        "while the other model features are held constant. "
        "It does not automatically prove causation."
    )


    st.divider()


    # ACTUAL VS PREDICTED
 

    st.subheader(
        "🎯 Actual vs Predicted Orders"
    )


    st.scatter_chart(
        test_results,

        x="Actual_Orders",

        y="Predicted_Orders",

        use_container_width=True
    )

    # TEST TABLE

    with st.expander(
        "📋 Open Test-Set Prediction Table"
    ):

        st.dataframe(
            test_results,

            use_container_width=True
        )


# TAB 4
# EXPLORE DATA
#

with data_tab:

    st.subheader(
        "📁 Explore the Synthetic Dataset"
    )

    st.write(
        "Each row represents one historical day. "
        "Daily_Orders is the target variable."
    )

    st.divider()


    # FEATURE SELECTOR
  

    selected_feature = st.selectbox(
        "Choose a feature to compare with Daily Orders",

        options=feature_columns
    )


    st.subheader(
        f"📈 {selected_feature} vs Daily Orders"
    )


    st.scatter_chart(
        df,

        x=selected_feature,

        y="Daily_Orders",

        use_container_width=True
    )

    # COMPLETE DATASET
   

    show_full_data = st.checkbox(
        "Show the complete dataset"
    )


    if show_full_data:

        st.dataframe(
            df,

            use_container_width=True,

            hide_index=True
        )


    # SUMMARY STATISTICS


    with st.expander(
        "📊 View Summary Statistics"
    ):

        st.dataframe(
            df.describe(),

            use_container_width=True
        )


# FOOTER

st.markdown(
    '<div class="footer-text">'
    '🛒 Daily Orders Analytics • '
    'Linear Regression • Streamlit • '
    'Created by Sundas Waqas'
    '</div>',
    unsafe_allow_html=True
)