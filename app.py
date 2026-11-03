
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="E-commerce Purchase Intention",
    page_icon="🛍️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "purchase_prediction_model.pkl"
DATA_PATH = BASE_DIR / "data" / "online_shoppers_intention.csv"

# =====================================================
# 2. LOAD MODEL AND DATA
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    return data.drop_duplicates().copy()

if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset file not found.")
    st.stop()

model = load_model()
df = load_dataset()

# =====================================================
# 3. APPLICATION HEADER
# =====================================================

st.title("🛍️ E-commerce Customer Purchase Intention")

st.caption(
    "Machine Learning for Business Applications | "
    "Customer Purchase Intention Prediction"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Business Problem",
        "Data Insights",
        "Purchase Prediction"
    ]
)

# =====================================================
# PAGE 1: BUSINESS PROBLEM
# =====================================================

if page == "Business Problem":

    st.header("Understanding the Business Problem")

    st.write("""
    E-commerce platforms receive a large number of
    website visits, but only a proportion of visitors
    complete a purchase.

    This project uses machine learning to analyse
    customer browsing behaviour and predict whether
    a browsing session is likely to result in a purchase.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Total Sessions")
        st.markdown(f"## {len(df):,}")

    with col2:
        st.markdown("### Dataset Attributes")
        st.markdown("## 18")

    with col3:
        st.markdown("### ML Task")
        st.markdown("## Classification")

    st.divider()

    st.subheader("Project Objective")

    st.write("""
    The objective is to identify browsing patterns
    associated with customer purchase behaviour and
    generate predictions using a trained machine
    learning model.
    """)

    st.subheader("Target Variable: Revenue")

    st.write("""
    - True: The session resulted in a purchase.
    - False: The session did not result in a purchase.
    """)

    st.subheader("Potential Business Applications")

    st.write("""
    - Understanding customer browsing behaviour.
    - Identifying sessions with potential purchase intent.
    - Supporting website and checkout optimisation.
    - Exploring customer engagement opportunities.
    """)

    st.info(
        "The model provides exploratory decision support. "
        "A prediction does not guarantee a purchase."
    )

# =====================================================
# PAGE 2: DATA INSIGHTS
# =====================================================

elif page == "Data Insights":

    st.header("Customer Behaviour and Data Insights")

    df["Purchase_Status"] = df["Revenue"].map({
        True: "Purchase",
        False: "No Purchase",
        1: "Purchase",
        0: "No Purchase"
    })

    # -------------------------------------------------
    # CHART 1: PURCHASE DISTRIBUTION
    # -------------------------------------------------

    st.subheader("1. Purchase Distribution")

    purchase_counts = (
        df["Purchase_Status"]
        .value_counts()
        .reindex(["Purchase", "No Purchase"])
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    purchase_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Purchase vs No Purchase Sessions")
    ax.set_xlabel("Purchase Status")
    ax.set_ylabel("Number of Sessions")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------
    # CHART 2: VISITOR TYPE PURCHASE RATE
    # -------------------------------------------------

    st.subheader("2. Purchase Rate by Visitor Type")

    visitor_summary = df.groupby("VisitorType").agg(
        Sessions=("Revenue", "count"),
        Purchases=("Revenue", "sum")
    )

    visitor_summary["Purchase Rate (%)"] = (
        visitor_summary["Purchases"] /
        visitor_summary["Sessions"] * 100
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    visitor_summary["Purchase Rate (%)"].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Purchase Rate by Visitor Type")
    ax.set_xlabel("Visitor Type")
    ax.set_ylabel("Purchase Rate (%)")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # HTML TABLE INSTEAD OF STREAMLIT DATAFRAME

    st.markdown("**Visitor Type Summary**")

    visitor_html = visitor_summary.round(2).to_html(
        border=0,
        classes="visitor-table"
    )

    st.markdown(
        visitor_html,
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # CHART 3: PRODUCT BROWSING DURATION
    # -------------------------------------------------

    st.subheader("3. Product Browsing Duration")

    duration_summary = (
        df.groupby("Purchase_Status")[
            "ProductRelated_Duration"
        ].mean()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    duration_summary.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Average Product Browsing Duration")
    ax.set_xlabel("Purchase Status")
    ax.set_ylabel("Average Duration (Seconds)")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------
    # CHART 4: MONTHLY PURCHASE RATE
    # -------------------------------------------------

    st.subheader("4. Monthly Purchase Rate")

    monthly_summary = df.groupby("Month").agg(
        Sessions=("Revenue", "count"),
        Purchases=("Revenue", "sum")
    )

    monthly_summary["Purchase Rate (%)"] = (
        monthly_summary["Purchases"] /
        monthly_summary["Sessions"] * 100
    )

    month_order = [
        "Feb", "Mar", "May", "June", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    available_months = [
        month for month in month_order
        if month in monthly_summary.index
    ]

    monthly_summary = monthly_summary.reindex(
        available_months
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        monthly_summary.index,
        monthly_summary["Purchase Rate (%)"],
        marker="o"
    )

    ax.set_title("Monthly Purchase Rate")
    ax.set_xlabel("Month")
    ax.set_ylabel("Purchase Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "These charts describe historical patterns "
        "and do not establish causation."
    )

# =====================================================
# PAGE 3: PURCHASE PREDICTION
# =====================================================

elif page == "Purchase Prediction":

    st.header("Predict Customer Purchase Intention")

    st.write("""
    Enter the customer browsing-session details below.
    The trained machine learning model will analyse
    the information and generate a purchase prediction.
    """)

    # CATEGORY OPTIONS

    month_options = sorted(
        df["Month"].dropna().unique().tolist()
    )

    visitor_options = sorted(
        df["VisitorType"].dropna().unique().tolist()
    )

    os_options = sorted(
        df["OperatingSystems"].dropna().unique().tolist()
    )

    browser_options = sorted(
        df["Browser"].dropna().unique().tolist()
    )

    region_options = sorted(
        df["Region"].dropna().unique().tolist()
    )

    traffic_options = sorted(
        df["TrafficType"].dropna().unique().tolist()
    )

    # INPUT FORM

    with st.form("customer_prediction_form"):

        st.subheader("Customer Browsing Information")

        col1, col2 = st.columns(2)

        with col1:

            month = st.selectbox("Month", month_options)

            visitor_type = st.selectbox(
                "Visitor Type", visitor_options
            )

            weekend = st.selectbox(
                "Weekend Session", [False, True]
            )

            administrative = st.number_input(
                "Administrative Pages Visited",
                min_value=0,
                value=0
            )

            administrative_duration = st.number_input(
                "Administrative Duration (seconds)",
                min_value=0.0,
                value=0.0
            )

            informational = st.number_input(
                "Informational Pages Visited",
                min_value=0,
                value=0
            )

            informational_duration = st.number_input(
                "Informational Duration (seconds)",
                min_value=0.0,
                value=0.0
            )

        with col2:

            product_related = st.number_input(
                "Product-related Pages Visited",
                min_value=0,
                value=5
            )

            product_duration = st.number_input(
                "Product-related Duration (seconds)",
                min_value=0.0,
                value=300.0
            )

            bounce_rate = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.02
            )

            exit_rate = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.04
            )

            special_day = st.number_input(
                "Special Day Value",
                min_value=0.0,
                max_value=1.0,
                value=0.0
            )

        st.subheader("Technical and Traffic Information")

        col3, col4, col5 = st.columns(3)

        with col3:
            operating_system = st.selectbox(
                "Operating System", os_options
            )

        with col4:
            browser = st.selectbox(
                "Browser", browser_options
            )

        with col5:
            region = st.selectbox(
                "Region", region_options
            )

        traffic = st.selectbox(
            "Traffic Type", traffic_options
        )

        predict_button = st.form_submit_button(
            "Predict Purchase Intention"
        )

    # GENERATE PREDICTION

    if predict_button:

        input_data = pd.DataFrame([{

            "Administrative": administrative,

            "Administrative_Duration":
                administrative_duration,

            "Informational": informational,

            "Informational_Duration":
                informational_duration,

            "ProductRelated": product_related,

            "ProductRelated_Duration":
                product_duration,

            "BounceRates": bounce_rate,

            "ExitRates": exit_rate,

            "SpecialDay": special_day,

            "Month": month,

            "OperatingSystems": operating_system,

            "Browser": browser,

            "Region": region,

            "TrafficType": traffic,

            "VisitorType": visitor_type,

            "Weekend": weekend

        }])

        try:

            prediction = model.predict(input_data)[0]

            purchase_score = (
                model.predict_proba(input_data)[0][1]
            )

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:

                st.success("Predicted Outcome: Purchase")

            else:

                st.warning("Predicted Outcome: No Purchase")

            st.markdown("### Model Purchase Score")

            st.markdown(f"## {purchase_score:.1%}")

            st.caption(
                "This is a model-generated score, not a "
                "guaranteed or independently calibrated "
                "probability of purchase."
            )

            st.subheader("Suggested Business Response")

            if prediction == 1:

                st.write("""
                The session has been classified as a
                potential purchase session.

                Possible business responses:

                - Keep product information accessible.
                - Maintain a simple checkout process.
                - Display relevant product recommendations.
                """)

            else:

                st.write("""
                The session has been classified as a
                potential non-purchase session.

                Possible business responses:

                - Review product discovery and navigation.
                - Examine possible checkout friction.
                - Explore customer engagement activities.
                """)

            st.info(
                "These suggestions are exploratory and "
                "are not proven actions that will cause "
                "a customer to purchase."
            )

        except Exception as error:

            st.error("The prediction could not be generated.")
            st.exception(error)

# =====================================================
# SIDEBAR FOOTER
# =====================================================

st.sidebar.divider()

st.sidebar.caption(
    "MLBA Project | E-commerce Purchase Intention"
)
