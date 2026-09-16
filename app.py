import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# BANK CAMPAIGN ICP PREDICTION
# FINAL STREAMLIT DASHBOARD
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_FILE = PROJECT_DIR / "models" / "best_model.joblib"

# IMPORTANT:
# Use FULL DATASET predictions for business/customer targeting
FULL_PREDICTIONS_FILE = (
    PROJECT_DIR / "models" / "full_customer_predictions.csv"
)

# Test-set predictions are kept separately for model evaluation
TEST_PREDICTIONS_FILE = (
    PROJECT_DIR / "models" / "test_predictions.csv"
)

MODEL_COMPARISON_FILE = (
    PROJECT_DIR / "models" / "model_comparison.csv"
)

FEATURE_IMPORTANCE_FILE = (
    PROJECT_DIR / "models" / "feature_importance.csv"
)


# ------------------------------------------------------------
# MODEL EVALUATION IMAGES
# ------------------------------------------------------------

CONFUSION_MATRIX_IMAGE = (
    PROJECT_DIR / "eda_outputs" / "confusion_matrix.png"
)

ROC_CURVE_IMAGE = (
    PROJECT_DIR / "eda_outputs" / "roc_curve.png"
)

PR_CURVE_IMAGE = (
    PROJECT_DIR / "eda_outputs" / "precision_recall_curve.png"
)

FEATURE_IMPORTANCE_IMAGE = (
    PROJECT_DIR / "eda_outputs" / "feature_importance.png"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Campaign ICP Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD FILES
# ============================================================

@st.cache_data
def load_full_predictions():
    return pd.read_csv(FULL_PREDICTIONS_FILE)


@st.cache_data
def load_test_predictions():
    if TEST_PREDICTIONS_FILE.exists():
        return pd.read_csv(TEST_PREDICTIONS_FILE)

    return pd.DataFrame()


@st.cache_data
def load_model_comparison():
    if MODEL_COMPARISON_FILE.exists():
        return pd.read_csv(MODEL_COMPARISON_FILE)

    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_FILE.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_FILE)

    return pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# ============================================================
# FILE CHECK
# ============================================================

if not MODEL_FILE.exists():

    st.error(
        f"Model file not found:\n{MODEL_FILE}"
    )

    st.stop()


if not FULL_PREDICTIONS_FILE.exists():

    st.error(
        "Full customer prediction file not found.\n\n"
        "Please run:\n"
        "python full_customer_prediction.py"
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = load_full_predictions()

test_df = load_test_predictions()

model = load_model()

model_comparison = load_model_comparison()

feature_importance = load_feature_importance()


# ============================================================
# PROJECT CONSTANTS
# ============================================================

TOTAL_DATASET = len(df)

TRAINING_CUSTOMERS = 32950

TEST_CUSTOMERS = 8238

ORIGINAL_INPUT_FEATURES = 20

MODEL_FEATURES = 19


# ============================================================
# CALCULATE FULL DATASET METRICS
# ============================================================

predicted_subscribers = (
    df["predicted_subscription"]
    .eq("yes")
    .sum()
)


predicted_non_subscribers = (
    df["predicted_subscription"]
    .eq("no")
    .sum()
)


average_probability = (
    df["subscription_probability"].mean()
)


high_priority = (
    df["campaign_priority"]
    .eq("High")
    .sum()
)


medium_priority = (
    df["campaign_priority"]
    .eq("Medium")
    .sum()
)


low_priority = (
    df["campaign_priority"]
    .eq("Low")
    .sum()
)


# ============================================================
# TOP 10%
# ============================================================

top_10_count = int(TOTAL_DATASET * 0.10)

top_10_df = (
    df
    .sort_values(
        "subscription_probability",
        ascending=False
    )
    .head(top_10_count)
)

top_10_average = (
    top_10_df["subscription_probability"].mean()
)


top_10_predicted_subscribers = (
    top_10_df["predicted_subscription"]
    .eq("yes")
    .sum()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Bank Campaign ICP Prediction")

st.sidebar.markdown(
    "### ML Campaign Targeting System"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "Customer Targeting",
        "Business Insights",
        "Model Performance",
        "Project Information"
    ]
)


st.sidebar.divider()


st.sidebar.info(
    f"""
    **Final Model**

    Random Forest

    **ROC-AUC:** 0.8146

    **Model Features:** 19

    **Customers Scored:** {TOTAL_DATASET:,}

    `duration` excluded due to
    data leakage.
    """
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "Bank Campaign ICP Prediction"
)

st.caption(
    "Machine Learning Based Customer Targeting & Campaign Prioritization"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Campaign Overview")

    st.markdown(
        """
        This system predicts the likelihood that a bank customer
        will subscribe to a term deposit and ranks customers
        according to their predicted probability.

        The final Random Forest model was evaluated on a held-out
        test set and then used to generate propensity scores for
        the complete historical dataset.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # DATASET COVERAGE
    # --------------------------------------------------------

    st.subheader("Dataset & Prediction Coverage")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Dataset",
            f"{TOTAL_DATASET:,}"
        )


    with col2:

        st.metric(
            "Training Customers",
            f"{TRAINING_CUSTOMERS:,}"
        )


    with col3:

        st.metric(
            "Test Customers",
            f"{TEST_CUSTOMERS:,}"
        )


    with col4:

        st.metric(
            "Model Features",
            f"{MODEL_FEATURES}"
        )


    st.caption(
        "The complete historical dataset contains 41,188 customer records. "
        "All 41,188 records have been scored for customer prioritization."
    )


    st.divider()


    # --------------------------------------------------------
    # FULL DATASET PREDICTION RESULTS
    # --------------------------------------------------------

    st.subheader("Full Dataset ICP Prediction Results")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Predicted Subscribers",
            f"{predicted_subscribers:,}"
        )


    with col2:

        st.metric(
            "Average Probability",
            f"{average_probability:.2%}"
        )


    with col3:

        st.metric(
            "High Priority",
            f"{high_priority:,}"
        )


    with col4:

        st.metric(
            "Customers Scored",
            f"{TOTAL_DATASET:,}"
        )


    st.divider()


    # --------------------------------------------------------
    # PRIORITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Campaign Priority Distribution"
    )


    priority_data = pd.DataFrame(
        {
            "Priority": [
                "High",
                "Medium",
                "Low"
            ],

            "Customers": [
                high_priority,
                medium_priority,
                low_priority
            ]
        }
    )


    col1, col2 = st.columns(2)


    with col1:

        st.bar_chart(
            priority_data.set_index("Priority")
        )


    with col2:

        st.dataframe(
            priority_data,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Predicted Subscription Distribution"
    )


    prediction_data = pd.DataFrame(
        {
            "Prediction": [
                "Subscribe",
                "Do Not Subscribe"
            ],

            "Customers": [
                predicted_subscribers,
                predicted_non_subscribers
            ]
        }
    )


    st.bar_chart(
        prediction_data.set_index("Prediction")
    )


    st.divider()


    # --------------------------------------------------------
    # TOP 10%
    # --------------------------------------------------------

    st.subheader(
        "Highest-Probability Customer Segment"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Top 10% Customers",
            f"{top_10_count:,}"
        )


    with col2:

        st.metric(
            "Average Probability",
            f"{top_10_average:.2%}"
        )


    with col3:

        st.metric(
            "Predicted Subscribers",
            f"{top_10_predicted_subscribers:,}"
        )


    st.info(
        """
        The top 10% segment represents the customers with the
        highest predicted probability of subscribing. This segment
        can be considered first when allocating limited campaign
        resources.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # BUSINESS STORY
    # --------------------------------------------------------

    st.subheader(
        "Business Interpretation"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            f"""
            **1. Identify**

            The model scored **{TOTAL_DATASET:,}**
            historical customer records according to
            predicted subscription probability.
            """
        )


    with col2:

        st.info(
            f"""
            **2. Prioritize**

            **{high_priority:,} customers** are classified
            as High Priority for campaign targeting.
            """
        )


    with col3:

        st.info(
            """
            **3. Allocate Resources**

            The bank can focus campaign resources on
            higher-probability customers instead of
            treating every customer equally.
            """
        )


    st.warning(
        """
        **Important:** These are model-generated propensity scores
        for the available historical dataset. A probability is not
        a guarantee that a customer will subscribe.
        """
    )


# ============================================================
# CUSTOMER TARGETING
# ============================================================

elif page == "Customer Targeting":

    st.header(
        "Customer-Level Campaign Targeting"
    )


    st.markdown(
        """
        Use the filters below to identify customers who can be
        prioritized for a future marketing campaign.
        """
    )


    st.caption(
        f"Customer targeting is based on all {TOTAL_DATASET:,} "
        "scored historical records."
    )


    st.divider()


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        selected_priority = st.selectbox(
            "Campaign Priority",
            [
                "All",
                "High",
                "Medium",
                "Low"
            ]
        )


    with col2:

        selected_prediction = st.selectbox(
            "Predicted Outcome",
            [
                "All",
                "yes",
                "no"
            ]
        )


    with col3:

        minimum_probability = st.slider(
            "Minimum Probability",
            0.00,
            1.00,
            0.00,
            0.05
        )


    with col4:

        if "job" in df.columns:

            jobs = sorted(
                df["job"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_job = st.selectbox(
                "Job",
                ["All"] + jobs
            )

        else:

            selected_job = "All"


    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_df = df.copy()


    if selected_priority != "All":

        filtered_df = filtered_df[
            filtered_df["campaign_priority"]
            == selected_priority
        ]


    if selected_prediction != "All":

        filtered_df = filtered_df[
            filtered_df["predicted_subscription"]
            == selected_prediction
        ]


    filtered_df = filtered_df[
        filtered_df["subscription_probability"]
        >= minimum_probability
    ]


    if (
        selected_job != "All"
        and "job" in filtered_df.columns
    ):

        filtered_df = filtered_df[
            filtered_df["job"]
            == selected_job
        ]


    filtered_df = (
        filtered_df
        .sort_values(
            "subscription_probability",
            ascending=False
        )
        .reset_index(drop=True)
    )


    st.divider()


    # --------------------------------------------------------
    # FILTER RESULT METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Customers Found",
            f"{len(filtered_df):,}"
        )


    with col2:

        if len(filtered_df) > 0:

            avg_filtered_probability = (
                filtered_df[
                    "subscription_probability"
                ].mean()
            )

            st.metric(
                "Average Probability",
                f"{avg_filtered_probability:.2%}"
            )

        else:

            st.metric(
                "Average Probability",
                "0%"
            )


    with col3:

        filtered_yes = (
            filtered_df[
                "predicted_subscription"
            ]
            .eq("yes")
            .sum()
        )

        st.metric(
            "Predicted Subscribers",
            f"{filtered_yes:,}"
        )


    st.divider()


    # --------------------------------------------------------
    # CUSTOMER TABLE
    # --------------------------------------------------------

    st.subheader(
        "Customer Target List"
    )


    display_columns = [
        "customer_rank",
        "age",
        "job",
        "marital",
        "education",
        "housing",
        "loan",
        "contact",
        "month",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "subscription_probability",
        "predicted_subscription",
        "campaign_priority"
    ]


    display_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]


    customer_table = (
        filtered_df[
            display_columns
        ]
        .copy()
    )


    if "subscription_probability" in customer_table.columns:

        customer_table[
            "subscription_probability"
        ] = (
            customer_table[
                "subscription_probability"
            ]
            .map(
                lambda x: f"{x:.2%}"
            )
        )


    st.dataframe(
        customer_table,
        use_container_width=True,
        height=550,
        hide_index=True
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = filtered_df.to_csv(
        index=False
    )


    st.download_button(
        "Download Filtered Customer List",
        data=csv_data,
        file_name="campaign_target_customers.csv",
        mime="text/csv"
    )
# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.header("Business Insights")

    st.markdown(
        """
        This section translates the model's customer propensity scores
        into practical campaign targeting insights.

        The analysis is based on the complete historical dataset of
        scored customers. Results should be interpreted as targeting
        signals rather than guaranteed future customer behavior.
        """
    )

    st.divider()


    # ============================================================
    # 1. CAMPAIGN OPPORTUNITY SUMMARY
    # ============================================================

    st.subheader("Campaign Opportunity Summary")

    TOTAL_DATASET = len(df)

    high_pct = (
        high_priority / TOTAL_DATASET
        if TOTAL_DATASET > 0 else 0
    )

    predicted_subscriber_pct = (
        predicted_subscribers / TOTAL_DATASET
        if TOTAL_DATASET > 0 else 0
    )

    top_10_count = max(1, int(TOTAL_DATASET * 0.10))

    top_10_df = (
        df.sort_values(
            "subscription_probability",
            ascending=False
        ).head(top_10_count)
    )

    top_10_average = (
        top_10_df["subscription_probability"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers Scored",
            f"{TOTAL_DATASET:,}"
        )

    with col2:
        st.metric(
            "Predicted Subscribers",
            f"{predicted_subscribers:,}"
        )

    with col3:
        st.metric(
            "High-Priority Customers",
            f"{high_priority:,}"
        )

    with col4:
        st.metric(
            "Top 10% Avg. Probability",
            f"{top_10_average:.2%}"
        )

    st.caption(
        f"{high_priority:,} customers ({high_pct:.2%}) are classified "
        "as High Priority based on the campaign-priority thresholds."
    )

    st.divider()


    # ============================================================
    # 2. PRIORITY STRATEGY
    # ============================================================

    st.subheader("Campaign Priority Strategy")

    priority_data = pd.DataFrame({
        "Priority": [
            "High",
            "Medium",
            "Low"
        ],
        "Customers": [
            high_priority,
            medium_priority,
            low_priority
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            priority_data.set_index("Priority")
        )

    with col2:

        st.dataframe(
            priority_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            **Recommended resource allocation**

            - **High:** Primary campaign target
            - **Medium:** Secondary / selective targeting
            - **Low:** Lower campaign priority
            """
        )

    st.divider()


    # ============================================================
    # 3. TOP 10% OPPORTUNITY
    # ============================================================

    st.subheader("Highest-Probability Customer Segment")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Top 10% Customers",
            f"{top_10_count:,}"
        )

    with col2:

        st.metric(
            "Average Probability",
            f"{top_10_average:.2%}"
        )

    with col3:

        top_10_subscribers = (
            top_10_df["predicted_subscription"]
            .eq("yes")
            .sum()
        )

        st.metric(
            "Predicted Subscribers",
            f"{top_10_subscribers:,}"
        )

    st.success(
        f"""
        **Business interpretation:** If campaign resources are limited,
        the bank can begin with the top 10% of customers ranked by
        predicted subscription probability.

        This segment contains **{top_10_count:,} customers** with an
        average predicted probability of **{top_10_average:.2%}**.
        """
    )

    st.divider()


    # ============================================================
    # 4. JOB SEGMENT ANALYSIS
    # ============================================================

    if "job" in df.columns:

        st.subheader("Customer Segment Analysis — Job")

        job_analysis = (
            df.groupby("job")
            .agg(
                Customers=(
                    "subscription_probability",
                    "size"
                ),
                Average_Probability=(
                    "subscription_probability",
                    "mean"
                ),
                High_Priority=(
                    "campaign_priority",
                    lambda x: (x == "High").sum()
                ),
                Predicted_Subscriptions=(
                    "predicted_subscription",
                    lambda x: (x == "yes").sum()
                )
            )
            .reset_index()
            .sort_values(
                "Average_Probability",
                ascending=False
            )
        )

        job_chart = job_analysis[
            [
                "job",
                "Average_Probability"
            ]
        ].copy()

        job_chart["Average_Probability"] *= 100

        st.bar_chart(
            job_chart.set_index("job")
        )

        st.dataframe(
            job_analysis,
            use_container_width=True,
            hide_index=True
        )

        best_job = job_analysis.iloc[0]

        st.info(
            f"""
            **Highest predicted-probability job segment:**
            **{best_job['job']}**

            Average predicted probability:
            **{best_job['Average_Probability']:.2%}**

            Customers in this segment:
            **{int(best_job['Customers']):,}**
            """
        )

    st.divider()


    # ============================================================
    # 5. CONTACT METHOD ANALYSIS
    # ============================================================

    if "contact" in df.columns:

        st.subheader("Contact Method Analysis")

        contact_analysis = (
            df.groupby("contact")
            .agg(
                Customers=(
                    "subscription_probability",
                    "size"
                ),
                Average_Probability=(
                    "subscription_probability",
                    "mean"
                ),
                High_Priority=(
                    "campaign_priority",
                    lambda x: (x == "High").sum()
                ),
                Predicted_Subscriptions=(
                    "predicted_subscription",
                    lambda x: (x == "yes").sum()
                )
            )
            .reset_index()
            .sort_values(
                "Average_Probability",
                ascending=False
            )
        )

        contact_chart = contact_analysis[
            [
                "contact",
                "Average_Probability"
            ]
        ].copy()

        contact_chart["Average_Probability"] *= 100

        col1, col2 = st.columns(2)

        with col1:

            st.bar_chart(
                contact_chart.set_index("contact")
            )

        with col2:

            st.dataframe(
                contact_analysis,
                use_container_width=True,
                hide_index=True
            )

        best_contact = contact_analysis.iloc[0]

        st.success(
            f"""
            **Targeting signal:** Customers contacted through
            **{best_contact['contact']}** have the highest average
            predicted subscription probability in this dataset:
            **{best_contact['Average_Probability']:.2%}**.
            """
        )

    st.divider()


    # ============================================================
    # 6. CAMPAIGN MONTH ANALYSIS
    # ============================================================

    if "month" in df.columns:

        st.subheader("Campaign Month Analysis")

        month_analysis = (
            df.groupby("month")
            .agg(
                Customers=(
                    "subscription_probability",
                    "size"
                ),
                Average_Probability=(
                    "subscription_probability",
                    "mean"
                ),
                High_Priority=(
                    "campaign_priority",
                    lambda x: (x == "High").sum()
                ),
                Predicted_Subscriptions=(
                    "predicted_subscription",
                    lambda x: (x == "yes").sum()
                )
            )
            .reset_index()
            .sort_values(
                "Average_Probability",
                ascending=False
            )
        )

        month_chart = month_analysis[
            [
                "month",
                "Average_Probability"
            ]
        ].copy()

        month_chart["Average_Probability"] *= 100

        st.bar_chart(
            month_chart.set_index("month")
        )

        st.dataframe(
            month_analysis,
            use_container_width=True,
            hide_index=True
        )

        best_month = month_analysis.iloc[0]

        st.info(
            f"""
            **Highest predicted-probability campaign month in the
            historical data:** **{best_month['month']}**

            Average predicted probability:
            **{best_month['Average_Probability']:.2%}**

            This indicates a strong historical targeting signal,
            but it should not be interpreted as proof that the month
            itself causes higher subscription.
            """
        )

    st.divider()


    # ============================================================
    # 7. PREVIOUS CAMPAIGN OUTCOME
    # ============================================================

    if "poutcome" in df.columns:

        st.subheader("Previous Campaign Outcome Analysis")

        poutcome_analysis = (
            df.groupby("poutcome")
            .agg(
                Customers=(
                    "subscription_probability",
                    "size"
                ),
                Average_Probability=(
                    "subscription_probability",
                    "mean"
                ),
                High_Priority=(
                    "campaign_priority",
                    lambda x: (x == "High").sum()
                ),
                Predicted_Subscriptions=(
                    "predicted_subscription",
                    lambda x: (x == "yes").sum()
                )
            )
            .reset_index()
            .sort_values(
                "Average_Probability",
                ascending=False
            )
        )

        poutcome_chart = poutcome_analysis[
            [
                "poutcome",
                "Average_Probability"
            ]
        ].copy()

        poutcome_chart["Average_Probability"] *= 100

        col1, col2 = st.columns(2)

        with col1:

            st.bar_chart(
                poutcome_chart.set_index("poutcome")
            )

        with col2:

            st.dataframe(
                poutcome_analysis,
                use_container_width=True,
                hide_index=True
            )

        best_poutcome = poutcome_analysis.iloc[0]

        st.success(
            f"""
            **Strongest historical campaign signal:**
            customers with previous campaign outcome
            **{best_poutcome['poutcome']}** have the highest average
            predicted probability at
            **{best_poutcome['Average_Probability']:.2%}**.
            """
        )

        st.divider()


    # ============================================================
    # 8. CAMPAIGN CONTACT FREQUENCY ANALYSIS
    # ============================================================

    if "campaign" in df.columns:

        st.subheader("Campaign Contact Frequency Analysis")

        st.markdown(
            """
            This analysis examines how the number of contacts made during
            the current campaign is associated with customer subscription
            propensity.

            The analysis uses the **full 41,188-customer scored dataset**.
            """
        )

        # ------------------------------------------------------------
        # Create practical contact-frequency groups
        # ------------------------------------------------------------

        df_frequency = df.copy()

        df_frequency["contact_frequency_group"] = pd.cut(
            df_frequency["campaign"],
            bins=[0, 1, 2, 3, 5, float("inf")],
            labels=[
                "1 contact",
                "2 contacts",
                "3 contacts",
                "4–5 contacts",
                "6+ contacts"
            ],
            include_lowest=True
        )

        # ------------------------------------------------------------
        # Calculate business metrics
        # ------------------------------------------------------------

        frequency_analysis = (
            df_frequency
            .groupby(
                "contact_frequency_group",
                observed=False
            )
            .agg(
                Customers=(
                    "subscription_probability",
                    "size"
                ),
                Average_Probability=(
                    "subscription_probability",
                    "mean"
                ),
                High_Priority=(
                    "campaign_priority",
                    lambda x: (x == "High").sum()
                ),
                Predicted_Subscriptions=(
                    "predicted_subscription",
                    lambda x: (x == "yes").sum()
                )
            )
            .reset_index()
        )

        # ------------------------------------------------------------
        # Convert probability to percentage for display
        # ------------------------------------------------------------

        frequency_display = frequency_analysis.copy()

        frequency_display["Average_Probability"] = (
            frequency_display["Average_Probability"] * 100
        )

        # ------------------------------------------------------------
        # Chart
        # ------------------------------------------------------------

        st.markdown("### Average Predicted Subscription Probability")

        chart_data = frequency_display[
            [
                "contact_frequency_group",
                "Average_Probability"
            ]
        ].copy()

        chart_data = chart_data.set_index(
            "contact_frequency_group"
        )

        st.bar_chart(chart_data)

        # ------------------------------------------------------------
        # Table
        # ------------------------------------------------------------

        st.markdown("### Contact Frequency Summary")

        st.dataframe(
            frequency_display,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------------------------
        # Business interpretation
        # ------------------------------------------------------------

        first_probability = frequency_analysis.iloc[0][
            "Average_Probability"
        ]

        last_probability = frequency_analysis.iloc[-1][
            "Average_Probability"
        ]

        difference = first_probability - last_probability

        st.info(
            f"""
            **Business Insight**

            Customers contacted **once** have an average predicted
            subscription probability of **{first_probability:.2%}**.

            Customers contacted **6 or more times** have an average
            predicted probability of **{last_probability:.2%}**.

            This represents a difference of approximately
            **{difference:.2%}**.

            **Interpretation:** In this historical dataset, subscription
            propensity generally decreases as the number of campaign
            contacts increases.
            """
        )

        # ------------------------------------------------------------
        # Business recommendation
        # ------------------------------------------------------------

        st.success(
            """
            **Campaign Recommendation**

            The bank can prioritize high-probability customers earlier
            in the campaign rather than repeatedly allocating resources
            to lower-probability customers.

            However, contact frequency should be treated as a
            **targeting signal, not a causal rule**. The model considers
            all 19 input features together when generating an individual
            customer's probability.
            """
        )

        # ------------------------------------------------------------
        # Download analysis
        # ------------------------------------------------------------

        st.download_button(
            "Download Contact Frequency Analysis",
            data=frequency_display.to_csv(index=False),
            file_name="campaign_contact_frequency_analysis.csv",
            mime="text/csv"
        )


    # ============================================================
    # 9. BUSINESS RECOMMENDATIONS
    # ============================================================

    st.subheader("Business Recommendations")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 1. Prioritize High-Probability Customers

            Focus initial campaign resources on customers with the
            highest predicted subscription probability.

            This can help the bank avoid treating all customers
            equally when campaign resources are limited.
            """
        )

        st.markdown(
            """
            ### 2. Use Targeting Signals for Contact Planning

            The contact-method analysis can help identify which
            customer groups historically show stronger predicted
            subscription propensity.

            These results should support campaign planning rather
            than be treated as causal evidence.
            """
        )

    with col2:

        st.markdown(
            """
            ### 3. Build Customer Segments

            Job, previous campaign outcome and other customer
            characteristics can be used to create focused campaign
            segments.

            High-probability segments can receive earlier attention.
            """
        )

        st.markdown(
            """
            ### 4. Use Probability for Resource Allocation

            Instead of using only a yes/no prediction, the bank can
            rank customers by probability and select a campaign
            cutoff according to available budget and sales capacity.
            """
        )

    st.divider()


    # ============================================================
    # 10. FINAL BUSINESS STORY
    # ============================================================

    st.subheader("Final Business Story")

    st.markdown(
        f"""
        ### From Prediction → Targeting → Action

        **1. Score**

        The Random Forest model assigns a subscription propensity
        score to each of the **{TOTAL_DATASET:,}** available
        historical customer records.

        **2. Rank**

        Customers are ranked according to their predicted probability
        of subscribing to the term deposit.

        **3. Prioritize**

        **{high_priority:,} customers ({high_pct:.2%})** are classified
        as High Priority for campaign targeting.

        **4. Focus**

        When campaign capacity is limited, the bank can begin with
        the highest-ranked customers, including the top 10% segment.

        **5. Measure**

        Future campaigns should compare actual campaign outcomes
        against model predictions to evaluate and improve targeting
        performance.
        """
    )

    st.warning(
        """
        **Important interpretation**

        The model outputs are propensity scores generated from the
        available historical dataset. They indicate relative
        likelihood for customer prioritization and are not guarantees
        that customers will subscribe.

        Historical associations such as month, contact method or job
        should also not automatically be interpreted as causal effects.
        """
    )

    st.divider()


    # ============================================================
    # 11. DOWNLOAD BUSINESS SUMMARY
    # ============================================================

    st.subheader("Download Scored Customer Data")

    st.caption(
        "Use the Customer Targeting page to filter customers and "
        "download a campaign-specific target list."
    )

    st.download_button(
        "Download Full Customer Scoring Data",
        data=df.to_csv(index=False),
        file_name="full_customer_scoring.csv",
        mime="text/csv"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "Machine Learning Model Performance"
    )

    st.markdown(
        """
        Model performance is evaluated using the held-out test
        dataset of 8,238 customers. These metrics measure how well
        the model generalizes to unseen test observations.
        """
    )

    st.info(
        """
        **Important distinction:**

        • Model Performance → evaluated on 8,238 test customers.

        • Customer Targeting → propensity scores generated for
        all 41,188 historical customer records.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    if not model_comparison.empty:

        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )

    st.divider()


    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    st.subheader(
        "Selected Model: Random Forest"
    )

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Accuracy",
            "85.87%"
        )


    with col2:

        st.metric(
            "Precision",
            "41.73%"
        )


    with col3:

        st.metric(
            "Recall",
            "64.12%"
        )


    with col4:

        st.metric(
            "F1-Score",
            "50.55%"
        )


    with col5:

        st.metric(
            "ROC-AUC",
            "81.46%"
        )


    st.caption(
        "Evaluation metrics calculated on the 8,238-customer test set."
    )


    st.divider()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    if CONFUSION_MATRIX_IMAGE.exists():

        st.subheader(
            "Confusion Matrix"
        )

        st.image(
            str(CONFUSION_MATRIX_IMAGE),
            use_container_width=True
        )

        st.caption(
            "Confusion matrix for the Random Forest model on the test set."
        )


    st.divider()


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    if ROC_CURVE_IMAGE.exists():

        st.subheader(
            "ROC Curve"
        )

        st.image(
            str(ROC_CURVE_IMAGE),
            use_container_width=True
        )


    st.divider()


    # --------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------

    if PR_CURVE_IMAGE.exists():

        st.subheader(
            "Precision-Recall Curve"
        )

        st.image(
            str(PR_CURVE_IMAGE),
            use_container_width=True
        )


    st.divider()


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance"
    )

    if not feature_importance.empty:

        st.dataframe(
            feature_importance.head(20),
            use_container_width=True,
            hide_index=True
        )


    if FEATURE_IMPORTANCE_IMAGE.exists():

        st.image(
            str(FEATURE_IMPORTANCE_IMAGE),
            use_container_width=True
        )


    st.info(
        """
        The most influential model inputs include economic
        indicators such as `euribor3m`, `nr.employed`, and
        `emp.var.rate`, along with campaign and customer-history
        variables.
        """
    )
# ============================================================
# PROJECT INFORMATION
# ============================================================

elif page == "Project Information":

    st.header(
        "Project Information"
    )


    # --------------------------------------------------------
    # BUSINESS PROBLEM
    # --------------------------------------------------------

    st.subheader(
        "Business Problem"
    )


    st.write(
        """
        The objective is to predict whether a bank customer will
        subscribe to a term deposit and use the predicted
        probabilities to prioritize customers for marketing
        campaigns.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader(
        "Dataset"
    )


    dataset_info = pd.DataFrame(
        {
            "Component": [
                "Total observations",
                "Original input features",
                "Features used for modelling",
                "Training observations",
                "Testing observations",
                "Full dataset customers scored",
                "Target variable",
                "Problem type"
            ],

            "Value": [
                "41,188",
                "20",
                "19",
                "32,950",
                "8,238",
                "41,188",
                "y",
                "Binary Classification"
            ]
        }
    )


    st.dataframe(
        dataset_info,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    st.subheader(
        "Model Features"
    )


    model_features = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"
    ]


    feature_df = pd.DataFrame(
        {
            "Model Features": model_features
        }
    )


    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # --------------------------------------------------------
    # MACHINE LEARNING PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "Machine Learning Pipeline"
    )


    st.markdown(
        """
        **1. Data Understanding**

        → Dataset structure, target, data types, class imbalance
        and unknown values were examined.

        **2. Exploratory Data Analysis**

        → Customer, campaign and economic patterns were analyzed.

        **3. Feature Engineering**

        → Predictive features were defined and `duration` was excluded.

        **4. Preprocessing**

        → Numerical and categorical variables were processed using
        a consistent preprocessing pipeline.

        **5. Model Training**

        → Logistic Regression, Decision Tree, Random Forest and
        Gradient Boosting were evaluated.

        **6. Model Evaluation**

        → Accuracy, Precision, Recall, F1-score and ROC-AUC were
        calculated on the held-out test set.

        **7. Model Selection**

        → Random Forest was selected based on ROC-AUC and the
        overall balance of classification metrics.

        **8. Full Dataset Scoring**

        → The final model generated propensity scores for all
        41,188 available historical customer records.

        **9. Campaign Prioritization**

        → Customers were categorized into High, Medium and Low
        campaign priority groups.

        **10. Dashboard**

        → Business users can explore customer predictions,
        business insights and model performance.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # DATA LEAKAGE
    # --------------------------------------------------------

    st.subheader(
        "Data Leakage Control"
    )


    st.warning(
        """
        `duration` was excluded from the final predictive model.

        Duration represents the length of the current marketing
        call and is only known after the call has occurred.

        Using it for future campaign targeting would therefore
        introduce data leakage.

        Although duration may have strong predictive power,
        excluding it makes the model more realistic for
        pre-contact customer targeting.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # BUSINESS OBJECTIVE
    # --------------------------------------------------------

    st.subheader(
        "Business Objective"
    )


    st.success(
        """
        The final system helps the bank prioritize customers
        according to predicted subscription probability so that
        campaign resources can be allocated more efficiently.

        The model should be used as a decision-support and
        prioritization tool rather than as a guarantee of
        customer subscription.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # IMPORTANT SCORING NOTE
    # --------------------------------------------------------

    st.subheader(
        "Full Dataset Scoring Note"
    )


    st.info(
        """
        The 41,188 customer scores represent propensity estimates
        generated for the available historical dataset.

        They should be interpreted as customer prioritization
        scores, not guaranteed future outcomes.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Bank Campaign ICP Prediction | "
    "Random Forest Customer Targeting System | "
    f"{TOTAL_DATASET:,} Customer Records | "
    "19 Model Features"
)