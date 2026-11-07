
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from modules.data_generation import generate_dataset
from modules.mle_estimation import calculate_mle
from modules.bayesian_estimation import calculate_bayesian
from modules.classifiers import run_classification_experiment


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MLE vs Bayesian Parameter Estimation",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# APPLICATION HEADER
# =====================================================

st.title("MLE vs Bayesian Parameter Estimation")

st.markdown("""
### Comparative Analysis Using Simulated Data

This project compares Maximum Likelihood Estimation (MLE)
and Bayesian Parameter Estimation using a simulated spam
classification dataset.

**Main objectives:**
- Generate synthetic data.
- Estimate parameters using MLE and Bayesian methods.
- Compare classification performance.
- Visualize actual and estimated probabilities.
""")

st.divider()


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

session_keys = [
    "df",
    "true_params",
    "mle_results",
    "bayesian_results",
    "classification_results"
]

for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None


# =====================================================
# SIDEBAR CONFIGURATION
# =====================================================

st.sidebar.header("Dataset Configuration")

n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)

spam_ratio = st.sidebar.slider(
    "Spam Ratio",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=10000,
    value=42,
    step=1
)

st.sidebar.divider()

st.sidebar.header("Bayesian Prior Settings")

alpha = st.sidebar.number_input(
    "Beta Prior Alpha",
    min_value=0.1,
    max_value=100.0,
    value=1.0,
    step=0.5
)

beta = st.sidebar.number_input(
    "Beta Prior Beta",
    min_value=0.1,
    max_value=100.0,
    value=1.0,
    step=0.5
)

class_prior = st.sidebar.number_input(
    "Class Prior Strength",
    min_value=0.1,
    max_value=100.0,
    value=1.0,
    step=0.5
)


# =====================================================
# 1. SYNTHETIC DATASET GENERATION
# =====================================================

st.header("1. Synthetic Dataset Generation")

st.write("""
Generate a simulated dataset containing three binary features
that help distinguish spam and non-spam messages.
""")

if st.button("Generate Dataset", type="primary"):

    df, true_params = generate_dataset(
        n_samples=n_samples,
        spam_ratio=spam_ratio,
        random_state=int(random_state)
    )

    st.session_state.df = df
    st.session_state.true_params = true_params

    # Clear previous calculations
    st.session_state.mle_results = None
    st.session_state.bayesian_results = None
    st.session_state.classification_results = None

    st.success("Dataset generated successfully!")


# =====================================================
# CHECK DATASET AVAILABILITY
# =====================================================

if st.session_state.df is None:

    st.info(
        "Configure the dataset using the sidebar and click "
        "'Generate Dataset' to begin."
    )

    st.stop()


df = st.session_state.df
true_params = st.session_state.true_params


# =====================================================
# 1.1 DATASET OVERVIEW
# =====================================================

st.subheader("1.1 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Samples",
    f"{len(df):,}"
)

col2.metric(
    "Spam Messages",
    f"{int((df['target'] == 1).sum()):,}"
)

col3.metric(
    "Non-Spam Messages",
    f"{int((df['target'] == 0).sum()):,}"
)


# =====================================================
# 1.2 DATASET PREVIEW
# =====================================================

st.subheader("1.2 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =====================================================
# 1.3 CLASS DISTRIBUTION
# =====================================================

st.subheader("1.3 Class Distribution")

class_counts = (
    df["target"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

distribution_df = pd.DataFrame({
    "Class": ["Not Spam", "Spam"],
    "Count": [
        class_counts.loc[0],
        class_counts.loc[1]
    ]
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(
        distribution_df,
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.bar_chart(
        distribution_df.set_index("Class")
    )


# =====================================================
# 1.4 ACTUAL FEATURE PROBABILITIES
# =====================================================

st.subheader("1.4 Actual Feature Probabilities")

st.write(
    "These are the actual probabilities used to generate "
    "the synthetic dataset. They serve as reference values."
)

st.dataframe(
    true_params,
    use_container_width=True
)


# =====================================================
# 2. MAXIMUM LIKELIHOOD ESTIMATION
# =====================================================

st.divider()

st.header("2. Maximum Likelihood Estimation (MLE)")

st.write("""
MLE estimates parameters by finding the values that maximize
the likelihood of observing the available data.
""")

if st.button("Calculate MLE Parameters"):

    mle_class_probs, mle_feature_probs = calculate_mle(df)

    st.session_state.mle_results = (
        mle_class_probs,
        mle_feature_probs
    )


if st.session_state.mle_results is not None:

    mle_class_probs, mle_feature_probs = (
        st.session_state.mle_results
    )

    st.subheader("2.1 MLE Class Probabilities")

    mle_class_df = pd.DataFrame(
        list(mle_class_probs.items()),
        columns=["Class", "Probability"]
    )

    mle_class_df["Class"] = mle_class_df["Class"].map({
        0: "Not Spam",
        1: "Spam"
    })

    st.dataframe(
        mle_class_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("2.2 MLE Feature Probabilities")

    mle_feature_df = pd.DataFrame(mle_feature_probs)

    mle_feature_df = mle_feature_df.rename(
        columns={
            0: "Not Spam",
            1: "Spam"
        }
    )

    mle_feature_df.index.name = "Feature"

    st.dataframe(
        mle_feature_df.round(4),
        use_container_width=True
    )


# =====================================================
# 3. BAYESIAN PARAMETER ESTIMATION
# =====================================================

st.divider()

st.header("3. Bayesian Parameter Estimation")

st.write("""
Bayesian estimation combines observed data with prior beliefs
to calculate posterior probabilities using a Beta distribution.
""")

st.caption(
    f"Feature Prior: Beta({alpha}, {beta}) | "
    f"Class Prior Strength: {class_prior}"
)

if st.button("Calculate Bayesian Parameters"):

    bayesian_class_probs, bayesian_feature_probs = (
        calculate_bayesian(
            df,
            alpha=alpha,
            beta=beta,
            class_prior=class_prior
        )
    )

    st.session_state.bayesian_results = (
        bayesian_class_probs,
        bayesian_feature_probs
    )


if st.session_state.bayesian_results is not None:

    bayesian_class_probs, bayesian_feature_probs = (
        st.session_state.bayesian_results
    )

    st.subheader("3.1 Bayesian Class Probabilities")

    bayesian_class_df = pd.DataFrame(
        list(bayesian_class_probs.items()),
        columns=["Class", "Probability"]
    )

    bayesian_class_df["Class"] = (
        bayesian_class_df["Class"].map({
            0: "Not Spam",
            1: "Spam"
        })
    )

    st.dataframe(
        bayesian_class_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("3.2 Bayesian Feature Probabilities")

    bayesian_feature_df = pd.DataFrame(
        bayesian_feature_probs
    )

    bayesian_feature_df = bayesian_feature_df.rename(
        columns={
            0: "Not Spam",
            1: "Spam"
        }
    )

    bayesian_feature_df.index.name = "Feature"

    st.dataframe(
        bayesian_feature_df.round(4),
        use_container_width=True
    )


# =====================================================
# 4. CLASSIFICATION PERFORMANCE COMPARISON
# =====================================================

st.divider()

st.header("4. Classification Performance Comparison")

st.write("""
The estimated parameters are used in a Naive Bayes classifier.
We compare MLE and Bayesian classification performance using
a separate test dataset.
""")

test_size = st.slider(
    "Test Dataset Proportion",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)


if st.button("Run Classification Experiment"):

    try:

        with st.spinner("Running classification experiment..."):

            results = run_classification_experiment(
                df,
                alpha=alpha,
                beta=beta,
                class_prior=class_prior,
                test_size=test_size,
                random_state=int(random_state)
            )

            st.session_state.classification_results = results

    except Exception as error:

        st.session_state.classification_results = None

        st.error(
            f"Classification experiment failed: {error}"
        )


# =====================================================
# 4.1 DISPLAY CLASSIFICATION RESULTS
# =====================================================

if st.session_state.classification_results is not None:

    results = st.session_state.classification_results

    try:

        # Exact keys confirmed from your classifiers.py

        y_test = results["y_test"]

        mle_predictions = results["mle_predictions"]

        bayesian_predictions = results["bayes_predictions"]

        # Calculate metrics directly using scikit-learn

        def calculate_metrics(y_true, y_pred):

            return {
                "accuracy": accuracy_score(y_true, y_pred),

                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "f1_score": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "confusion_matrix": confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1]
                )
            }

        mle_metrics = calculate_metrics(
            y_test,
            mle_predictions
        )

        bayesian_metrics = calculate_metrics(
            y_test,
            bayesian_predictions
        )

        # -------------------------------------------------
        # PERFORMANCE METRICS
        # -------------------------------------------------

        st.subheader("4.1 Performance Metrics")

        metrics_df = pd.DataFrame([
            {
                "Method": "MLE",
                "Accuracy": mle_metrics["accuracy"],
                "Precision": mle_metrics["precision"],
                "Recall": mle_metrics["recall"],
                "F1 Score": mle_metrics["f1_score"]
            },
            {
                "Method": "Bayesian",
                "Accuracy": bayesian_metrics["accuracy"],
                "Precision": bayesian_metrics["precision"],
                "Recall": bayesian_metrics["recall"],
                "F1 Score": bayesian_metrics["f1_score"]
            }
        ])

        st.dataframe(
            metrics_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # ACCURACY COMPARISON
        # -------------------------------------------------

        st.subheader("4.2 Accuracy Comparison")

        accuracy_df = metrics_df[
            ["Method", "Accuracy"]
        ].set_index("Method")

        st.bar_chart(accuracy_df)

        # -------------------------------------------------
        # CONFUSION MATRICES
        # -------------------------------------------------

        st.subheader("4.3 Confusion Matrices")

        col1, col2 = st.columns(2)

        with col1:

            st.write("MLE Confusion Matrix")

            mle_confusion = pd.DataFrame(
                mle_metrics["confusion_matrix"],
                index=["Actual Not Spam", "Actual Spam"],
                columns=[
                    "Predicted Not Spam",
                    "Predicted Spam"
                ]
            )

            st.dataframe(
                mle_confusion,
                use_container_width=True
            )

        with col2:

            st.write("Bayesian Confusion Matrix")

            bayesian_confusion = pd.DataFrame(
                bayesian_metrics["confusion_matrix"],
                index=["Actual Not Spam", "Actual Spam"],
                columns=[
                    "Predicted Not Spam",
                    "Predicted Spam"
                ]
            )

            st.dataframe(
                bayesian_confusion,
                use_container_width=True
            )

        # -------------------------------------------------
        # SAMPLE PREDICTIONS
        # -------------------------------------------------

        st.subheader("4.4 Sample Predictions")

        prediction_df = pd.DataFrame({
            "Actual Class": y_test,
            "MLE Prediction": mle_predictions,
            "Bayesian Prediction": bayesian_predictions
        })

        prediction_df = prediction_df.head(20).copy()

        prediction_df = prediction_df.replace({
            0: "Not Spam",
            1: "Spam"
        })

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )

    except (KeyError, TypeError, ValueError) as error:

        st.error(
            f"Could not display classification results: {error}"
        )

        st.write(
            "Available result keys:",
            list(results.keys())
        )


# =====================================================
# 5. PARAMETER ESTIMATION VISUALIZATION
# =====================================================

st.divider()

st.header("5. Parameter Estimation Visualization")

st.write("""
Compare the actual probabilities used to generate the dataset
with the estimates obtained through MLE and Bayesian methods.
""")

if (
    st.session_state.mle_results is not None
    and st.session_state.bayesian_results is not None
):

    _, mle_feature_probs = st.session_state.mle_results

    _, bayesian_feature_probs = (
        st.session_state.bayesian_results
    )

    feature_names = [
        "contains_offer",
        "contains_link",
        "contains_money"
    ]

    comparison_records = []

    for class_id in [0, 1]:

        class_name = (
            "Not Spam" if class_id == 0 else "Spam"
        )

        actual_column = (
            "Not Spam (0)"
            if class_id == 0
            else "Spam (1)"
        )

        for feature in feature_names:

            comparison_records.append({
                "Feature": feature,
                "Class": class_name,

                "Actual": float(
                    true_params.loc[
                        feature,
                        actual_column
                    ]
                ),

                "MLE": mle_feature_probs[class_id][feature],

                "Bayesian": bayesian_feature_probs[class_id][feature]
            })

    comparison_df = pd.DataFrame(comparison_records)

    st.subheader("5.1 Actual vs Estimated Probabilities")

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------
    # PARAMETER COMPARISON BAR CHART
    # -------------------------------------------------

    st.subheader("5.2 Parameter Comparison Chart")

    labels = (
        comparison_df["Class"]
        + " - "
        + comparison_df["Feature"]
    )

    x = range(len(comparison_df))

    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        [i - width for i in x],
        comparison_df["Actual"],
        width,
        label="Actual"
    )

    ax.bar(
        list(x),
        comparison_df["MLE"],
        width,
        label="MLE"
    )

    ax.bar(
        [i + width for i in x],
        comparison_df["Bayesian"],
        width,
        label="Bayesian"
    )

    ax.set_title(
        "Actual vs MLE vs Bayesian Estimates"
    )

    ax.set_ylabel("Probability")

    ax.set_ylim(0, 1)

    ax.set_xticks(list(x))

    ax.set_xticklabels(
        labels,
        rotation=30,
        ha="right"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

else:

    st.info(
        "Calculate both MLE and Bayesian parameters to view "
        "the parameter comparison visualization."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center;">

**MLE vs Bayesian Parameter Estimation**

Comparative Analysis Using Simulated Data

Developed using Python, NumPy, Pandas, Scikit-learn,
Matplotlib and Streamlit.

</div>
""", unsafe_allow_html=True)
