
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from modules.mle_estimation import calculate_mle
from modules.bayesian_estimation import calculate_bayesian


# -------------------------------------------------
# 1. Bernoulli Naive Bayes Prediction
# -------------------------------------------------

def predict_naive_bayes(
    X,
    class_probs,
    feature_probs
):

    predictions = []

    feature_names = list(X.columns)

    # Avoid numerical problems when taking logarithms
    epsilon = 1e-10

    for _, sample in X.iterrows():

        class_scores = {}

        for c in [0, 1]:

            # Log of class prior probability
            prior = max(class_probs[c], epsilon)

            log_probability = np.log(prior)

            for feature in feature_names:

                x = sample[feature]

                # P(feature = 1 | class)
                p = feature_probs[c][feature]

                # Keep probabilities numerically safe
                p = np.clip(p, epsilon, 1 - epsilon)

                # Bernoulli likelihood
                if x == 1:
                    log_probability += np.log(p)

                else:
                    log_probability += np.log(1 - p)

            class_scores[c] = log_probability

        # Select class with highest log probability
        predicted_class = max(
            class_scores,
            key=class_scores.get
        )

        predictions.append(predicted_class)

    return np.array(predictions)


# -------------------------------------------------
# 2. Calculate Evaluation Metrics
# -------------------------------------------------

def evaluate_classifier(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Not Spam", "Spam"],
            zero_division=0,
            output_dict=True
        )
    }


# -------------------------------------------------
# 3. Train and Compare Both Classifiers
# -------------------------------------------------

def run_classification_experiment(
    df,
    alpha=1.0,
    beta=1.0,
    class_prior=1.0,
    test_size=0.2,
    random_state=42
):

    feature_names = [
        "contains_offer",
        "contains_link",
        "contains_money"
    ]

    X = df[feature_names]
    y = df["target"]

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Create training DataFrame
    train_df = X_train.copy()
    train_df["target"] = y_train

    # ---------------------------------------------
    # MLE Parameter Estimation
    # ---------------------------------------------

    mle_class_probs, mle_feature_probs = calculate_mle(
        train_df
    )

    # ---------------------------------------------
    # Bayesian Parameter Estimation
    # ---------------------------------------------

    bayes_class_probs, bayes_feature_probs = (
        calculate_bayesian(
            train_df,
            alpha=alpha,
            beta=beta,
            class_prior=class_prior
        )
    )

    # ---------------------------------------------
    # MLE Predictions
    # ---------------------------------------------

    mle_predictions = predict_naive_bayes(
        X_test,
        mle_class_probs,
        mle_feature_probs
    )

    # ---------------------------------------------
    # Bayesian Predictions
    # ---------------------------------------------

    bayes_predictions = predict_naive_bayes(
        X_test,
        bayes_class_probs,
        bayes_feature_probs
    )

    # ---------------------------------------------
    # Evaluate Both Models
    # ---------------------------------------------

    mle_results = evaluate_classifier(
        y_test,
        mle_predictions
    )

    bayes_results = evaluate_classifier(
        y_test,
        bayes_predictions
    )

    return {
        "X_test": X_test,
        "y_test": y_test,

        "mle_predictions": mle_predictions,
        "bayes_predictions": bayes_predictions,

        "mle_results": mle_results,
        "bayes_results": bayes_results,

        "mle_class_probs": mle_class_probs,
        "mle_feature_probs": mle_feature_probs,

        "bayes_class_probs": bayes_class_probs,
        "bayes_feature_probs": bayes_feature_probs,

        "X_train": X_train,
        "y_train": y_train
    }
