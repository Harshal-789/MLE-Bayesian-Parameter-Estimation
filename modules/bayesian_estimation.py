
import pandas as pd


def calculate_bayesian(
    df,
    alpha=1.0,
    beta=1.0,
    class_prior=1.0
):

    feature_names = [
        "contains_offer",
        "contains_link",
        "contains_money"
    ]

    classes = [0, 1]

    total_samples = len(df)

    # Store Bayesian estimates
    class_probabilities = {}
    feature_probabilities = {}

    # Estimate class probabilities using Dirichlet prior
    total_prior = class_prior * len(classes)

    for c in classes:

        class_count = (df["target"] == c).sum()

        class_probabilities[c] = (
            class_count + class_prior
        ) / (
            total_samples + total_prior
        )

        feature_probabilities[c] = {}

        class_data = df[df["target"] == c]
        class_count = len(class_data)

        for feature in feature_names:

            # Count positive observations
            feature_count = class_data[feature].sum()

            # Bayesian posterior mean
            posterior_mean = (
                feature_count + alpha
            ) / (
                class_count + alpha + beta
            )

            feature_probabilities[c][feature] = (
                posterior_mean
            )

    return class_probabilities, feature_probabilities
