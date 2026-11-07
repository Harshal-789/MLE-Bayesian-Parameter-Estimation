import numpy as np
import pandas as pd


def calculate_mle(df):

    feature_names = [
        "contains_offer",
        "contains_link",
        "contains_money"
    ]

    classes = [0, 1]

    # Store estimated parameters
    class_probabilities = {}
    feature_probabilities = {}

    total_samples = len(df)

    for c in classes:

        class_data = df[df["target"] == c]

        class_count = len(class_data)

        # Estimate prior probability of each class
        class_probabilities[c] = class_count / total_samples

        feature_probabilities[c] = {}

        for feature in feature_names:

            # Count feature occurrences
            feature_count = class_data[feature].sum()

            # Maximum Likelihood Estimate
            probability = feature_count / class_count

            feature_probabilities[c][feature] = probability

    return class_probabilities, feature_probabilities