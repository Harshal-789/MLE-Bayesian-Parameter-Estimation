import numpy as np
import pandas as pd


def generate_dataset(n_samples=1000, spam_ratio=0.5, random_state=42):

    # 1. Initialize random number generator
    rng = np.random.default_rng(random_state)

    # 2. Define feature names
    feature_names = [
        "contains_offer",
        "contains_link",
        "contains_money"
    ]

    # 3. Define actual probabilities for each class
    true_probabilities = {
        0: [0.20, 0.30, 0.15],  # Not Spam
        1: [0.80, 0.70, 0.75]   # Spam
    }

    # 4. Calculate number of samples in each class
    n_spam = int(n_samples * spam_ratio)
    n_not_spam = n_samples - n_spam

    # 5. Generate Not Spam samples
    not_spam_features = rng.binomial(
        n=1,
        p=true_probabilities[0],
        size=(n_not_spam, len(feature_names))
    )

    # 6. Generate Spam samples
    spam_features = rng.binomial(
        n=1,
        p=true_probabilities[1],
        size=(n_spam, len(feature_names))
    )

    # 7. Combine both datasets
    X = np.vstack([
        not_spam_features,
        spam_features
    ])

    # 8. Generate class labels
    y = np.array(
        [0] * n_not_spam + [1] * n_spam
    )

    # 9. Shuffle the dataset
    indices = rng.permutation(n_samples)

    X = X[indices]
    y = y[indices]

    # 10. Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    df["target"] = y

    # 11. Store actual probabilities for later comparison
    true_params = pd.DataFrame(
        true_probabilities,
        index=feature_names
    )

    true_params.columns = ["Not Spam (0)", "Spam (1)"]

    return df, true_params