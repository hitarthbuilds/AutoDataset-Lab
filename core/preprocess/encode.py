from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def get_onehot_encoder():
    # For sklearn >= 1.4, "sparse_output" replaces "sparse"
    return OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )


def get_ordinal_encoder(categories="auto"):
    return OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )


def get_encoder(name: str):
    """
    Returns the correct encoder instance based on name.
    """
    name = name.lower().strip()

    if name == "onehot":
        return get_onehot_encoder()

    if name == "ordinal":
        return get_ordinal_encoder()

    if name == "none":
        return None

    raise ValueError(f"Unknown encoder: {name}")
