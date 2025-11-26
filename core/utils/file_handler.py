import polars as pl
import os

UPLOAD_DIR = "data"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to disk and return Polars DataFrame + path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load immediately as Polars DF
    df = pl.read_csv(file_path)

    return df, file_path


def load_saved_file(file_path):
    """Load DataFrame from disk using Polars."""
    if os.path.exists(file_path):
        return pl.read_csv(file_path)
    return None
