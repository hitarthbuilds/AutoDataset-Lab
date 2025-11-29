import os

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded CSV file to disk."""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path
