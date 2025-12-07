# pages/6_Per_Column_Insights.py
from core.insights.per_column_insights import app

def app_page():
    app()

# Streamlit multipage expects `app()` top-level, but we used app() inside module.
def app():
    app_page()

if __name__ == "__main__":
    app()
