# src/config.py
import multiprocessing
import psutil
from pathlib import Path

# --- System Settings ---
N_JOBS = min(4, multiprocessing.cpu_count())
MEMORY_GB = psutil.virtual_memory().total / (1024**3)

# --- File Paths ---
# Dynamically finds the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# --- Required Data Columns (Anonymized) ---
# Column names expected by the data loader
COLUMN_COMPLAINT = 'YAKINMA'  # Main complaint text
COLUMN_HISTORY = 'ÖYKÜ'      # Patient history text
COLUMN_DIAGNOSIS = 'TANI'    # Optional: Diagnosis text for enrichment
COLUMN_TRIAGE = 'Triaj'    # Optional: Triage level (e.g., 'GREEN', 'YELLOW')
COLUMN_GROUP = 'data_group'  # Column name to be created for identifying the data source

# --- LDA Model Parameters ---
N_TOPICS = 6
RANDOM_STATE = 42
MAX_ITER_LDA = 200
LEARNING_METHOD = 'batch'

# --- Vectorizer Parameters ---
MAX_DF = 0.75  # More restrictive
MIN_DF = 50    # Higher minimum for stability
MAX_FEATURES = 1000
NGRAM_RANGE = (1, 1)

# --- Medical Categorization ---
CONFIDENCE_THRESHOLD = 0.6  # Stricter threshold for medical categorization

# --- Visualization ---
CHART_DPI = 300
WORDCLOUD_MAX_WORDS = 15

# --- Dependency Availability ---
try:
    import gensim
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    TURKISH_STOPWORDS = set(stopwords.words('turkish'))
except ImportError:
    NLTK_AVAILABLE = False
    TURKISH_STOPWORDS = set()

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False