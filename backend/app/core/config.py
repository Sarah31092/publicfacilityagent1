import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env", override=False)

CSV_PATH = Path(
    os.getenv(
        "FACILITY_CSV_PATH",
        str(BACKEND_DIR / "data" / "전국공공시설개방정보표준데이터.csv"),
    )
)
if not CSV_PATH.is_absolute():
    CSV_PATH = BACKEND_DIR / CSV_PATH

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BACKEND_DIR / "chroma_db")))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = BACKEND_DIR / CHROMA_DIR

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
