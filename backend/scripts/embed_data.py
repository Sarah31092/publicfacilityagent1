import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vector_store import build_vector_index


if __name__ == "__main__":
    count = build_vector_index()
    print(f"완료: 공공시설 {count}건을 ChromaDB에 저장했습니다.")
