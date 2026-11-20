from functools import lru_cache
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.core.config import CHROMA_DIR, EMBEDDING_MODEL
from app.services.facility_data import load_facilities

COLLECTION_NAME = "public_facilities"
BATCH_SIZE = 200


class VectorIndexNotReadyError(RuntimeError):
    """Raised when a vector search is requested before data has been embedded."""


@lru_cache(maxsize=1)
def _client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def _embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _document(row: Any) -> str:
    fields = (
        "개방시설명",
        "개방장소명",
        "개방시설유형구분",
        "소재지도로명주소",
        "소재지지번주소",
        "휴관일",
        "평일운영시작시각",
        "평일운영종료시각",
        "주말운영시작시각",
        "주말운영종료시각",
        "유료사용여부",
        "사용료",
        "부대시설정보",
        "신청방법구분",
        "관리기관명",
        "사용안내전화번호",
    )
    return " ".join(f"{field}: {row[field]}" for field in fields if row[field])


def _metadata(row: Any) -> dict[str, str]:
    return {
        "facility_id": str(row["_id"]),
        "region": str(row["_region"]),
        "facility_name": str(row["개방시설명"]),
        "facility_type": str(row["개방시설유형구분"]),
        "address": str(row["소재지도로명주소"]),
    }


def build_vector_index() -> int:
    """Rebuild the Chroma collection from the normalized CSV."""
    client = _client()
    existing_names = {
        collection.name for collection in client.list_collections()
    }
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"description": "전국 공공시설 개방 정보"},
    )
    frame = load_facilities()
    count = len(frame)
    for start in range(0, count, BATCH_SIZE):
        batch = frame.iloc[start : start + BATCH_SIZE]
        collection.add(
            ids=batch["_id"].tolist(),
            documents=[_document(row) for _, row in batch.iterrows()],
            metadatas=[_metadata(row) for _, row in batch.iterrows()],
        )
        print(f"임베딩 진행: {min(start + len(batch), count)}/{count}")
    return count


def search_similar_facilities(
    query: str, area: str = "전국", limit: int = 10
) -> list[dict[str, Any]]:
    client = _client()
    names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME not in names:
        raise VectorIndexNotReadyError(
            "벡터 데이터가 아직 준비되지 않았습니다. "
            "backend 폴더에서 python scripts/embed_data.py를 먼저 실행하세요."
        )

    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=_embedding_function()
    )
    if collection.count() == 0:
        raise VectorIndexNotReadyError("ChromaDB에 저장된 시설 데이터가 없습니다.")

    where: dict[str, Any] | None = None
    if area == "서울특별시":
        where = {"region": "서울특별시"}
    elif area == "경기도":
        where = {"region": "경기도"}
    elif area == "서울특별시+경기도":
        where = {
            "$or": [
                {"region": "서울특별시"},
                {"region": "경기도"},
            ]
        }
    elif area != "전국":
        raise ValueError(f"지원하지 않는 지역입니다: {area}")

    result = collection.query(
        query_texts=[query],
        n_results=min(limit, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "id": metadata["facility_id"],
            "개방시설명": metadata["facility_name"],
            "개방시설유형구분": metadata["facility_type"],
            "소재지도로명주소": metadata["address"],
            "document": document,
            "distance": distance,
        }
        for _, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]
