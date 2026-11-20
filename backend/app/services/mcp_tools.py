from typing import Any

from app.services.facility_data import get_facility, search_facilities as structured_search
from app.services.vector_store import search_similar_facilities


def search_facilities(
    query: str = "",
    area: str = "전국",
    mode: str = "structured",
    limit: int = 10,
) -> dict[str, Any]:
    """Search by facility name/type or find semantically similar facilities."""
    if mode == "vector":
        results = search_similar_facilities(query, area=area, limit=limit)
    elif mode == "structured":
        results = structured_search(query, area=area, limit=limit)
    else:
        raise ValueError("검색 방식은 structured 또는 vector여야 합니다.")
    return {"query": query, "area": area, "mode": mode, "results": results}


def get_facility_details(facility_id: str) -> dict[str, Any]:
    """Return all available CSV fields for one facility."""
    facility = get_facility(facility_id)
    if facility is None:
        return {"error": f"시설 ID {facility_id}를 찾을 수 없습니다."}
    return facility
