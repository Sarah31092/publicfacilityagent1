from functools import lru_cache
from typing import Any

import pandas as pd

from app.core.config import CSV_PATH

FACILITY_NAME = "개방시설명"
LOCATION_NAME = "개방장소명"
FACILITY_TYPE = "개방시설유형구분"
ADDRESS = "소재지도로명주소"
OLD_ADDRESS = "소재지지번주소"
REQUIRED_COLUMNS = (FACILITY_NAME, FACILITY_TYPE, ADDRESS)

AREA_OPTIONS = ("전국", "서울특별시", "경기도", "서울특별시+경기도")
AREA_NAMES = {
    "서울특별시": ("서울특별시", "서울시"),
    "경기도": ("경기도",),
}


@lru_cache(maxsize=1)
def load_facilities() -> pd.DataFrame:
    """Load and normalize the CSV once for searches and detail pages."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"공공시설 CSV 파일을 찾을 수 없습니다: {CSV_PATH}")

    try:
        frame = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(CSV_PATH, encoding="cp949", low_memory=False)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"CSV 필수 열이 없습니다: {', '.join(missing_columns)}")

    frame = frame.fillna("")
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str).str.strip()
    frame["_id"] = frame.index.astype(str)
    frame["_region"] = frame.apply(_find_region, axis=1)
    return frame


def _find_region(row: pd.Series) -> str:
    address = f"{row.get(ADDRESS, '')} {row.get(OLD_ADDRESS, '')}"
    if "서울" in address:
        return "서울특별시"
    if "경기" in address:
        return "경기도"
    return "기타"


def _area_mask(frame: pd.DataFrame, area: str) -> pd.Series:
    if area == "전국" or not area:
        return pd.Series(True, index=frame.index)
    if area == "서울특별시+경기도":
        return frame["_region"].isin(("서울특별시", "경기도"))
    if area in AREA_NAMES:
        return frame["_region"] == area
    raise ValueError(f"지원하지 않는 지역입니다: {area}")


def _to_record(row: pd.Series) -> dict[str, Any]:
    return {
        key: ("" if pd.isna(value) else str(value))
        for key, value in row.items()
        if not key.startswith("_")
    } | {"id": str(row["_id"])}


def search_facilities(
    query: str = "", area: str = "전국", limit: int = 30
) -> list[dict[str, Any]]:
    """Search facility name and type using structured CSV filters."""
    matches = _matching_facilities(query, area)
    return [_to_record(row) for _, row in matches.head(limit).iterrows()]


def count_facilities(query: str = "", area: str = "전국") -> int:
    """Count all facilities matching the same filters as structured search."""
    return len(_matching_facilities(query, area))


def _matching_facilities(query: str, area: str) -> pd.DataFrame:
    frame = load_facilities()
    matches = frame.loc[_area_mask(frame, area)]
    query = query.strip()
    if query:
        name_matches = matches[FACILITY_NAME].str.contains(
            query, case=False, regex=False, na=False
        )
        type_matches = matches[FACILITY_TYPE].str.contains(
            query, case=False, regex=False, na=False
        )
        matches = matches.loc[name_matches | type_matches]
    return matches


def get_facility(facility_id: str) -> dict[str, Any] | None:
    frame = load_facilities()
    matches = frame.loc[frame["_id"] == str(facility_id)]
    if matches.empty:
        return None
    return _to_record(matches.iloc[0])


def get_facility_count(area: str = "전국") -> int:
    frame = load_facilities()
    return int(_area_mask(frame, area).sum())
