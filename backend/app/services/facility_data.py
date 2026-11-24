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
FEE = "유료사용여부"
WEEKDAY_START = "평일운영시작시각"
WEEKDAY_END = "평일운영종료시각"
WEEKEND_START = "주말운영시작시각"
WEEKEND_END = "주말운영종료시각"
LATITUDE = "위도"
LONGITUDE = "경도"

AREA_OPTIONS = (
    "전국",
    "서울특별시",
    "경기도",
    "부산광역시",
    "대구광역시",
    "인천광역시",
    "광주광역시",
    "대전광역시",
    "울산광역시",
    "세종특별자치시",
    "강원특별자치도",
    "충청북도",
    "충청남도",
    "전북특별자치도",
    "전라남도",
    "경상북도",
    "경상남도",
    "제주특별자치도",
    "기타 시·도",
    "서울특별시+경기도",
)
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
    province_names = (
        "서울특별시", "부산광역시", "대구광역시", "인천광역시",
        "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
        "경기도", "강원특별자치도", "강원도", "충청북도", "충청남도",
        "전북특별자치도", "전라북도", "전라남도", "경상북도",
        "경상남도", "제주특별자치도", "제주도",
    )
    for province in province_names:
        if province in address:
            if province in ("서울시", "서울특별시"):
                return "서울특별시"
            if province in ("경기도",):
                return "경기도"
            return province if province not in ("강원도", "전라북도", "제주도") else {
                "강원도": "강원특별자치도",
                "전라북도": "전북특별자치도",
                "제주도": "제주특별자치도",
            }[province]
    return "기타"


def _area_mask(frame: pd.DataFrame, area: str) -> pd.Series:
    if area == "전국" or not area:
        return pd.Series(True, index=frame.index)
    if area == "기타 시·도":
        return frame["_region"] == "기타"
    if area == "서울특별시+경기도":
        return frame["_region"].isin(("서울특별시", "경기도"))
    if area in AREA_OPTIONS:
        return frame["_region"] == area
    raise ValueError(f"지원하지 않는 지역입니다: {area}")


def _to_record(row: pd.Series) -> dict[str, Any]:
    return {
        key: ("" if pd.isna(value) else str(value))
        for key, value in row.items()
        if not key.startswith("_")
    } | {"id": str(row["_id"])}


def search_facilities(
    query: str = "",
    area: str = "전국",
    facility_type: str = "",
    fee: str = "전체",
    operating: str = "전체",
    facility_name: str = "",
    place_name: str = "",
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Search the CSV with combined facility filters."""
    matches = _matching_facilities(
        query=query,
        area=area,
        facility_type=facility_type,
        fee=fee,
        operating=operating,
        facility_name=facility_name,
        place_name=place_name,
    )
    return [_to_record(row) for _, row in matches.head(limit).iterrows()]


def count_facilities(
    query: str = "",
    area: str = "전국",
    facility_type: str = "",
    fee: str = "전체",
    operating: str = "전체",
    facility_name: str = "",
    place_name: str = "",
) -> int:
    """Count all facilities matching the same filters as structured search."""
    return len(_matching_facilities(
        query=query,
        area=area,
        facility_type=facility_type,
        fee=fee,
        operating=operating,
        facility_name=facility_name,
        place_name=place_name,
    ))


def _has_operating_time(row: pd.Series, operating: str) -> bool:
    if operating == "평일 운영":
        start = str(row.get(WEEKDAY_START, "")).strip()
        end = str(row.get(WEEKDAY_END, "")).strip()
    elif operating == "주말 운영":
        start = str(row.get(WEEKEND_START, "")).strip()
        end = str(row.get(WEEKEND_END, "")).strip()
    else:
        return True
    return any(value not in ("", "0:00", "00:00") for value in (start, end))


def get_filter_options() -> tuple[list[str], list[str]]:
    frame = load_facilities()
    facility_types = sorted(
        {str(value).strip() for value in frame[FACILITY_TYPE].tolist() if str(value).strip()}
    )
    return facility_types, ["전체", "무료", "유료"]


def get_valid_map_markers(facilities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only rows with usable Korean latitude and longitude values."""
    markers: list[dict[str, Any]] = []
    for facility in facilities:
        try:
            latitude = float(facility.get(LATITUDE, ""))
            longitude = float(facility.get(LONGITUDE, ""))
        except (TypeError, ValueError):
            continue
        if not (33.0 <= latitude <= 43.0 and 124.0 <= longitude <= 132.0):
            continue
        fee_value = str(facility.get(FEE, "")).strip().upper()
        fee_label = "유료" if fee_value == "Y" else "무료" if fee_value == "N" else "정보 없음"
        weekday_start = str(facility.get(WEEKDAY_START, "")).strip()
        weekday_end = str(facility.get(WEEKDAY_END, "")).strip()
        markers.append({
            "id": str(facility.get("id", "")),
            "name": str(facility.get(FACILITY_NAME, "")).strip(),
            "type": str(facility.get(FACILITY_TYPE, "")).strip(),
            "address": str(
                facility.get(ADDRESS, "") or facility.get(OLD_ADDRESS, "")
            ).strip(),
            "weekday": (
                f"{weekday_start} ~ {weekday_end}"
                if weekday_start or weekday_end
                else "정보 없음"
            ),
            "fee": fee_label,
            "latitude": latitude,
            "longitude": longitude,
        })
    return markers


def filter_facility_records(
    facilities: list[dict[str, Any]],
    area: str = "전국",
    facility_type: str = "",
    fee: str = "전체",
    operating: str = "전체",
    facility_name: str = "",
    place_name: str = "",
    query: str = "",
) -> list[dict[str, Any]]:
    """Apply detailed filters to existing result records, including vector results."""
    filtered: list[dict[str, Any]] = []
    for facility in facilities:
        address = f"{facility.get(ADDRESS, '')} {facility.get(OLD_ADDRESS, '')}"
        region = "기타"
        for province in (
            "서울특별시", "부산광역시", "대구광역시", "인천광역시",
            "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
            "경기도", "강원특별자치도", "강원도", "충청북도", "충청남도",
            "전북특별자치도", "전라북도", "전라남도", "경상북도",
            "경상남도", "제주특별자치도", "제주도",
        ):
            if province in address:
                if province in ("강원도",):
                    region = "강원특별자치도"
                elif province in ("전라북도",):
                    region = "전북특별자치도"
                elif province in ("제주도",):
                    region = "제주특별자치도"
                else:
                    region = province
                break
        if area == "서울특별시+경기도":
            area_match = region in ("서울특별시", "경기도")
        elif area == "기타 시·도":
            area_match = region == "기타"
        else:
            area_match = area == "전국" or region == area
        if not area_match:
            continue
        if facility_type and facility.get(FACILITY_TYPE, "") != facility_type:
            continue
        if fee != "전체":
            fee_value = str(facility.get(FEE, "")).strip().upper()
            if fee == "무료" and fee_value != "N":
                continue
            if fee == "유료" and fee_value != "Y":
                continue
        if operating == "평일 운영":
            start = str(facility.get(WEEKDAY_START, "")).strip()
            end = str(facility.get(WEEKDAY_END, "")).strip()
            if not any(value not in ("", "0:00", "00:00") for value in (start, end)):
                continue
        if operating == "주말 운영":
            start = str(facility.get(WEEKEND_START, "")).strip()
            end = str(facility.get(WEEKEND_END, "")).strip()
            if not any(value not in ("", "0:00", "00:00") for value in (start, end)):
                continue
        name = str(facility.get(FACILITY_NAME, "")).lower()
        location = str(facility.get(LOCATION_NAME, "")).lower()
        if facility_name and facility_name.lower() not in name:
            continue
        if place_name and place_name.lower() not in location:
            continue
        if query:
            needle = query.lower()
            if not any(needle in str(facility.get(field, "")).lower() for field in (
                FACILITY_NAME, LOCATION_NAME, FACILITY_TYPE
            )):
                continue
        filtered.append(facility)
    return filtered


def _matching_facilities(
    query: str,
    area: str,
    facility_type: str = "",
    fee: str = "전체",
    operating: str = "전체",
    facility_name: str = "",
    place_name: str = "",
) -> pd.DataFrame:
    frame = load_facilities()
    matches = frame.loc[_area_mask(frame, area)]
    if facility_type:
        matches = matches.loc[matches[FACILITY_TYPE] == facility_type]
    if fee in ("무료", "유료") and FEE in frame.columns:
        if fee == "무료":
            matches = matches.loc[matches[FEE].str.upper().isin(("N",))]
        else:
            matches = matches.loc[matches[FEE].str.upper().isin(("Y",))]
    elif fee in ("무료", "유료"):
        matches = matches.iloc[0:0]
    if operating != "전체":
        matches = matches.loc[matches.apply(
            lambda row: _has_operating_time(row, operating), axis=1
        )]

    query = query.strip()
    if query:
        name_matches = matches[FACILITY_NAME].str.contains(
            query, case=False, regex=False, na=False
        )
        location_matches = (
            matches[LOCATION_NAME].str.contains(query, case=False, regex=False, na=False)
            if LOCATION_NAME in matches.columns
            else pd.Series(False, index=matches.index)
        )
        type_matches = matches[FACILITY_TYPE].str.contains(
            query, case=False, regex=False, na=False
        )
        matches = matches.loc[name_matches | location_matches | type_matches]
    if facility_name:
        matches = matches.loc[matches[FACILITY_NAME].str.contains(
            facility_name.strip(), case=False, regex=False, na=False
        )]
    if place_name and LOCATION_NAME in matches.columns:
        matches = matches.loc[matches[LOCATION_NAME].str.contains(
            place_name.strip(), case=False, regex=False, na=False
        )]
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
