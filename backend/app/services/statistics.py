from typing import Any

import pandas as pd

from app.services.facility_data import (
    ADDRESS,
    FEE,
    FACILITY_TYPE,
    OLD_ADDRESS,
    WEEKDAY_START,
    WEEKEND_START,
    load_facilities,
)


def _operating_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].fillna("").astype(str).str.strip()
    return int(values.ne("").sum() - values.isin(("0:00", "00:00")).sum())


def get_dashboard_statistics(
    area: str = "전국",
    facility_type: str = "",
) -> dict[str, Any]:
    """Aggregate dashboard values directly from the normalized CSV."""
    frame = load_facilities()
    if area == "기타 시·도":
        filtered = frame.loc[frame["_region"] == "기타"].copy()
    elif area == "전국":
        filtered = frame.copy()
    elif area in frame["_region"].unique():
        filtered = frame.loc[frame["_region"] == area].copy()
    else:
        filtered = frame.iloc[0:0].copy()

    if facility_type:
        filtered = filtered.loc[
            filtered[FACILITY_TYPE].fillna("").astype(str).str.strip() == facility_type
        ].copy()

    facility_name_count = int(
        filtered["개방시설명"].fillna("").astype(str).str.strip().ne("").sum()
    )
    type_counts = (
        filtered[FACILITY_TYPE]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .value_counts()
        .sort_values(ascending=False)
    )

    fee_values = filtered[FEE].fillna("").astype(str).str.strip().str.upper()
    free_count = int(fee_values.eq("N").sum())
    paid_count = int(fee_values.eq("Y").sum())
    unknown_fee_count = int(len(filtered) - free_count - paid_count)

    weekday_count = _operating_count(filtered, WEEKDAY_START)
    weekend_count = _operating_count(filtered, WEEKEND_START)
    operating_unknown_count = int(len(filtered))
    # Count rows with neither valid weekday nor valid weekend start time.
    if WEEKDAY_START in filtered.columns and WEEKEND_START in filtered.columns:
        weekday_values = filtered[WEEKDAY_START].fillna("").astype(str).str.strip()
        weekend_values = filtered[WEEKEND_START].fillna("").astype(str).str.strip()
        no_weekday = weekday_values.eq("") | weekday_values.isin(("0:00", "00:00"))
        no_weekend = weekend_values.eq("") | weekend_values.isin(("0:00", "00:00"))
        operating_unknown_count = int((no_weekday & no_weekend).sum())

    provider_column = "관리기관명"
    if provider_column in filtered.columns:
        provider_counts = (
            filtered[provider_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .loc[lambda values: values.ne("")]
            .value_counts()
            .head(15)
        )
    else:
        provider_counts = pd.Series(dtype="int64")

    region_names = [
        "서울특별시", "경기도", "부산광역시", "대구광역시", "인천광역시",
        "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
        "강원특별자치도", "충청북도", "충청남도", "전북특별자치도",
        "전라남도", "경상북도", "경상남도", "제주특별자치도", "기타",
    ]
    region_counts = {
        name: int((filtered["_region"] == name).sum())
        for name in region_names
    }
    region_counts = {
        name: count for name, count in region_counts.items() if count > 0
    }

    basis_values = []
    if "데이터기준일자" in filtered.columns:
        basis_values = sorted({
            str(value).strip()
            for value in filtered["데이터기준일자"].dropna().tolist()
            if str(value).strip()
        })

    return {
        "total_count": int(len(filtered)),
        "facility_name_count": facility_name_count,
        "facility_type_count": int(type_counts.size),
        "provider_count": int(
            filtered[provider_column].fillna("").astype(str).str.strip().ne("").nunique()
        ) if provider_column in filtered.columns else 0,
        "free_count": free_count,
        "paid_count": paid_count,
        "unknown_fee_count": unknown_fee_count,
        "weekday_count": weekday_count,
        "weekend_count": weekend_count,
        "operating_unknown_count": operating_unknown_count,
        "region_counts": region_counts,
        "type_counts": {
            name: int(count) for name, count in type_counts.head(15).items()
        },
        "provider_counts": {
            name: int(count) for name, count in provider_counts.items()
        },
        "data_basis": {
            "count": len(basis_values),
            "min": basis_values[0] if basis_values else "",
            "max": basis_values[-1] if basis_values else "",
        },
    }
