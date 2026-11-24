import json
import re
from typing import Any

from openai import OpenAI, OpenAIError

from app.core.config import OPENAI_API_KEY, OPENAI_MODEL
from app.services.mcp_tools import get_facility_details, search_facilities
from app.services.facility_data import (
    AREA_OPTIONS,
    get_filter_options,
    search_facilities as csv_search,
)
from app.services.vector_store import search_similar_facilities
from app.services.vector_store import VectorIndexNotReadyError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_facilities",
            "description": "공공시설을 지역, 이름, 유형 또는 의미 유사도로 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "시설 검색어 또는 질문"},
                    "area": {
                        "type": "string",
                        "enum": list(AREA_OPTIONS),
                    },
                    "mode": {"type": "string", "enum": ["structured", "vector"]},
                    "facility_type": {"type": "string"},
                    "fee": {"type": "string", "enum": ["전체", "무료", "유료"]},
                    "operating": {"type": "string", "enum": ["전체", "평일 운영", "주말 운영"]},
                },
                "required": ["query", "area", "mode"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_facility_details",
            "description": "검색 결과에 표시된 ID를 이용해 공공시설 상세 정보를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "facility_id": {"type": "string", "description": "검색 결과 시설 ID"}
                },
                "required": ["facility_id"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_facilities": search_facilities,
    "get_facility_details": get_facility_details,
}


def extract_recommendation_criteria(question: str, area: str = "전국") -> dict[str, Any]:
    """Extract only conditions that are explicit or inferable from the Korean question."""
    text = question.strip()
    selected_area = area
    area_aliases = {
        "서울": "서울특별시",
        "경기": "경기도",
        "부산": "부산광역시",
        "대구": "대구광역시",
        "인천": "인천광역시",
        "광주": "광주광역시",
        "대전": "대전광역시",
        "울산": "울산광역시",
        "세종": "세종특별자치시",
        "강원": "강원특별자치도",
        "충북": "충청북도",
        "충남": "충청남도",
        "전북": "전북특별자치도",
        "전남": "전라남도",
        "경북": "경상북도",
        "경남": "경상남도",
        "제주": "제주특별자치도",
    }
    for alias, province in area_aliases.items():
        if alias in text:
            selected_area = province
            break
    for province in AREA_OPTIONS:
        if province != "전국" and province in text:
            selected_area = province
            break

    fee = "전체"
    if "무료" in text:
        fee = "무료"
    elif "유료" in text:
        fee = "유료"

    operating = "전체"
    evening = any(word in text for word in ("저녁", "야간", "18시", "19시", "20시"))
    if "주말" in text:
        operating = "주말 운영"
    elif "평일" in text:
        operating = "평일 운영"
    elif evening:
        operating = "평일 운영"

    facility_types, _ = get_filter_options()
    facility_type = ""
    found_types = [item for item in facility_types if item in text]
    if found_types:
        facility_type = max(found_types, key=len)

    capacity_match = re.search(r"(\d[\d,\s]*)명|(\d[\d,\s]*)인", text)
    minimum_capacity = 0
    if capacity_match:
        raw_capacity = capacity_match.group(1) or capacity_match.group(2)
        minimum_capacity = int(re.sub(r"[,\s]", "", raw_capacity))

    return {
        "area": selected_area,
        "facility_type": facility_type,
        "fee": fee,
        "operating": operating,
        "evening": evening,
        "minimum_capacity": minimum_capacity,
    }


def _recommendation_reasons(
    facility: dict[str, Any],
    criteria: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    area = criteria["area"]
    address = f"{facility.get('소재지도로명주소', '')} {facility.get('소재지지번주소', '')}"
    if area == "전국" or area in address:
        reasons.append(f"요청 지역({area})과 일치합니다.")
    else:
        reasons.append("요청 지역과 주소가 일치하지 않거나 확인되지 않습니다.")

    facility_type = criteria["facility_type"]
    actual_type = str(facility.get("개방시설유형구분", "")).strip()
    if facility_type:
        reasons.append(
            "요청 시설 유형과 일치합니다."
            if facility_type == actual_type
            else f"시설 유형은 '{actual_type or '정보 없음'}'입니다."
        )

    fee = criteria["fee"]
    fee_value = str(facility.get("유료사용여부", "")).strip().upper()
    if fee != "전체":
        fee_match = (fee == "무료" and fee_value == "N") or (fee == "유료" and fee_value == "Y")
        reasons.append(
            f"{fee} 조건과 일치합니다."
            if fee_match
            else ("이 시설의 이용료는 원본 데이터 기준 일치 여부를 확인할 수 없습니다."
                  if not fee_value else "요청한 이용료 조건과 일치하지 않습니다.")
        )

    operating = criteria["operating"]
    if operating == "주말 운영":
        start = str(facility.get("주말운영시작시각", "")).strip()
        end = str(facility.get("주말운영종료시각", "")).strip()
        reasons.append(
            f"주말 운영시간 정보가 있습니다: {start or '정보 없음'} ~ {end or '정보 없음'}."
            if start not in ("", "0:00", "00:00") or end not in ("", "0:00", "00:00")
            else "주말 운영시간 정보가 없습니다."
        )
    elif operating == "평일 운영":
        start = str(facility.get("평일운영시작시각", "")).strip()
        end = str(facility.get("평일운영종료시각", "")).strip()
        reasons.append(
            f"평일 운영시간 정보가 있습니다: {start or '정보 없음'} ~ {end or '정보 없음'}."
            if start not in ("", "0:00", "00:00") or end not in ("", "0:00", "00:00")
            else "평일 운영시간 정보가 없습니다."
        )

    if criteria["evening"]:
        end = str(facility.get("평일운영종료시각", "")).strip()
        try:
            hour = int(end.split(":")[0])
        except (ValueError, IndexError, AttributeError):
            hour = 0
        reasons.append(
            "평일 종료시각이 18시 이후입니다."
            if hour >= 18
            else ("평일 종료시각이 18시 이전이거나 정보가 없습니다."
                  if hour else "평일 저녁 이용 가능 여부를 확인할 운영종료시각 정보가 없습니다.")
        )

    if criteria["minimum_capacity"]:
        try:
            capacity = int(float(str(facility.get("수용가능인원수", "")).replace(",", "").strip()))
        except (TypeError, ValueError):
            capacity = 0
        minimum_capacity = criteria["minimum_capacity"]
        if capacity:
            reasons.append(
                f"{capacity:,}명 수용 가능으로 요청한 최소 {minimum_capacity:,}명 조건과 "
                + ("일치합니다." if capacity >= minimum_capacity else "일치하지 않습니다.")
            )
        else:
            reasons.append("수용인원 정보가 없어 요청 조건과 일치 여부를 확인할 수 없습니다.")
    return reasons


def _get_recommendations(question: str, area: str) -> dict[str, Any]:
    criteria = extract_recommendation_criteria(question, area)
    exact = csv_search(
        "",
        area=criteria["area"],
        facility_type=criteria["facility_type"],
        fee=criteria["fee"],
        operating=criteria["operating"],
        limit=30,
    )

    try:
        vector_results = search_similar_facilities(question, area=criteria["area"], limit=12)
    except VectorIndexNotReadyError:
        vector_results = []

    combined: dict[str, dict[str, Any]] = {
        str(item["id"]): item for item in exact
    }
    for item in vector_results:
        combined.setdefault(str(item["id"]), item)

    facilities = list(combined.values())[:30]
    for facility in facilities:
        facility["recommendation_reasons"] = _recommendation_reasons(facility, criteria)

    facilities.sort(
        key=lambda item: sum(
            1 for reason in item["recommendation_reasons"]
            if "일치" in reason and "일치하지" not in reason
        ),
        reverse=True,
    )
    return {
        "answer": (
            "조건에 맞는 공공시설을 원본 CSV 기준으로 비교했습니다."
            if facilities
            else "조건에 맞는 시설을 찾지 못했습니다. 지역, 이용료 또는 시설 유형을 변경해 다시 검색해 주세요."
        ),
        "criteria": criteria,
        "facilities": facilities,
    }


def _local_rag_answer(question: str, area: str) -> dict[str, Any]:
    return _get_recommendations(question, area)


def _call_tool(name: str, arguments: dict[str, Any]) -> str:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return json.dumps({"error": f"허용되지 않은 도구입니다: {name}"}, ensure_ascii=False)
    try:
        return json.dumps(function(**arguments), ensure_ascii=False)
    except VectorIndexNotReadyError as error:
        return json.dumps({"error": str(error)}, ensure_ascii=False)


def ask_agent(question: str, area: str = "전국") -> dict[str, Any]:
    """Let the model call the same search tools exposed by the MCP server."""
    recommendation = _get_recommendations(question, area)
    if not OPENAI_API_KEY:
        return _local_rag_answer(question, area)

    client = OpenAI(api_key=OPENAI_API_KEY)
    facilities_by_id: dict[str, dict[str, Any]] = {
        str(facility["id"]): facility
        for facility in recommendation["facilities"]
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "당신은 공공시설 검색 도우미입니다. 시설 검색과 상세 조회는 반드시 도구를 "
                "사용하고, 도구 결과에 없는 정보를 추측하지 마세요. 답변은 한국어로 간단히 작성하세요."
            ),
        },
        {"role": "user", "content": f"선택 지역: {area}\n질문: {question}"},
    ]

    try:
        for _ in range(3):
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return {
                    "answer": message.content or "검색 결과를 바탕으로 답변을 만들지 못했습니다.",
                    "facilities": list(facilities_by_id.values()),
                    "criteria": recommendation["criteria"],
                }

            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                tool_result = _call_tool(tool_call.function.name, arguments)
                if tool_call.function.name == "search_facilities":
                    payload = json.loads(tool_result)
                    for facility in payload.get("results", []):
                        if facility.get("id") is not None:
                            facilities_by_id[str(facility["id"])] = facility
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
    except (OpenAIError, json.JSONDecodeError, TypeError, ValueError) as error:
        return {
            "answer": f"AI 에이전트 처리 중 오류가 발생했습니다: {error}",
            "facilities": list(facilities_by_id.values()),
            "criteria": recommendation["criteria"],
        }

    return {
        "answer": "검색 도구 호출 횟수를 초과했습니다. 질문을 나누어 다시 시도해 주세요.",
        "facilities": list(facilities_by_id.values()),
        "criteria": recommendation["criteria"],
    }
