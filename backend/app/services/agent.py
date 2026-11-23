import json
from typing import Any

from openai import OpenAI, OpenAIError

from app.core.config import OPENAI_API_KEY, OPENAI_MODEL
from app.services.mcp_tools import get_facility_details, search_facilities
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
                        "enum": ["전국", "서울특별시", "경기도", "서울특별시+경기도"],
                    },
                    "mode": {"type": "string", "enum": ["structured", "vector"]},
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


def _local_rag_answer(question: str, area: str) -> dict[str, Any]:
    try:
        result = search_facilities(question, area=area, mode="vector", limit=5)
    except VectorIndexNotReadyError as error:
        return {"answer": str(error), "facilities": []}

    facilities = result["results"]
    if not facilities:
        return {
            "answer": "질문과 가까운 공공시설을 찾지 못했습니다. 지역이나 시설 종류를 바꿔 다시 검색해 주세요.",
            "facilities": [],
        }
    return {
        "answer": "질문과 관련된 공공시설을 찾았습니다.",
        "facilities": facilities,
    }


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
    if not OPENAI_API_KEY:
        return _local_rag_answer(question, area)

    client = OpenAI(api_key=OPENAI_API_KEY)
    facilities_by_id: dict[str, dict[str, Any]] = {}
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
        }

    return {
        "answer": "검색 도구 호출 횟수를 초과했습니다. 질문을 나누어 다시 시도해 주세요.",
        "facilities": list(facilities_by_id.values()),
    }
