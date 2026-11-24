from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import BACKEND_DIR
from app.services.agent import ask_agent
from app.services.facility_data import (
    AREA_OPTIONS,
    count_facilities,
    filter_facility_records,
    get_facility,
    get_facility_count,
    get_filter_options,
    get_valid_map_markers,
    search_facilities,
)
from app.services.vector_store import (
    VectorIndexNotReadyError,
    search_similar_facilities,
)
from app.services.statistics import get_dashboard_statistics

router = APIRouter()
templates = Jinja2Templates(directory=str(BACKEND_DIR / "app" / "templates"))


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    facility_types, fee_options = get_filter_options()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "areas": AREA_OPTIONS,
            "facility_types": facility_types,
            "fee_options": fee_options,
            "operating_options": ("전체", "평일 운영", "주말 운영"),
            "total_count": get_facility_count(),
            "selected_area": "전국",
            "selected_type": "",
            "selected_fee": "전체",
            "selected_operating": "전체",
        },
    )


@router.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q: str = "",
    area: str = "전국",
    mode: str = "structured",
    facility_type: str = "",
    fee: str = "전체",
    operating: str = "전체",
    facility_name: str = "",
    place_name: str = "",
):
    if area not in AREA_OPTIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 지역입니다.")
    if mode not in ("structured", "vector"):
        raise HTTPException(status_code=400, detail="지원하지 않는 검색 방식입니다.")
    facility_types, fee_options = get_filter_options()
    if facility_type and facility_type not in facility_types:
        raise HTTPException(status_code=400, detail="지원하지 않는 시설 유형입니다.")
    if fee not in fee_options:
        raise HTTPException(status_code=400, detail="지원하지 않는 이용료 조건입니다.")
    operating_options = ("전체", "평일 운영", "주말 운영")
    if operating not in operating_options:
        raise HTTPException(status_code=400, detail="지원하지 않는 운영일 조건입니다.")

    error = None
    if mode == "vector":
        try:
            results = search_similar_facilities(q, area=area, limit=300)
            results = filter_facility_records(
                results,
                area=area,
                facility_type=facility_type,
                fee=fee,
                operating=operating,
                facility_name=facility_name,
                place_name=place_name,
                query=q,
            )[:30]
        except VectorIndexNotReadyError as exception:
            results = []
            error = str(exception)
    else:
        results = search_facilities(
            q,
            area=area,
            facility_type=facility_type,
            fee=fee,
            operating=operating,
            facility_name=facility_name,
            place_name=place_name,
            limit=30,
        )
    total_matches = count_facilities(
        q,
        area,
        facility_type=facility_type,
        fee=fee,
        operating=operating,
        facility_name=facility_name,
        place_name=place_name,
    ) if mode == "structured" else None
    map_markers = get_valid_map_markers(results)
    back_url = request.url.path
    if request.url.query:
        back_url = f"{back_url}?{request.url.query}"

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "areas": AREA_OPTIONS,
            "facility_types": facility_types,
            "fee_options": fee_options,
            "operating_options": operating_options,
            "q": q,
            "area": area,
            "mode": mode,
            "selected_type": facility_type,
            "selected_fee": fee,
            "selected_operating": operating,
            "facility_name": facility_name,
            "place_name": place_name,
            "results": results,
            "total_matches": total_matches,
            "error": error,
            "back_url": back_url,
            "map_markers": map_markers,
        },
    )


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, area: str = "전국", facility_type: str = ""):
    facility_types, _ = get_filter_options()
    if area not in (*AREA_OPTIONS,):
        raise HTTPException(status_code=400, detail="지원하지 않는 지역입니다.")
    if facility_type and facility_type not in facility_types:
        raise HTTPException(status_code=400, detail="지원하지 않는 시설 유형입니다.")
    statistics = get_dashboard_statistics(area=area, facility_type=facility_type)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "areas": (*AREA_OPTIONS,),
            "facility_types": facility_types,
            "selected_area": area,
            "selected_type": facility_type,
            "statistics": statistics,
        },
    )


@router.get("/facilities/{facility_id}", response_class=HTMLResponse)
def facility_detail(request: Request, facility_id: str, back: str = ""):
    facility = get_facility(facility_id)
    if facility is None:
        raise HTTPException(status_code=404, detail="공공시설을 찾을 수 없습니다.")
    back_url = back if back.startswith("/") else "/"
    return templates.TemplateResponse(
        request=request,
        name="detail.html",
        context={"facility": facility, "back_url": back_url},
    )


@router.post("/ask", response_class=HTMLResponse)
def ask(
    request: Request,
    question: str = Form(..., min_length=1, max_length=1000),
    area: str = Form("전국"),
):
    if area not in AREA_OPTIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 지역입니다.")
    response = ask_agent(question.strip(), area)
    facility_types, fee_options = get_filter_options()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "areas": AREA_OPTIONS,
            "facility_types": facility_types,
            "fee_options": fee_options,
            "operating_options": ("전체", "평일 운영", "주말 운영"),
            "total_count": get_facility_count(),
            "question": question,
            "answer": response["answer"],
            "facilities": response["facilities"],
            "criteria": response.get("criteria"),
            "selected_area": area,
        },
    )
