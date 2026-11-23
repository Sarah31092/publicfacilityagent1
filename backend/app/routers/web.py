from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import BACKEND_DIR
from app.services.agent import ask_agent
from app.services.facility_data import (
    AREA_OPTIONS,
    count_facilities,
    get_facility,
    get_facility_count,
    search_facilities,
)
from app.services.vector_store import (
    VectorIndexNotReadyError,
    search_similar_facilities,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(BACKEND_DIR / "app" / "templates"))


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "areas": AREA_OPTIONS,
            "total_count": get_facility_count(),
            "selected_area": "전국",
        },
    )


@router.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q: str = "",
    area: str = "전국",
    mode: str = "structured",
):
    if area not in AREA_OPTIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 지역입니다.")
    if mode not in ("structured", "vector"):
        raise HTTPException(status_code=400, detail="지원하지 않는 검색 방식입니다.")

    error = None
    if mode == "vector":
        try:
            results = search_similar_facilities(q, area=area, limit=30)
        except VectorIndexNotReadyError as exception:
            results = []
            error = str(exception)
    else:
        results = search_facilities(q, area=area, limit=30)
    total_matches = count_facilities(q, area) if mode == "structured" else None

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "areas": AREA_OPTIONS,
            "q": q,
            "area": area,
            "mode": mode,
            "results": results,
            "total_matches": total_matches,
            "error": error,
        },
    )


@router.get("/facilities/{facility_id}", response_class=HTMLResponse)
def facility_detail(request: Request, facility_id: str):
    facility = get_facility(facility_id)
    if facility is None:
        raise HTTPException(status_code=404, detail="공공시설을 찾을 수 없습니다.")
    return templates.TemplateResponse(
        request=request,
        name="detail.html",
        context={"facility": facility},
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
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "areas": AREA_OPTIONS,
            "total_count": get_facility_count(),
            "question": question,
            "answer": response["answer"],
            "facilities": response["facilities"],
            "selected_area": area,
        },
    )
