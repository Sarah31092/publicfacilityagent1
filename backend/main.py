from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import BACKEND_DIR
from app.routers.web import router

app = FastAPI(title="AI 공공시설 검색 에이전트")
app.mount(
    "/static",
    StaticFiles(directory=str(BACKEND_DIR / "app" / "static")),
    name="static",
)
app.include_router(router)
