from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.database import init_db
from backend.app.routers import candidates, sessions
from backend.app.services.pipeline import mark_interrupted_analyses

init_db()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    mark_interrupted_analyses()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(candidates.router)
app.include_router(sessions.router)


@app.get("/health")
def health():
    return {"status": "ok"}
