from fastapi import APIRouter, HTTPException
from core.contracts import ToyAskRequest, ToyAskResponse
from lc.router import Graph
from app.settings import APPSETTINGS

router = APIRouter()


@router.post("/toy/ask", response_model=ToyAskResponse)
def toy_ask(req: ToyAskRequest):
    if not APPSETTINGS.toy.enabled:
        raise HTTPException(status_code=403, detail="Toy disabled")
    runnable = Graph()
    out: ToyAskResponse = runnable.invoke(req.model_dump())
    return out
