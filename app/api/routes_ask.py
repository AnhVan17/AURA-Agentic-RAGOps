from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from lc.chains.qa_chain import answer_with_citation
from lc.chains.summarize_chain import summarize_mode
from app.deps import enforce_json_size

router = APIRouter()

class AskRequest(BaseModel):
    session_id: str
    question: str
    k: int = 8
    lang: str = "auto"

@router.post("/ask")
def ask(req: AskRequest, _=Depends(enforce_json_size)):
    if not req.session_id or not req.question:
        raise HTTPException(status_code=400, detail="Missing session_id or question")
    
    res = answer_with_citation(req.session_id, req.question, k=req.k)
    
    # Tối ưu hóa song ngữ cho phần hiển thị Nguồn (Footnotes)
    if res.get("footnotes"):
        from core.citation.citation import render_footnotes
        src = render_footnotes(res["footnotes"])
        
        # Quyết định dùng chữ "Nguồn" hay "Source" dựa trên ngôn ngữ đã phát hiện
        lang = res.get("lang", "vi")
        prefix = "Nguồn" if lang == "vi" else "Source"
        
        res["answer"] = f"{res['answer']}\n\n{prefix}: {src}"
        
    return res

class SumRequest(BaseModel):
    session_id: str
    mode: str   
    question: str | None = None
    k: int = 10

@router.post("/summarize_tldr")
def summarize_tldr(req: SumRequest):
    res = summarize_mode(req.session_id, "tldr", req.question, k=req.k)
    return res

@router.post("/summarize_exec")
def summarize_exec(req: SumRequest):
    res = summarize_mode(req.session_id, "executive", req.question, k=req.k)
    return res

@router.post("/summarize_qfs")
def summarize_qfs(req: SumRequest):
    res = summarize_mode(req.session_id, "qfs", req.question, k=req.k)
    return res
