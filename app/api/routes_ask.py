from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.deps import enforce_json_size
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class AskRequest(BaseModel):
    session_id: str
    question: str
    k: int = 8
    lang: str = "auto"

# Compile graph once at module load
_graph = None
def _get_graph():
    global _graph
    if _graph is None:
        from lc.agents.graph import build_core_graph
        _graph = build_core_graph()
    return _graph

@router.post("/ask")
def ask(req: AskRequest, _=Depends(enforce_json_size)):
    if not req.session_id or not req.question:
        raise HTTPException(status_code=400, detail="Missing session_id or question")
    
    try:
        graph = _get_graph()
        result = graph.invoke({
            "question": req.question,
            "session_id": req.session_id,
            "context": [],
            "context_text": "",
            "draft_answer": "",
            "final_answer": "",
            "attempts": 0,
            "next_action": "",
            "search_meta": {},
            "critic_score": "",
            "critic_feedback": "",
            "relevancy_score": "",
        })
        
        answer = result.get("final_answer", "") or result.get("draft_answer", "Không có câu trả lời.")
        next_action = result.get("next_action", "search")
        
        # Build response
        response = {
            "answer": answer,
            "next_action": next_action,
            "critic_score": result.get("critic_score", ""),
            "attempts": result.get("attempts", 0),
        }
        
        # Add citations only for search mode
        if next_action == "search":
            context = result.get("context", [])
            footnotes = []
            seen = set()
            for i, doc in enumerate(context):
                fname = doc.get("file_name", "")
                page = doc.get("page_idx")
                key = f"{fname}_{page}"
                if key not in seen and fname:
                    seen.add(key)
                    footnotes.append({
                        "n": i + 1,
                        "heading": fname,
                        "page": page,
                    })
            response["footnotes"] = footnotes
        else:
            response["footnotes"] = []
            
        return response
        
    except Exception as e:
        logger.error(f"Graph invoke failed: {e}", exc_info=True)
        # Fallback to old qa_chain if graph fails
        try:
            from lc.chains.qa_chain import answer_with_citation
            res = answer_with_citation(req.session_id, req.question, k=req.k)
            if res.get("footnotes"):
                from core.citation.citation import render_footnotes
                src = render_footnotes(res["footnotes"])
                lang = res.get("lang", "vi")
                prefix = "Nguồn" if lang == "vi" else "Source"
                res["answer"] = f"{res['answer']}\n\n{prefix}: {src}"
            return res
        except Exception as e2:
            logger.error(f"Fallback qa_chain also failed: {e2}", exc_info=True)
            return {"answer": f"Xin lỗi, hệ thống gặp lỗi: {str(e)}", "footnotes": []}

class SumRequest(BaseModel):
    session_id: str
    mode: str   
    question: str | None = None
    k: int = 10

@router.post("/summarize_tldr")
def summarize_tldr(req: SumRequest):
    from lc.chains.summarize_chain import summarize_mode
    res = summarize_mode(req.session_id, "tldr", req.question, k=req.k)
    return res

@router.post("/summarize_exec")
def summarize_exec(req: SumRequest):
    from lc.chains.summarize_chain import summarize_mode
    res = summarize_mode(req.session_id, "executive", req.question, k=req.k)
    return res

@router.post("/summarize_qfs")
def summarize_qfs(req: SumRequest):
    from lc.chains.summarize_chain import summarize_mode
    res = summarize_mode(req.session_id, "qfs", req.question, k=req.k)
    return res

