from langchain_core.runnables import RunnableBranch, RunnableLambda
from app.settings import APPSETTINGS
from core.telemetry.telemetry import timeit_stage
from core.contracts import ToyAskResponse, ToyAskRequest


try:
    import google.generativeai as genai
    if APPSETTINGS.google_api_key:
        genai.configure(api_key=APPSETTINGS.google_api_key)
        m_name = APPSETTINGS.toy.model
        if not m_name.startswith("models/"):
            m_name = f"models/{m_name}"
        model = genai.GenerativeModel(model_name=m_name)
    else:
        model = None
except ImportError:
    model = None


@timeit_stage("toy_ask")
def call_llm(prompt: str) -> str:
    if model is None:
        raise RuntimeError("Google Generative AI model is not configured.")

    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": APPSETTINGS.toy.max_output_tokens}
    )
    return response.text or ""


def build_prompt(req: ToyAskRequest) -> str:
    if req.mode == "tldr":
        return f"Bạn là trợ lý học thuật. Tóm tắt TL;DR (2–4 câu, tiếng Việt):\n\nNỘI DUNG:\n{req.text}"
    else:
        return f"Bạn là trợ lý học thuật. Trả lời câu hỏi dựa trên nội dung sau (tiếng Việt):\n\nNỘI DUNG:\n{req.text}"


def to_response(req: ToyAskRequest, out_text: str) -> ToyAskResponse:
    return ToyAskResponse(mode=req.mode, response=out_text)

# LCEL graph: (input)->(branch choose prompt)->(llm)->(wrap)


def Graph():
    # 1) Đóng gói input
    attach_input = RunnableLambda(lambda x: {"input": x})

    # 2) Chọn/đóng prompt nhưng GIỮ input
    def make_prompt(d):
        req = ToyAskRequest(**d["input"])
        return {"input": d["input"], "prompt": build_prompt(req)}

    choose = RunnableBranch(
        (lambda d: d["input"].get("mode") ==
         "tldr", RunnableLambda(make_prompt)),
        (lambda d: True, RunnableLambda(make_prompt)),
    )

    # 3) Gọi LLM nhưng GIỮ input
    llm = RunnableLambda(
        lambda d: {"input": d["input"], "raw": call_llm(d["prompt"])})

    # 4) Gói response dùng lại req
    wrap = RunnableLambda(lambda d: to_response(
        ToyAskRequest(**d["input"]), d["raw"]))

    pipe = attach_input | choose | llm | wrap
    return pipe
