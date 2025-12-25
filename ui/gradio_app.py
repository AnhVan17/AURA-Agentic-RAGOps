import os, json, time, threading
from typing import List, Tuple, Dict
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import requests

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TIMEOUT = 60 # Tăng timeout lên 60s để AI kịp trả lời dài

# ------- Helpers -------
def _post_json(path: str, payload: dict, timeout=TIMEOUT):
    url = f"{API_BASE}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _post_file(path: str, session_id: str, fpath: str, timeout=120):
    url = f"{API_BASE}{path}"
    with open(fpath, "rb") as f:
        files = {"file": (os.path.basename(fpath), f, "application/octet-stream")}
        params = {"session_id": session_id}
        r = requests.post(url, params=params, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _render_citations_html(answer: str, footnotes: List[dict], lang: str="vi") -> str:
    # biến [n] thành superscript có anchor
    html = answer
    for fn in footnotes:
        n = fn["n"]
        html = html.replace(f"[{n}]", f"<sup><a href=\"#src-{n}\">[{n}]</a></sup>")
    # footnotes
    lines = []
    if footnotes:
        tail = "Nguồn" if lang in ("vi","auto") else "Sources"
        lines.append(f"<hr/><div><b>{tail}:</b></div><ol style='margin-top:4px;'>")
        for fn in footnotes:
            h = (fn.get("heading") or "").strip() or "(no heading)"
            p = fn.get("page")
            extra = f" — trang {p}" if (p and lang!='en') else (f" — page {p}" if p else "")
            lines.append(f"<li id='src-{fn['n']}'><code>[{fn['n']}]</code> {h}{extra}</li>")
        lines.append("</ol>")
    return f"<div style='line-height:1.5'>{html}</div>" + "\n".join(lines)

# ------- Tab: Summarize -------
def ingest_file(session_id: str, file: gr.File) -> str:
    if not file: return "❗ Vui lòng chọn file."
    try:
        res = _post_file("/session/upload", session_id, file.name)
        return f"✅ Đã ingest {os.path.basename(file.name)} → chunks={res['counts']['chunks']}"
    except Exception as e:
        return f"❌ Ingest lỗi: {e}"

def do_summarize(session_id: str, mode: str, question: str) -> str:
    path = {
        "TL;DR": "/summarize_tldr",
        "Executive": "/summarize_exec",
        "QFS": "/summarize_qfs"
    }[mode]
    payload = {"session_id": session_id, "mode": mode.lower(), "question": question or None, "k": 10}
    try:
        res = _post_json(path, payload)
        return res.get("output", "")
    except Exception as e:
        return f"❌ Summarize lỗi: {e}"

# ------- Tab: Chat -------
def chat_upload_files(session_id: str, files: List[gr.File]):
    logs = []
    for f in (files or []):
        try:
            res = _post_file("/session/upload", session_id, f.name)
            logs.append(f"✅ {os.path.basename(f.name)} → {res['counts']['chunks']} chunks")
        except Exception as e:
            logs.append(f"❌ {os.path.basename(f.name)} → lỗi: {e}")
    return "\n".join(logs) if logs else "Không có file nào được tải lên."

def chat_ask(session_id: str, lang: str, question: str, history: List[Dict[str, str]]):
    if not question.strip():
        return history, gr.update(value="")
    payload = {"session_id": session_id, "question": question, "k": 8, "lang": lang}
    try:
        res = _post_json("/ask", payload)
        ans = res.get("answer", "")
        fns = res.get("footnotes", [])
        html = _render_citations_html(ans, fns, lang=lang)
        
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": html})
        return history, gr.update(value="")
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ Lỗi: {e}"})
        return history, gr.update(value="")

# ------- Build UI -------
with gr.Blocks(css="""
    body { background-color: #f9fafb; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    .gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding: 20px !important; }
    #chatbot { border-radius: 12px; border: 1px solid #e5e7eb; background: white !important; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .message.user { background: #f3f4f6 !important; border-radius: 12px !important; border: none !important; }
    .message.bot { background: #ffffff !important; border-radius: 12px !important; border: 1px solid #f3f4f6 !important; }
    sup a { text-decoration: none; color: #2563eb; font-weight: bold; margin-left: 2px; }
    sup a:hover { text-decoration: underline; }
    .primary-btn { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; color: white !important; border: none !important; transition: all 0.2s; }
    .primary-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }
    .tabs-nav { border-bottom: 2px solid #e5e7eb !important; }
    .tabitem { padding: 20px !important; }
    hr { margin: 15px 0; border: 0; border-top: 1px solid #eee; }
""", title="Academic OCR RAG Chatbot") as demo:

    gr.Markdown("## 📚 Academic OCR RAG — UI (v1.0)")
    with gr.Tabs():
        with gr.TabItem("Summarize"):
            with gr.Row():
                session_sum = gr.Textbox(label="Session ID", value="S1", scale=1)
                mode = gr.Radio(choices=["TL;DR","Executive","QFS"], value="TL;DR", label="Mode", scale=2)
            with gr.Row():
                file_sum = gr.File(label="Upload 1 file", file_count="single")
                btn_ingest = gr.Button("Ingest file", variant="primary")
            ingest_log = gr.Textbox(label="Log", interactive=False)

            with gr.Row():
                q_opt = gr.Textbox(label="(Optional) Câu hỏi/tiêu điểm cho QFS/Executive", placeholder="Ví dụ: Tập trung định nghĩa entropy")
                btn_sum = gr.Button("Summarize", variant="primary")
            sum_out = gr.Markdown(label="Kết quả")

            btn_ingest.click(fn=ingest_file, inputs=[session_sum, file_sum], outputs=ingest_log)
            btn_sum.click(fn=do_summarize, inputs=[session_sum, mode, q_opt], outputs=sum_out)

        with gr.TabItem("Chat with files"):
            with gr.Row():
                session_chat = gr.Textbox(label="Session ID", value="S1")
                lang = gr.Radio(choices=["auto","vi","en"], value="auto", label="Answer language")
            files_chat = gr.File(label="Upload 1..n file", file_count="multiple")
            btn_upload = gr.Button("Ingest selected files", variant="primary")
            upload_log = gr.Textbox(label="Ingest Log", interactive=False)

            gr.Markdown("---")
            chat = gr.Chatbot(
                elem_id="chatbot", 
                bubble_full_width=False, 
                height=500, 
                avatar_images=(None, None), 
                render_markdown=True,
                type="messages"
            )
            user_in = gr.Textbox(placeholder="Đặt câu hỏi...", label="Your question")
            with gr.Row():
                btn_send = gr.Button("Send", variant="primary")
                btn_clear = gr.Button("Clear chat")

            btn_upload.click(fn=chat_upload_files, inputs=[session_chat, files_chat], outputs=upload_log)
            btn_send.click(fn=chat_ask, inputs=[session_chat, lang, user_in, chat], outputs=[chat, user_in])
            btn_clear.click(lambda: [], outputs=chat)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
