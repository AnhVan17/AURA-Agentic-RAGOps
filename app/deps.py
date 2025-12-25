from fastapi import Request, HTTPException, status
from app.settings import APPSETTINGS

async def enforce_json_size(request: Request):
    """
    Dependency to enforce a maximum size for JSON request bodies.
    It checks both the Content-Length header and the actual body size to prevent bypasses.
    """
    # Lấy giới hạn từ config (mặc định 64KB nếu thiếu)
    max_kb = APPSETTINGS.api.get("max_json_kb", 64)
    max_bytes = max_kb * 1024

    # 1. Kiểm tra nhanh qua Header (Early return)
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Dữ liệu yêu cầu vượt quá giới hạn cho phép ({max_kb} KB)."
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Giá trị Content-Length không hợp lệ."
            )

    # 2. Kiểm tra thực tế Body (Đề phòng trường hợp Header giả hoặc thiếu)
    # FastAPI caches the body, so this is safe to do in a dependency.
    body = await request.body()
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Kích thước dữ liệu thực tế vượt quá giới hạn ({max_kb} KB)."
        )
