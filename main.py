from fastapi import FastAPI, UploadFile, File, HTTPException
from image_tagger import get_image_tags # 호진님의 엔진 불러오기
import os
import shutil
import uuid

app = FastAPI()

# 임시 파일 저장 경로
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/analyze-tags")
async def analyze_image(file: UploadFile = File(...)):
    # 1. 파일 확장자 체크 (이미지만 허용)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    # 2. 서버에 임시 저장 (고유 파일명 생성)
    file_extension = os.path.splitext(file.filename)[1]
    temp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_extension}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. AI 엔진 가동
        result = get_image_tags(temp_file_path)

        return result # {"tags": ["바다", "골든아워", ...]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 4. 분석 완료 후 임시 파일 삭제 (용량 관리)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)