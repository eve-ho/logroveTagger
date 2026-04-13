### **📝 Logrove AI Tagger - README.md**

```markdown
# 🏆 Logrove AI Tagger
> **Logrove** 프로젝트의 사진 학습 및 커뮤니티 활성화를 위한 AI 사진 분석 엔진입니다.

본 마이크로서비스는 사용자가 업로드한 사진의 **피사체, 구도, 촬영 기법, 분위기**를 분석하여 최적의 태그를 자동으로 추출합니다.

---

## 🛠 Tech Stack
- **Language**: Python 3.9+
- **Framework**: FastAPI
- **AI Model**: OpenAI CLIP (ViT-B/32)
- **Deep Learning**: PyTorch
- **Image Processing**: Pillow (PIL)

## 🧠 Key Features & Logic
단순한 이미지 분류를 넘어 사진학적 관점의 태깅을 수행합니다.

1. **Zero-shot 태깅**: 학습되지 않은 새로운 태그도 CLIP 모델의 문맥 이해 능력을 통해 추출 가능합니다.
2. **카테고리 우선순위 정렬**: 
   - `피사체 > 구도 > 촬영법 > 색감 > 시간 > 기타` 순으로 정렬하여 사용자에게 논리적인 정보를 제공합니다.
3. **신뢰도 필터링**: `THRESHOLD = 0.025` 설정을 통해 연관성이 낮은 태그는 필터링하고 핵심 태그 5개만 반환합니다.

## 🚀 How to Run (Local)

### 1. Requirements 설치
```bash
pip install torch torchvision clip pillow fastapi uvicorn python-multipart
```

### 2. 서버 실행
```bash
python main.py
```

### 3. API 테스트
- 서버 실행 후 `http://localhost:8000/docs` 접속 (Swagger UI)

## 📂 API Specifications
### **POST** `/analyze-tags`
- **Request**: `multipart/form-data` (Image file)
- **Response**: 
  ```json
  {
    "tags": ["바다", "3분할", "골든아워", "윤슬"]
  }
  ```
```

