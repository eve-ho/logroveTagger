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

---

# CLIP: Contrastive Language-Image Pre-training

OpenAI에서 개발한 **CLIP**은 이미지와 텍스트를 동일한 벡터 공간 내에서 연결하여, 자연어 설명만으로 시각적 개념을 학습하고 분류할 수 있는 혁신적인 모델입니다.

## 1. 기반 기술 (Foundations)
CLIP은 특정 데이터셋에 국한되지 않고 인터넷상의 방대한 지식을 흡수하도록 설계되었습니다.

* **Dataset:** 인터넷에서 수집한 약 **4억 개의 이미지-텍스트 쌍(Image-Text Pairs)**을 기반으로 학습되었습니다.
* **Architecture:** * **Image Encoder:** ResNet 또는 ViT(Vision Transformer)를 사용하여 이미지 특징 추출.
    * **Text Encoder:** Transformer 기반의 언어 모델을 사용하여 텍스트 맥락 파악.

## 2. 작동 원리 (How it works)
CLIP은 **대조 학습(Contrastive Learning)** 방식을 채택하여 이미지와 텍스트 사이의 상관관계를 이해합니다.

1.  **Multi-Modal Embedding:** 이미지와 텍스트를 각각의 인코더에 통과시켜 동일한 차원의 벡터 공간에 투영합니다.
2.  **Pair Alignment:** 실제 매칭되는 '이미지-텍스트' 쌍은 코사인 유사도(Cosine Similarity)가 최대화되도록 가깝게 당깁니다.
3.  **Noise Reduction:** 매칭되지 않는 나머지 조합들은 서로 멀어지도록 배치하여 모델이 문맥을 정교하게 구분하게 합니다.

## 3. 주요 장점 (Key Advantages)

* **Zero-shot Transfer:** 별도의 추가 학습(Fine-tuning) 없이도 새로운 데이터셋이나 사물을 텍스트 설명만으로 즉시 분류할 수 있습니다.
* **Robustness:** 훈련 데이터에 최적화된 기존 모델들과 달리, 스케치, 추상화, 실제 사진 등 다양한 형태의 시각 정보에 대해 매우 강력한 일반화 성능을 보입니다.
* **Flexibility:** "A photo of a [Object]"와 같은 프롬프트를 변경하는 것만으로도 분류기(Classifier)의 성격을 자유롭게 바꿀 수 있습니다.

## 4. 활용 사례 (Use Cases)

* **Generative AI:** DALL-E, Stable Diffusion 등 생성 모델에서 사용자의 텍스트 입력을 이미지 생성 가이드로 변환하는 핵심 두뇌 역할.
* **Semantic Search:** 키워드 매칭을 넘어 "해질녘 해변에서 조깅하는 강아지"와 같은 문장 단위의 고차원 이미지 검색 구현.
* **Auto Tagging:** 이미지의 속성, 장르, 분위기 등을 파악하여 자동으로 태그를 생성하는 시스템.

---

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

