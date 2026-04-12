import torch
import clip
from PIL import Image
import json

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device=device)

# 1. 카테고리별 태그 맵 정의
categories = {
    "피사체": {
        "인물": "a portrait of a human, a close-up of a person, a clear shot of a person's face",
        "풍경": "a scenic landscape photo with a wide view",
        "야경": "a photo taken at night with artificial lights",
        "도시": "a cityscape with many tall buildings and skyscrapers",
        "스트리트": "a street photography shot with roads and sidewalks",
        "건물": "a photo focusing on architectural structures and buildings",
        "하늘": "a clear view of the sky, clouds, or blue atmosphere",
        "바다": "a photo of the ocean, sea, or a large body of water",
        "산": "a photo of a mountain peak, majestic mountain range",
        "꽃": "a close-up photo of a blooming flower",
        "별": "a night sky filled with many bright stars",
        "식물": "a photo of green plants, leaves, or vegetation",
        "동물": "a photo of a pet or wild animal",
        "음식": "a photo of delicious food on a plate",
        "정물": "a photo of inanimate objects arranged on a surface",
        "윤슬": "shining and sparkling sunlight reflecting on water surface"
    },
    "구도": {
        "3분할": "a photo following the rule of thirds composition",
        "중앙 배치": "a subject placed exactly in the center of the frame",
        "대칭": "a perfectly symmetrical photo with balance",
        "비대칭": "an asymmetrical composition with intentional imbalance",
        "여백": "a photo with a lot of empty negative space",
        "프레이밍": "a photo where the subject is framed by surrounding objects",
        "소실점": "a photo with a strong vanishing point leading into the distance",
        "레이어": "a photo with distinct foreground, middle ground, and background layers",
        "패턴": "a photo with repetitive geometric patterns or shapes",
        "대비": "a photo with high contrast between light and dark areas",
        "클로즈업": "a tight close-up shot of a specific detail",
        "로우앵글": "a low angle shot looking up at the subject",
        "하이앵글": "a high angle shot looking down from above",
        "황금비율": "a photo following the golden ratio spiral composition",
        "리딩라인": "a photo with strong leading lines directing the eye",
        "반영": "a clear reflection of a subject in water or a mirror"
    },
    "촬영법": {
        "장노출": "a long exposure photo with motion blur",
        "역광": "a backlit photo with the sun behind the subject",
        "직광": "a photo with direct harsh lighting on the subject",
        "패닝샷": "a panning shot with a sharp subject and a blurred background",
        "보케": "a photo with soft out-of-focus background blur",
        "연출": "a carefully staged and directed professional photo",
        "접사": "an extreme macro photography shot of tiny details",
        "파노라마": "a very wide panoramic view",
        "틸트": "a photo with a tilt-shift miniature effect",
        "어안": "a photo taken with a wide-angle fisheye lens"
    },
    "색감": {
        "흑백": "a black and white photo, grayscale image, no color",
        "웜톤": "a photo with warm golden and orange color tones",
        "쿨톤": "a photo with cool blue and crisp color tones"
    },
    "시간": {
        "봄": "a photo with spring atmosphere and blossoms",
        "여름": "a photo with bright summer sun and lush greenery",
        "가을": "a photo with autumn colors, red and yellow leaves",
        "겨울": "a photo with cold winter atmosphere and snow",
        "골든아워": "a photo taken during the warm golden hour of sunrise or sunset",
        "블루아워": "a photo taken during the blue hour of dawn or twilight"
    },
    "기타": {
        "카페": "an interior photo of a cozy cafe",
        "스튜디오": "a professional photo taken inside a photo studio",
        "비": "a photo of rain falling or wet surfaces",
        "눈": "a photo of falling snow or snow-covered ground",
        "안개": "a photo with thick fog or misty atmosphere",
        "빛": "a photo with prominent and beautiful light sources",
        "그림자": "a photo with strong and dramatic shadows",
        "필름": "a photo with vintage film grain and aesthetic",
        "노을": "a beautiful sunset with orange and red sky",
        "달": "a clear photo of the moon in the night sky"
    }
}

# 2. 데이터 정리
tag_list_kr, tag_list_en, tag_to_category = [], [], {}
for cat_name, tags in categories.items():
    for kr, en in tags.items():
        tag_list_kr.append(kr)
        tag_list_en.append(en)
        tag_to_category[kr] = cat_name

category_priority = ["피사체", "구도", "촬영법", "색감", "시간", "기타"]
text_inputs = clip.tokenize(tag_list_en).to(device)

THRESHOLD, MAX_CANDIDATES, MAX_FINAL_TAGS = 0.025, 10, 5

def get_image_tags(image_path: str) -> dict:
    try:
        image_raw = Image.open(image_path)
        image_input = preprocess(image_raw).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return {"tags": []}

    with torch.no_grad():
        logits_per_image, _ = model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    tag_results = []
    for i in range(len(tag_list_kr)):
        tag_results.append({
            "tag": tag_list_kr[i],
            "score": probs[i],
            "category": tag_to_category[tag_list_kr[i]]
        })

    top_candidates = sorted(tag_results, key=lambda x: x['score'], reverse=True)[:MAX_CANDIDATES]
    valid_candidates = [t for t in top_candidates if t['score'] >= THRESHOLD]
    if not valid_candidates: valid_candidates = [top_candidates[0]]

    final_sorted = sorted(valid_candidates, key=lambda x: (category_priority.index(x['category']), -x['score']))
    
    final_tags = []
    for item in final_sorted:
        if item['tag'] not in final_tags:
            final_tags.append(item['tag'])
        if len(final_tags) >= MAX_FINAL_TAGS:
            break
    return {"tags": final_tags}

if __name__ == "__main__":
    example_image_path = "/media/sea.jpg"
    print(get_image_tags(example_image_path))
