# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1uj2lD8goJDLo9uSg_8HcT4bxnl2trPc8")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
    labels[0]:{"texts":["중국식 냉면은 맛있어"],"images":["https://www.esquirekorea.co.kr/resources_old/online/org_online_image/eq/ae3b94d6-fed6-4144-a18e-8e92a094ee0b.jpg"]},
    labels[1]:{"texts":["짜장면은 맛있어"],"images":["https://image.8dogam.com/resized/product/asset/v1/upload/6833c73eae0949eb8bcbed560c903198.jpeg?type=big&res=3x&ext=jpg"]},
        labels[2]:{"texts":["짬뽕은 매워"],"images":["data:image/jpeg;base64,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"]},
labels[3]:{"texts":["탕수육은 바삭해"],"images":["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmUiqnPdBD0Zu2rHOXB7RaTI1Te33tsf4T5Q&s"]}
}


# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
