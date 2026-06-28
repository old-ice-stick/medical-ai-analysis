import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path
from logic import POSProcessor, SideProcessor

# 1. 页面配置
st.set_page_config(page_title="AI 姿势角度分析", layout="wide")

st.title("基于YOLO的足部X光平片角度预测")
st.markdown("上传 X 光平片，系统将自动旋转识别并计算医学角度。")

# 2. 路径处理
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PATH_POS = MODELS_DIR / "pos.pt"
PATH_SIDE = MODELS_DIR / "side.pt"

# 3. 模型加载函数
@st.cache_resource
def load_yolo_model(model_path):
    path_str = str(model_path)
    if not os.path.exists(path_str):
        return None
    try:
        return YOLO(path_str)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

# --- 侧边栏设置 ---
with st.sidebar:
    st.header("设置")
    mode = st.radio("选择检测模式", ["正位 (POS)", "侧位 (SIDE)"])
    current_model_path = PATH_POS if mode == "正位 (POS)" else PATH_SIDE

# 加载模型
model = load_yolo_model(current_model_path)

if model is None:
    st.error(f"❌ 错误：无法在路径 {current_model_path} 找到或加载权重文件。")
    st.stop()

# --- 核心逻辑：图片上传与处理 ---
uploaded_file = st.file_uploader("上传图片", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    rotations = [
        (None, "原始方向"),
        (cv2.ROTATE_90_CLOCKWISE, "顺时针 90°"),
        (cv2.ROTATE_180, "旋转 180°"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "逆时针 90°")
    ]
    
    final_img = None
    final_kps = None
    rot_name = ""

    with st.spinner('正在进行多角度识别与旋转校正...'):
        for code, name in rotations:
            curr_img = original_img.copy() if code is None else cv2.rotate(original_img, code)
            res = model.predict(curr_img, conf=0.2, verbose=False)
            
            if res and len(res[0].keypoints) > 0:
                kps = res[0].keypoints.xy[0].tolist()
                if len(kps) > 5: 
                    final_img = curr_img
                    final_kps = kps
                    rot_name = name
                    break

    if final_kps is None:
        st.error("❌ 无法识别关键点。请确保图片清晰且属于选定的检测模式。")
    else:
        if rot_name != "原始方向":
            st.info(f"✅ 自动识别成功：已将图片旋转 **{rot_name}**")

        processor = POSProcessor() if mode == "正位 (POS)" else SideProcessor()
        analysis = processor.process(final_img, final_kps)

        # --- UI 展示 ---
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="分析用图", width=None)
        with col2:
            overview = final_img.copy()
            for i, pt in enumerate(final_kps, 1):
                if pt[0] > 0 and pt[1] > 0:
                    cv2.circle(overview, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            st.image(cv2.cvtColor(overview, cv2.COLOR_BGR2RGB), caption="关键点定位总览", width=None)

        # 1. 测量结果表格展示 (修复 ArrowTypeError 的关键区域)
        st.subheader("📊 测量结果汇总")
        if analysis['values']:
            # 创建 DataFrame
            df = pd.DataFrame(list(analysis['values'].items()), columns=['测量项目', '角度 (°)'])
            # 使用 st.table 避开 PyArrow 序列化混合类型的问题，且视觉上更像医疗报告
            st.table(df)

        # 2. 详细角度卡片展示
        st.subheader("🔍 详细角度可视化")
        
        st.markdown("""
        <style>
        .angle-card { 
            background: rgba(255,255,255,0.05); 
            border-radius: 8px; 
            padding: 10px; 
            margin: 5px 0; 
            border: 1px solid rgba(255,255,255,0.1); 
            text-align: center; 
        }
        .angle-label { color: #E0E0E0; font-size: 14px; font-weight: 600; margin-bottom: 5px; }
        .angle-number { color: #FFAA00; font-size: 24px; font-weight: 900; }
        </style>
        """, unsafe_allow_html=True)

        cols_per_row = 4
        items = list(analysis['values'].items())
        imgs = analysis['images']

        for i in range(0, len(items), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(items):
                    with cols[j]:
                        st.image(cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB), width=None)
                        st.markdown(f"""
                            <div class="angle-card">
                                <div class="angle-label">{items[idx][0]}</div>
                                <div class="angle-number">{items[idx][1]}°</div>
                            </div>
                        """, unsafe_allow_html=True)
