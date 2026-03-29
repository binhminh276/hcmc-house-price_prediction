import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Dự đoán giá nhà TP.HCM", layout="wide")

@st.cache_resource
def load_all_models():
    price_models = {
        "Raw - Random Forest": joblib.load('models/raw_model/rf_pipeline.pkl'),
        "Raw - XGBoost": joblib.load('models/raw_model/xgb_pipeline.pkl'),
        "Tuned - Random Forest": joblib.load('models/tuned_model/best_rf_model.pkl'),
        "Tuned - XGBoost": joblib.load('models/tuned_model/best_xgb_model.pkl')
    }
    return price_models

price_models = load_all_models()

def feature_engineering(data):
    df_fe = data.copy()
    
    bins = [0, 30, 120, np.inf]
    labels = ['Duoi_30', 'Tu_30_den_120', 'Tren_120']
    df_fe['Phan_khuc_dien_tich'] = pd.cut(df_fe['Diện tích'], bins=bins, labels=labels)
    
    df_fe['Ty_le_Bath_Bed'] = df_fe['Số phòng tắm, vệ sinh'] / (df_fe['Số phòng ngủ'] + 1e-5)
    df_fe['Loai_hinh_Quan'] = df_fe['Cao tầng'].astype(str) + '_' + df_fe['Quận']
    
    df_fe['Tong_tien_ich'] = df_fe['Gần bệnh viện'] + df_fe['Gần chợ'] + df_fe['Gần trường học']
    
    df_fe = df_fe.drop(columns=['Gần bệnh viện', 'Gần chợ', 'Gần trường học'])
    
    return df_fe

st.title("🏡 Dự đoán giá Bất động sản tại TP.HCM")

selected_model_name = st.selectbox(
    "Chọn mô hình dự đoán:", 
    options=list(price_models.keys())
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    dien_tich = st.number_input("Diện tích (m2)", min_value=1.0, value=50.0, step=1.0)
    if dien_tich < 10:
        st.warning("⚠️ Diện tích < 10m2 nằm ngoài phạm vi học của mô hình.")
        
    phong_ngu = st.number_input("Số phòng ngủ", min_value=1, value=2, step=1)
    phong_tam = st.number_input("Số phòng tắm, vệ sinh", min_value=1, value=1, step=1)
    cao_tang = st.number_input("Số tầng (Cao tầng)", min_value=1, value=1, step=1)

with col2:
    quan = st.selectbox("Quận/Huyện", ["Quận 1", "Quận 2", "Quận 3", "Quận 7", "Quận 9", "Bình Thạnh", "Thủ Đức", "Khác"]) 
    
    st.markdown("**Tiện ích xung quanh:**")
    gan_bv = st.checkbox("Gần bệnh viện")
    gan_cho = st.checkbox("Gần chợ")
    gan_th = st.checkbox("Gần trường học")

if st.button("Dự đoán giá", type="primary"):
    input_data = {
        'Diện tích': [dien_tich],
        'Số phòng ngủ': [phong_ngu],
        'Số phòng tắm, vệ sinh': [phong_tam],
        'Cao tầng': [cao_tang],
        'Quận': [quan],
        'Gần bệnh viện': [int(gan_bv)],
        'Gần chợ': [int(gan_cho)],
        'Gần trường học': [int(gan_th)]
    }
    df_input = pd.DataFrame(input_data)
    
    try:
        model_to_use = price_models[selected_model_name]
        
        if "Raw" in selected_model_name:
            pred_price = model_to_use.predict(df_input)[0]
            real_price = pred_price
            
            st.info("🔄 Đang chạy luồng Raw: Dùng dữ liệu thô & Không áp dụng đảo Log.")
            
            with st.expander("Xem chi tiết dữ liệu đưa vào mô hình (Raw Data)"):
                st.dataframe(df_input)

        else:
            df_processed = feature_engineering(df_input)
            
            log_price = model_to_use.predict(df_processed)[0]
            real_price = np.expm1(log_price)
            
            st.info("⚙️ Đang chạy luồng Tuned: Đã áp dụng Feature Engineering & Đảo Log giá (expm1).")
            
            with st.expander("Xem chi tiết dữ liệu đưa vào mô hình (Sau Feature Engineering)"):
                st.dataframe(df_processed)
        
        if real_price < 0:
            st.error("Mô hình trả về giá trị âm do thông số nằm quá xa dữ liệu thực tế đã học!")
        else:
            formatted_price = "{:,.0f} VNĐ".format(real_price).replace(',', '.')
            st.success("Hoàn tất tính toán!")
            st.metric(label=f"Ước tính giá trị (theo {selected_model_name}):", value=formatted_price)

    except Exception as e:
        st.error(f"Lỗi hệ thống: {e}")