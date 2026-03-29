import gradio as gr
import pandas as pd
import joblib
import numpy as np

KMEANS_MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler.pkl'

RF_RAW_PATH = 'rf_pipeline.pkl'
XGB_RAW_PATH = 'xgb_pipeline.pkl'
RF_TUNED_PATH = 'best_rf_model.pkl'
XGB_TUNED_PATH = 'best_xgb_model.pkl'

DATASET_PATH = 'kmeans_clustered_full.csv'

def load_models():
    try:
        kmeans = joblib.load(KMEANS_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        rf_raw = joblib.load(RF_RAW_PATH)
        xgb_raw = joblib.load(XGB_RAW_PATH)
        rf_tuned = joblib.load(RF_TUNED_PATH)
        xgb_tuned = joblib.load(XGB_TUNED_PATH)
        
        print("Đã tải thành công tất cả 6 file mô hình!")
        return kmeans, scaler, rf_raw, xgb_raw, rf_tuned, xgb_tuned
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None, None, None, None, None, None

kmeans_model, scaler, rf_raw, xgb_raw, rf_tuned, xgb_tuned = load_models()

try:
    df_sample = pd.read_csv(DATASET_PATH, sep=';', decimal=',', encoding='utf-8').head(500)
    available_districts = sorted(df_sample['Quận'].dropna().unique().tolist())
except Exception as e:
    print(f"Lỗi tải dataset: {e}")
    available_districts = ["Quận 1", "Quận 2", "Quận 3", "Quận 7", "Quận 9", "Bình Thạnh", "Gò Vấp"]

def predict_all_models(district, area, price_range, beds, baths):
    if kmeans_model is None:
        return "Lỗi model", "Lỗi model", "Lỗi model", "Lỗi model", "Chưa tải được mô hình"

    try:
        # TÍNH CLUSTER
        input_cluster_df = pd.DataFrame({
            'Diện tích': [float(area)],
            'Khoảng giá': [float(price_range)]
        })
        
        input_scaled = scaler.transform(input_cluster_df)
        cluster_id = kmeans_model.predict(input_scaled)[0]
        cluster_info = f"Nhà thuộc Nhóm (Cluster): {cluster_id}"

        # CHUẨN BỊ INPUT CHO MÔ HÌNH GIÁ
        input_price_df = pd.DataFrame({
            'Quận': [district],
            'Diện tích': [float(area)],
            'Khoảng giá': [float(price_range)],
            'Số phòng ngủ': [float(beds)],
            'Số phòng tắm, vệ sinh': [float(baths)],
            'Cluster': [int(cluster_id)]
        })

        # CHẠY CẢ 4 MÔ HÌNH
        res_rf_raw = f"{round(rf_raw.predict(input_price_df)[0], 2)} Tỷ"
        res_xgb_raw = f"{round(xgb_raw.predict(input_price_df)[0], 2)} Tỷ"
        res_rf_tuned = f"{round(rf_tuned.predict(input_price_df)[0], 2)} Tỷ"
        res_xgb_tuned = f"{round(xgb_tuned.predict(input_price_df)[0], 2)} Tỷ"

        return res_rf_raw, res_xgb_raw, res_rf_tuned, res_xgb_tuned, cluster_info

    except Exception as e:
        error_msg = f"Lỗi: {e}"
        return error_msg, error_msg, error_msg, error_msg, "Lỗi tính toán Cluster"

with gr.Blocks(title="Demo 4 Mô Hình Giá Nhà TP.HCM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 So sánh Mô hình Dự báo Giá Nhà TP.HCM")
    gr.Markdown("Nhập thông số căn nhà để so sánh kết quả dự đoán giữa các mô hình trước và sau khi tinh chỉnh (Tuning).")

    with gr.Row():
        # Cột nhập liệu
        with gr.Column(scale=1):
            gr.Markdown("### 🛠 Thông số đầu vào")
            in_district = gr.Dropdown(choices=available_districts, label="Quận / Huyện", value=available_districts[0] if available_districts else None)
            in_area = gr.Number(label="Diện tích (m2)", value=60)
            in_price_range = gr.Number(label="Khoảng giá tham khảo", value=5.0) # Feature dùng cho K-Means
            in_beds = gr.Slider(minimum=1, maximum=10, step=1, label="Số phòng ngủ", value=2)
            in_baths = gr.Slider(minimum=1, maximum=10, step=1, label="Số phòng tắm, vệ sinh", value=2)
            
            predict_btn = gr.Button("🚀 Chạy tất cả mô hình", variant="primary")
            
            out_cluster = gr.Textbox(label="Thông tin Gom Cụm (Backend xử lý)", interactive=False)

        # Cột kết quả
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Kết quả so sánh")
            
            gr.Markdown("#### Mô hình Cơ bản (Raw)")
            with gr.Row():
                out_rf_raw = gr.Textbox(label="Random Forest (Raw)")
                out_xgb_raw = gr.Textbox(label="XGBoost (Raw)")
                
            gr.Markdown("#### Mô hình Đã tinh chỉnh (Tuned)")
            with gr.Row():
                out_rf_tuned = gr.Textbox(label="Random Forest (Tuned)")
                out_xgb_tuned = gr.Textbox(label="XGBoost (Tuned)")

    predict_btn.click(
        fn=predict_all_models,
        inputs=[in_district, in_area, in_price_range, in_beds, in_baths],
        outputs=[out_rf_raw, out_xgb_raw, out_rf_tuned, out_xgb_tuned, out_cluster]
    )

if __name__ == "__main__":
    demo.launch()