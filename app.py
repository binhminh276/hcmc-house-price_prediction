import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

centroids = {
    0: {'Diện tích': 275.22, 'Số phòng ngủ': 6.72, 'Số phòng tắm, vệ sinh': 5.57, 'Mặt tiền': 0.53, 'Gần bệnh viện': 0.09, 'Gần chợ': 0.19, 'Gần trường học': 0.15, 'Cao tầng': 0.75, 'Quy hoạch': 0.05},
    1: {'Diện tích': 81.92, 'Số phòng ngủ': 3.38, 'Số phòng tắm, vệ sinh': 3.05, 'Mặt tiền': 0.30, 'Gần bệnh viện': 0.15, 'Gần chợ': 0.37, 'Gần trường học': 0.28, 'Cao tầng': 0.64, 'Quy hoạch': 0.05},
    2: {'Diện tích': 631.34, 'Số phòng ngủ': 19.6, 'Số phòng tắm, vệ sinh': 10.8, 'Mặt tiền': 0.6, 'Gần bệnh viện': 0.11, 'Gần chợ': 0.17, 'Gần trường học': 0.11, 'Cao tầng': 0.91, 'Quy hoạch': 0.0}
}

quan_phuong_map = {
    "Bình Chánh": ["Thị trấn Tân Túc", "Xã An Phú Tây", "Xã Bình Chánh", "Xã Bình Hưng", "Xã Bình Lợi", "Xã Hưng Long", "Xã Lê Minh Xuân", "Xã Phong Phú", "Xã Phạm Văn Hai", "Xã Qui Đức", "Xã Tân Kiên", "Xã Tân Nhựt", "Xã Tân Quý Tây", "Xã Vĩnh Lộc A", "Xã Vĩnh Lộc B", "Xã Đa Phước"],
    "Bình Thạnh": ["Phường 1", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 17", "Phường 19", "Phường 2", "Phường 21", "Phường 22", "Phường 24", "Phường 25", "Phường 26", "Phường 27", "Phường 28", "Phường 3", "Phường 5", "Phường 6", "Phường 7"],
    "Bình Tân": ["Phường An Lạc", "Phường An Lạc A", "Phường Bình Hưng Hòa", "Phường Bình Hưng Hòa A", "Phường Bình Hưng Hòa B", "Phường Bình Trị Đông", "Phường Bình Trị Đông A", "Phường Bình Trị Đông B", "Phường Tân Tạo", "Phường Tân Tạo A"],
    "Cần Giờ": ["Thị trấn Cần Thạnh", "Xã Bình Khánh", "Xã Long Hòa", "Xã Lý Nhơn"],
    "Củ Chi": ["Thị trấn Củ Chi", "Xã An Nhơn Tây", "Xã Bình Mỹ", "Xã Hòa Phú", "Xã Nhuận Đức", "Xã Phú Hòa Đông", "Xã Phú Mỹ Hưng", "Xã Phước Hiệp", "Xã Phước Thạnh", "Xã Phước Vĩnh An", "Xã Thái Mỹ", "Xã Trung An", "Xã Trung Lập Hạ", "Xã Trung Lập Thượng", "Xã Tân An Hội", "Xã Tân Thông Hội", "Xã Tân Thạnh Tây", "Xã Tân Thạnh Đông"],
    "Gò Vấp": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 17", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9"],
    "Hóc Môn": ["Thị trấn Hóc Môn", "Xã Bà Điểm", "Xã Thới Tam Thôn", "Xã Trung Chánh", "Xã Tân Hiệp", "Xã Tân Thới Nhì", "Xã Tân Xuân", "Xã Xuân Thới Sơn", "Xã Xuân Thới Thượng", "Xã Xuân Thới Đông", "Xã Đông Thạnh"],
    "Nhà Bè": ["Thị trấn Nhà Bè", "Xã Hiệp Phước", "Xã Long Thới", "Xã Nhơn Đức", "Xã Phú Xuân", "Xã Phước Kiển", "Xã Phước Lộc"],
    "Phú Nhuận": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 17", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 7", "Phường 8", "Phường 9"],
    "Quận 1": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường An Phú Đông", "Phường Bến Nghé", "Phường Bến Thành", "Phường Cô Giang", "Phường Cầu Kho", "Phường Cầu Ông Lãnh", "Phường Hiệp Thành", "Phường Nguyễn Cư Trinh", "Phường Nguyễn Thái Bình", "Phường Phạm Ngũ Lão", "Phường Thạnh Lộc", "Phường Thạnh Xuân", "Phường Thới An", "Phường Trung Mỹ Tây", "Phường Tân Chánh Hiệp", "Phường Tân Hưng Thuận", "Phường Tân Thới Hiệp", "Phường Tân Thới Nhất", "Phường Tân Định", "Phường Đa Kao", "Phường Đông Hưng Thuận", "phường Cô Giang"],
    "Quận 3": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 9", "Phường Võ Thị Sáu"],
    "Quận 4": ["Phường 1", "Phường 10", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 18", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 8", "Phường 9", "Phường Khánh Hội"],
    "Quận 5": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường Chợ lớn tphcm"],
    "Quận 6": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9"],
    "Quận 7": ["Phường Bình Thuận", "Phường Phú Mỹ", "Phường Phú Thuận", "Phường Tân Hưng", "Phường Tân Kiểng", "Phường Tân Phong", "Phường Tân Phú", "Phường Tân Quy", "Phường Tân Thuận Tây", "Phường Tân Thuận Đông"],
    "Quận 8": ["Phường 1", "Phường 10", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường Rạch Ông"],
    "Thủ Đức": ["Phường  Thạnh Mỹ Lợi", "Phường An Khánh", "Phường An Lợi Đông", "Phường An Phú", "Phường Bình An", "Phường Bình Chiểu", "Phường Bình Khánh", "Phường Bình Thọ", "Phường Bình Trưng Tây", "Phường Bình Trưng Đông", "Phường Cát Lái", "Phường Hiệp Bình Chánh", "Phường Hiệp Bình Phước", "Phường Hiệp Phú", "Phường Linh Chiểu", "Phường Linh Trung", "Phường Linh Tây", "Phường Linh Xuân", "Phường Linh Đông", "Phường Long Bình", "Phường Long Phước", "Phường Long Thạnh Mỹ", "Phường Long Trường", "Phường Phú Hữu", "Phường Phước Bình", "Phường Phước Long A", "Phường Phước Long B", "Phường Tam Bình", "Phường Tam Phú", "Phường Thạnh Mỹ Lợi", "Phường Thảo Điền", "Phường Thủ Thiêm", "Phường Trường Thạnh", "Phường Trường Thọ", "Phường Tân Phú", "Phường Tăng Nhơn Phú A", "Phường Tăng Nhơn Phú B"],
    "Tân Bình": ["Phường 1", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9"],
    "Tân Phú": ["Phường Hiệp Tân", "Phường Hòa Thạnh", "Phường Phú Thạnh", "Phường Phú Thọ Hòa", "Phường Phú Trung", "Phường Sơn Kỳ", "Phường Tân Phú", "Phường Tân Quý", "Phường Tân Sơn Nhì", "Phường Tân Thành", "Phường Tân Thới Hòa", "Phường Tây Thạnh", "Xã Tân Phú Trung"]
}

phap_ly_list = ['Sổ riêng', 'Sổ chung', 'Hợp đồng mua bán', 'Đang chờ sổ', 'Vi bằng / uỷ quyền', 'Không rõ']
noi_that_list = ['Nội thất cơ bản', 'Full nội thất', 'Nội thất cao cấp', 'Không nội thất', 'Không rõ']

def find_nearest_cluster(numeric_features):
    """Tính khoảng cách Euclidean đến các tâm cụm và trả về cụm gần nhất"""
    min_dist = float('inf')
    assigned_cluster = 1
    for cluster_id, center in centroids.items():
        dist = 0
        for col in center.keys():
            dist += (numeric_features[col] - center[col]) ** 2
        if dist < min_dist:
            min_dist = dist
            assigned_cluster = cluster_id
    return str(assigned_cluster)

def cap_nhat_phuong(quan_duoc_chon):
    danh_sach_phuong = quan_phuong_map.get(quan_duoc_chon, [])
    return gr.update(choices=danh_sach_phuong, value=danh_sach_phuong[0] if danh_sach_phuong else None)

def feature_engineering(data):
    df_fe = data.copy()
    
    # Phân khúc diện tích
    bins = [0, 30, 120, np.inf]
    labels = ['Duoi_30', 'Tu_30_den_120', 'Tren_120']
    df_fe['Phan_khuc_dien_tich'] = pd.cut(df_fe['Diện tích'], bins=bins, labels=labels)
    
    # Tỷ lệ WC/Phòng ngủ
    df_fe['Ty_le_Bath_Bed'] = df_fe['Số phòng tắm, vệ sinh'] / (df_fe['Số phòng ngủ'] + 1e-5)
    
    # Loại hình kết hợp Quận
    df_fe['Loai_hinh_Quan'] = df_fe['Cao tầng'].astype(str) + '_' + df_fe['Quận'].astype(str)
    
    # Tổng tiện ích
    util_cols = ['Gần bệnh viện', 'Gần chợ', 'Gần trường học']
    df_fe['Tong_tien_ich'] = df_fe[util_cols].sum(axis=1).astype(str)
    df_fe = df_fe.drop(columns=util_cols)
    
    return df_fe

def predict_price(model_name, dien_tich_str, so_phong_ngu, so_phong_tam, phap_ly, noi_that, 
                  mat_tien, gan_bv, gan_cho, gan_th, cao_tang, quy_hoach, phuong, quan):
    
    try:
        dt_val = str(dien_tich_str).replace(',', '.')
        dien_tich = float(dt_val)
        if dien_tich <= 0: return "Lỗi", "Diện tích phải > 0"
    except:
        return "Lỗi định dạng", "Vui lòng nhập diện tích hợp lệ (VD: 26,9)"

    numeric_features = {
        'Diện tích': dien_tich,
        'Số phòng ngủ': so_phong_ngu,
        'Số phòng tắm, vệ sinh': so_phong_tam,
        'Mặt tiền': 1 if mat_tien else 0,
        'Gần bệnh viện': 1 if gan_bv else 0,
        'Gần chợ': 1 if gan_cho else 0,
        'Gần trường học': 1 if gan_th else 0,
        'Cao tầng': 1 if cao_tang else 0,
        'Quy hoạch': 1 if quy_hoach else 0
    }
    
    assigned_cluster = find_nearest_cluster(numeric_features)

    input_df = pd.DataFrame([{
        **numeric_features,
        'Pháp lý': phap_ly,
        'Nội thất': noi_that,
        'Phường': phuong,
        'Quận': quan,
        'Cluster': assigned_cluster
    }])
    
    model_path = f"models/tuned_model/best_{'xgb' if model_name == 'XGBoost' else 'rf'}_model.pkl"
    if not os.path.exists(model_path):
        model_path = f"best_{'xgb' if model_name == 'XGBoost' else 'rf'}_model.pkl"
        if not os.path.exists(model_path):
            return "Lỗi", f"Không tìm thấy file {model_path}"

    try:
        model = joblib.load(model_path)
        data_fe = feature_engineering(input_df)
        
        pred_log = model.predict(data_fe)[0]
        pred_val = np.expm1(pred_log) # Tỷ VNĐ/m2
        
        return f"{pred_val * 1000:,.1f} Triệu VNĐ/m²", f"{pred_val * dien_tich:,.3f} Tỷ VNĐ (Cụm: {assigned_cluster})"
    except Exception as e:
        return "Lỗi hệ thống", str(e)

with gr.Blocks(title="Dự đoán Giá Nhà TP.HCM") as demo:
    gr.Markdown("# Hệ Thống Dự Đoán Giá Nhà (TP.HCM)")
    gr.Markdown("Xác định giá nhà dựa trên vị trí, diện tích và các tiện ích đi kèm.")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Thông tin nhà")
            with gr.Row():
                quan = gr.Dropdown(choices=list(quan_phuong_map.keys()), label="Quận", value="Quận 1")
                phuong = gr.Dropdown(choices=quan_phuong_map["Quận 1"], label="Phường", value="Phường Bến Nghé")
            
            dien_tich = gr.Textbox(label="Diện tích (m²)", value="50,0", placeholder="VD: 26,9")
            
            with gr.Row():
                so_phong_ngu = gr.Number(label="Số phòng ngủ", value=2, minimum=0)
                so_phong_tam = gr.Number(label="Số phòng tắm/WC", value=2, minimum=0)
                
            with gr.Row():
                phap_ly = gr.Dropdown(choices=phap_ly_list, label="Pháp lý", value="Sổ riêng")
                noi_that = gr.Dropdown(choices=noi_that_list, label="Nội thất", value="Nội thất cơ bản")
            
        with gr.Column(scale=1):
            gr.Markdown("### Tiện ích & Đặc điểm")
            mat_tien = gr.Checkbox(label="Mặt tiền (Mặt phố)")
            gan_bv = gr.Checkbox(label="Gần bệnh viện")
            gan_cho = gr.Checkbox(label="Gần chợ")
            gan_th = gr.Checkbox(label="Gần trường học")
            cao_tang = gr.Checkbox(label="Nhà cao tầng")
            quy_hoach = gr.Checkbox(label="Nằm trong quy hoạch")
            
    quan.change(fn=cap_nhat_phuong, inputs=quan, outputs=phuong)

    gr.Markdown("---")
    model_name = gr.Radio(choices=["XGBoost", "Random Forest"], label="Chọn Mô Hình Dự Đoán", value="XGBoost")
    btn = gr.Button("DỰ ĐOÁN GIÁ", variant="primary", size="lg")
    
    with gr.Row():
        out_gia_m2 = gr.Textbox(label="Giá dự kiến trên 1m²", text_align="center")
        out_tong_gia = gr.Textbox(label="Tổng giá trị dự kiến", text_align="center")

    btn.click(
        fn=predict_price,
        inputs=[model_name, dien_tich, so_phong_ngu, so_phong_tam, phap_ly, noi_that, 
                mat_tien, gan_bv, gan_cho, gan_th, cao_tang, quy_hoach, phuong, quan],
        outputs=[out_gia_m2, out_tong_gia]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)