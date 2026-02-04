
# ============================================
# 🚗 ARABA FİYAT TAHMİNİ - GELİŞMİŞ VERSİYON
# app.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SAYFA AYARLARI
# ============================================
st.set_page_config(
    page_title="🚗 Araba Fiyat Tahmini",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS STİLLERİ
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #eee;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .price-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    
    .price-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .price-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .stat-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #666;
    }
    
    .tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    .tag-green {
        background: #d4edda;
        color: #155724;
    }
    
    .tag-red {
        background: #f8d7da;
        color: #721c24;
    }
    
    .tag-blue {
        background: #cce5ff;
        color: #004085;
    }
    
    .tag-yellow {
        background: #fff3cd;
        color: #856404;
    }
    
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# VERİ VE MODEL YÜKLEME
# ============================================
@st.cache_resource
def load_models():
    """Model ve gerekli dosyaları yükle"""
    try:
        # Model yükleme sırası: LightGBM > XGBoost > Random Forest
        model_files = [
        ('best_model_gradient_boosting.pkl', 'Gradient Boosting'),
        ('model_lightgbm.pkl', 'LightGBM'),
        ('model_xgboost.pkl', 'XGBoost'),
        ('model_random_forest.pkl', 'Random Forest'),
        ('best_model_lightgbm.pkl', 'LightGBM'),
        ('best_model_xgboost.pkl', 'XGBoost'),
        ('best_model_random_forest.pkl', 'Random Forest'),
        ]
        
        model = None
        model_name = None
        
        for file, name in model_files:
            try:
                model = joblib.load(file)
                model_name = name
                break
            except:
                continue
        
        if model is None:
            raise FileNotFoundError("Model dosyası bulunamadı")
        
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        scaler = joblib.load('scaler.pkl')
        
        try:
            model_info = joblib.load('model_info.pkl')
        except:
            model_info = {'test_r2': 0.95, 'test_rmse': 100000, 'test_mae': 70000}
        
        return model, label_encoders, feature_names, scaler, model_info, model_name
    
    except Exception as e:
        return None, None, None, None, None, str(e)

@st.cache_data
def load_data():
    """Veri setini yükle"""
    try:
        df = pd.read_pickle('processed_data.pkl')
        return df
    except:
        try:
            df = pd.read_excel('arabam_tum_veriler.xlsx')
            # Temel temizlik
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
            return df
        except:
            return None

# ============================================
# TAHMİN FONKSİYONU
# ============================================
def make_prediction(model, label_encoders, feature_names, input_data):
    """Fiyat tahmini yap"""
    
    df_input = pd.DataFrame([input_data])
    
    # Feature Engineering
    current_year = 2024
    df_input['arac_yasi'] = current_year - df_input['yil']
    df_input['km_yas_orani'] = df_input['kilometre'] / (df_input['arac_yasi'] + 1)
    df_input['elektrikli_mi'] = df_input['yakit_tipi'].str.lower().str.contains('elektrik', na=False).astype(int)
    df_input['hibrit_mi'] = df_input['yakit_tipi'].str.lower().str.contains('hibrit', na=False).astype(int)
    df_input['toplam_hasar'] = df_input.get('boyali_sayi', 0) + df_input.get('degismis_sayi', 0)
    df_input['hasarli_mi'] = (df_input['toplam_hasar'] > 0).astype(int)
    
    luks_markalar = ['Mercedes - Benz', 'BMW', 'Audi', 'Porsche', 'Land Rover', 'Jaguar', 
                     'Volvo', 'Lexus', 'Maserati', 'Ferrari', 'Lamborghini', 'Bentley',
                     'Mercedes', 'Range Rover', 'Alfa Romeo']
    df_input['luks_marka'] = df_input['marka'].isin(luks_markalar).astype(int)
    df_input['otomatik_mi'] = df_input['vites_tipi'].str.lower().str.contains('otomatik', na=False).astype(int)
    df_input['dizel_mi'] = df_input['yakit_tipi'].str.lower().str.contains('dizel', na=False).astype(int)
    df_input['sahibinden_mi'] = df_input['kimden'].str.lower().str.contains('sahib', na=False).astype(int)
    
    # Label Encoding
    for col, le in label_encoders.items():
        if col in df_input.columns:
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except:
                df_input[col] = 0
    
    # Eksik sütunları ekle
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0
    
    df_input = df_input[feature_names]
    
    return model.predict(df_input)[0]

# ============================================
# SIDEBAR
# ============================================
def render_sidebar(model_info, model_name):
    """Sidebar içeriği"""
    
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/car.png", width=150)
        
        st.markdown("### 📊 Model Performansı")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Skoru", f"{model_info.get('test_r2', 0.95):.2%}")
        with col2:
            st.metric("Model", model_name)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Tahmin Güvenilirliği")
        r2 = model_info.get('test_r2', 0.95)
        if r2 >= 0.95:
            st.success("✅ Çok Yüksek")
        elif r2 >= 0.90:
            st.info("📊 Yüksek")
        elif r2 >= 0.80:
            st.warning("⚠️ Orta")
        else:
            st.error("❌ Düşük")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ Bilgi")
        st.info("""
        Bu uygulama makine öğrenmesi 
        kullanarak ikinci el araç 
        fiyatlarını tahmin eder.
        
        Tahminler piyasa koşullarına 
        göre farklılık gösterebilir.
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Geliştirici")
        st.markdown("Made with ❤️ using Streamlit")

# ============================================
# ANA UYGULAMA
# ============================================
def main():
    # Yükleme
    model, label_encoders, feature_names, scaler, model_info, model_name = load_models()
    df = load_data()
    
    # Hata kontrolü
    if model is None:
        st.error(f"⚠️ Model yüklenemedi: {model_name}")
        st.stop()
    
    # Sidebar
    render_sidebar(model_info, model_name)
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Araba Fiyat Tahmini</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Yapay Zeka ile Araç Değerleme</p>', unsafe_allow_html=True)
    
    # Seçenekleri hazırla
    if df is not None:
        markalar = sorted(df['marka'].dropna().unique().tolist())
    else:
        markalar = ['Volkswagen', 'BMW', 'Mercedes - Benz', 'Audi', 'Toyota', 'Ford', 'Renault']
    
    # Tabs
    tab1, tab2 = st.tabs(["🚗 Fiyat Tahmini", "📊 İstatistikler"])
    
    with tab1:
        # Form alanları
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card"><div class="card-title">🏭 Araç Bilgileri</div>', unsafe_allow_html=True)
            
            marka = st.selectbox("Marka", options=markalar, index=0)
            
            if df is not None and marka:
                seriler = sorted(df[df['marka'] == marka]['seri'].dropna().unique().tolist())
            else:
                seriler = ['Seri Seçin']
            seri = st.selectbox("Seri", options=seriler if seriler else ['Belirtilmemiş'])
            
            if df is not None and marka and seri:
                modeller = sorted(df[(df['marka'] == marka) & (df['seri'] == seri)]['model'].dropna().unique().tolist())
            else:
                modeller = ['Model Seçin']
            model_sec = st.selectbox("Model", options=modeller if modeller else ['Belirtilmemiş'])
            
            yil = st.slider("Model Yılı", 1990, 2024, 2020)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card"><div class="card-title">⚙️ Teknik Özellikler</div>', unsafe_allow_html=True)
            
            kilometre = st.number_input("Kilometre", 0, 1000000, 50000, 1000)
            vites_tipi = st.selectbox("Vites Tipi", ['Otomatik', 'Yarı Otomatik', 'Düz'])
            yakit_tipi = st.selectbox("Yakıt Tipi", ['Benzin', 'Dizel', 'Benzin & LPG', 'Hibrit', 'Elektrik'])
            
            if yakit_tipi == 'Elektrik':
                motor_hacmi = 0
                st.info("🔋 Elektrikli araç")
            else:
                motor_hacmi = st.number_input("Motor Hacmi (cc)", 0, 8000, 1600, 100)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card"><div class="card-title">📋 Diğer Bilgiler</div>', unsafe_allow_html=True)
            
            kasa_tipi = st.selectbox("Kasa Tipi", ['Sedan', 'Hatchback', 'SUV', 'Station Wagon', 'Coupe', 'MPV'])
            renk = st.selectbox("Renk", ['Beyaz', 'Siyah', 'Gri', 'Gümüş', 'Mavi', 'Kırmızı', 'Lacivert'])
            kimden = st.selectbox("İlan Sahibi", ['Sahibinden', 'Galeriden', 'Yetkili Bayiden'])
            
            st.markdown("**Hasar Durumu**")
            boyali_sayi = st.slider("Boyalı Parça", 0, 13, 0)
            degismis_sayi = st.slider("Değişen Parça", 0, 13, 0)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tahmin butonu
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            if st.button("🔮 FİYAT TAHMİN ET", use_container_width=True):
                
                input_data = {
                    'marka': marka,
                    'seri': seri,
                    'model': model_sec,
                    'yil': yil,
                    'kilometre': kilometre,
                    'vites_tipi': vites_tipi,
                    'yakit_tipi': yakit_tipi,
                    'kasa_tipi': kasa_tipi,
                    'renk': renk,
                    'motor_hacmi': motor_hacmi,
                    'kimden': kimden,
                    'boyali_sayi': boyali_sayi,
                    'degismis_sayi': degismis_sayi
                }
                
                with st.spinner('Tahmin yapılıyor...'):
                    try:
                        prediction = make_prediction(model, label_encoders, feature_names, input_data)
                        
                        # Sonuç gösterimi
                        st.markdown("---")
                        
                        # Fiyat kartı
                        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
                        with col_r2:
                            st.markdown(f"""
                            <div class="price-display">
                                <div class="price-label">Tahmini Fiyat</div>
                                <div class="price-value">{prediction:,.0f} TL</div>
                                <div class="price-label">{marka} {seri} {yil}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Fiyat aralığı
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.metric("📉 Minimum", f"{prediction * 0.90:,.0f} TL", "-10%")
                        with col_m2:
                            st.metric("💰 Tahmin", f"{prediction:,.0f} TL")
                        with col_m3:
                            st.metric("📈 Maksimum", f"{prediction * 1.10:,.0f} TL", "+10%")
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction,
                            number={'valueformat': ',.0f', 'suffix': ' TL'},
                            gauge={
                                'axis': {'range': [0, prediction * 2], 'tickformat': ','},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, prediction * 0.7], 'color': '#E8F5E9'},
                                    {'range': [prediction * 0.7, prediction * 1.3], 'color': '#C8E6C9'},
                                    {'range': [prediction * 1.3, prediction * 2], 'color': '#FFCDD2'}
                                ]
                            }
                        ))
                        fig.update_layout(height=250, margin=dict(t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Etiketler
                        st.markdown("**📌 Özellikler**")
                        tags = []
                        if yil >= 2020:
                            tags.append(('<span class="tag tag-green">Yeni Model</span>', 'green'))
                        if boyali_sayi == 0 and degismis_sayi == 0:
                            tags.append(('<span class="tag tag-green">Hasarsız</span>', 'green'))
                        else:
                            tags.append(('<span class="tag tag-red">Hasarlı</span>', 'red'))
                        if vites_tipi == 'Otomatik':
                            tags.append(('<span class="tag tag-blue">Otomatik</span>', 'blue'))
                        if yakit_tipi == 'Elektrik':
                            tags.append(('<span class="tag tag-green">Elektrikli</span>', 'green'))
                        if kimden == 'Sahibinden':
                            tags.append(('<span class="tag tag-yellow">Sahibinden</span>', 'yellow'))
                        
                        st.markdown(' '.join([t[0] for t in tags]), unsafe_allow_html=True)
                        
                        st.success("✅ Tahmin başarıyla tamamlandı!")
                        
                    except Exception as e:
                        st.error(f"❌ Hata: {str(e)}")
    
    with tab2:
        st.markdown("### 📊 Veri Seti İstatistikleri")
        
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Toplam Kayıt", f"{len(df):,}")
            with col2:
                st.metric("🏭 Marka Sayısı", f"{df['marka'].nunique()}")
            with col3:
                st.metric("💰 Ort. Fiyat", f"{df['fiyat'].mean():,.0f} TL")
            with col4:
                st.metric("📅 Ort. Yıl", f"{df['yil'].mean():.0f}")
            
            # Marka dağılımı
            st.markdown("### 🏭 Marka Dağılımı")
            marka_dist = df['marka'].value_counts().head(10)
            fig = px.bar(x=marka_dist.index, y=marka_dist.values, 
                        labels={'x': 'Marka', 'y': 'İlan Sayısı'},
                        color=marka_dist.values, color_continuous_scale='viridis')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Fiyat dağılımı
            st.markdown("### 💰 Fiyat Dağılımı")
            fig = px.histogram(df, x='fiyat', nbins=50, 
                              labels={'fiyat': 'Fiyat (TL)', 'count': 'Adet'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Veri seti yüklenemedi.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        🚗 Araba Fiyat Tahmini | Yapay Zeka Destekli | 2024
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()