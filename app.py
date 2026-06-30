# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════╗
║  مستقل | Mustaqil — الرفيق المالي الذكي للفريلانسر السعودي     ║
║  هاكاثون أمد — مصرف الإنماء × أكاديمية طويق                    ║
╚═══════════════════════════════════════════════════════════════╝

طريقة التشغيل:
    1) pip install streamlit pandas numpy scikit-learn plotly openpyxl
    2) ضع هذا الملف بجانب ملف البيانات mustaqil_dataset.xlsx
    3) في الترمنال:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ═══════════════════════════════════════════════════════════════
#  إعداد الصفحة والهوية البصرية
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="مستقل | Mustaqil", page_icon="💸", layout="wide")

# ألوان مصرف الإنماء (أخضر) + لمسة عصرية
PRIMARY   = "#00833E"   # أخضر الإنماء
ACCENT    = "#C9A227"   # ذهبي
DARK      = "#0E2A1F"
DRY_RED   = "#E63946"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
    }}
    .stApp {{ background: linear-gradient(160deg,#f6faf7 0%,#eef5f0 100%); }}

    .hero {{
        background: linear-gradient(135deg,{PRIMARY} 0%,{DARK} 100%);
        padding: 34px 40px; border-radius: 22px; color:#fff;
        box-shadow: 0 12px 40px rgba(0,131,62,.25); margin-bottom: 8px;
    }}
    .hero h1 {{ font-size: 44px; font-weight:800; margin:0; }}
    .hero p  {{ font-size: 18px; opacity:.92; margin:6px 0 0; }}
    .badge {{
        display:inline-block; background:{ACCENT}; color:{DARK};
        padding:4px 14px; border-radius:30px; font-weight:700; font-size:13px;
    }}
    .metric-card {{
        background:#fff; border-radius:18px; padding:22px;
        box-shadow:0 4px 18px rgba(0,0,0,.06); border-right:5px solid {PRIMARY};
        height:100%;
    }}
    .metric-card.gold   {{ border-right-color:{ACCENT}; }}
    .metric-card.red    {{ border-right-color:{DRY_RED}; }}
    .metric-val   {{ font-size:30px; font-weight:800; color:{DARK}; }}
    .metric-label {{ font-size:14px; color:#6b7c72; font-weight:500; }}

    .salary-box {{
        background:linear-gradient(135deg,{PRIMARY},#00a350);
        border-radius:22px; padding:30px; color:#fff; text-align:center;
        box-shadow:0 10px 30px rgba(0,131,62,.3);
    }}
    .salary-box .num {{ font-size:52px; font-weight:800; }}

    .alert-dry {{
        background:linear-gradient(135deg,{DRY_RED},#c1121f); color:#fff;
        padding:20px 26px; border-radius:18px; font-weight:600; font-size:17px;
        box-shadow:0 8px 24px rgba(230,57,70,.3);
    }}
    .alert-safe {{
        background:linear-gradient(135deg,#2a9d8f,#21867a); color:#fff;
        padding:20px 26px; border-radius:18px; font-weight:600; font-size:17px;
    }}
    section[data-testid="stSidebar"] {{ background:{DARK}; }}
    section[data-testid="stSidebar"] * {{ color:#eafff2 !important; }}
    /* القائمة المنسدلة لاختيار الحساب: خلفية بيضاء ونص أسود */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {{ color:#0E2A1F !important; }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{ background:#ffffff !important; }}
    /* قائمة الخيارات المنسدلة (الأسماء) نص أسود */
    div[data-baseweb="popover"] li {{ color:#0E2A1F !important; }}
    div[data-baseweb="popover"] * {{ color:#0E2A1F !important; }}
    .stButton>button {{
        background:{PRIMARY}; color:#fff; border-radius:12px; border:none;
        padding:10px 22px; font-weight:700; font-family:'Tajawal';
    }}
</style>
""", unsafe_allow_html=True)

MONTH_NAMES = {1:"يناير",2:"فبراير",3:"مارس",4:"أبريل",5:"مايو",6:"يونيو",
               7:"يوليو",8:"أغسطس",9:"سبتمبر",10:"أكتوبر",11:"نوفمبر",12:"ديسمبر"}

# ═══════════════════════════════════════════════════════════════
#  تحميل البيانات + تدريب الموديلات (مع التخزين المؤقت)
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    xls = "mustaqil_dataset.xlsx"
    monthly = pd.read_excel(xls, sheet_name="Monthly_Data")
    projects = pd.read_excel(xls, sheet_name="Projects_Data")
    return monthly, projects

@st.cache_resource
def train_models(monthly, projects):
    # --- موديل الجفاف ---
    m = monthly.sort_values(["Freelancer_ID","Year","Month"]).copy()
    m["dry"] = (m["Dry_Month_Label"]=="Yes").astype(int)
    m["Income_lag1"]  = m.groupby("Freelancer_ID")["Income"].shift(1)
    m["Income_roll3"] = m.groupby("Freelancer_ID")["Income"].transform(
        lambda x: x.rolling(3,min_periods=1).mean())
    m = m.dropna(subset=["Income_lag1"])
    f1 = ["Income_lag1","Income_roll3","Number_of_Projects","Total_Expenses","Month","Payment_Delay_Days"]
    clf = RandomForestClassifier(n_estimators=150,random_state=1,class_weight="balanced")
    clf.fit(m[f1], m["dry"])

    # --- موديل الدخل القادم (مع الشهر كميزة موسمية) ---
    m2 = m.dropna(subset=["Next_Month_Income"])
    f2 = ["Income","Income_lag1","Income_roll3","Number_of_Projects","Month"]
    reg = RandomForestRegressor(n_estimators=150,random_state=1)
    reg.fit(m2[f2], m2["Next_Month_Income"])

    # --- موديل التسعير ---
    p = pd.get_dummies(projects, columns=["Specialty","Client_Type"])
    drop = ["Project_ID","Freelancer_ID","Project_Value","Suggested_Price"]
    f3 = [c for c in p.columns if c not in drop]
    reg2 = RandomForestRegressor(n_estimators=200,random_state=1)
    reg2.fit(p[f3], p["Suggested_Price"])

    return clf, f1, reg, f2, reg2, f3

monthly, projects = load_data()
clf, F1, reg_income, F2, reg_price, F3 = train_models(monthly, projects)

# ═══════════════════════════════════════════════════════════════
#  الهيدر
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <span class="badge">هاكاثون أمد · مصرف الإنماء</span>
  <h1>مستقل 💸</h1>
  <p>الرفيق المالي الذكي للفريلانسر السعودي — راتبك أنت من يحدده</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  الشريط الجانبي — اختيار الفريلانسر
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 👤 حساب الفريلانسر")
    names = monthly[["Freelancer_ID","Name"]].drop_duplicates()
    options = {f"{r.Name} (#{r.Freelancer_ID})": r.Freelancer_ID for r in names.itertuples()}
    pick = st.selectbox("اختر الحساب", list(options.keys()))
    fid = options[pick]

    st.markdown("---")
    window = st.slider("نافذة حساب الراتب (أشهر)", 2, 6, 3)
    st.markdown("---")
    st.caption("مستقل — كل ريال له وظيفة،\nوكل فريلانسر له راتب")

user = monthly[monthly["Freelancer_ID"]==fid].sort_values(["Year","Month"]).reset_index(drop=True)
uname = user["Name"].iloc[0]
specialty = user["Specialty"].iloc[0]

# ═══════════════════════════════════════════════════════════════
#  التبويبات
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 اللوحة الرئيسية", "🌵 تحذير الجفاف", "📊 محفظة المشاريع", "💰 حاسبة التسعير"
])

# ───────────────────────── تبويب 1 : اللوحة ─────────────────────
with tab1:
    last = user.iloc[-1]
    avg_income = user["Income"].tail(window).mean()        # الراتب الاصطناعي
    total_income_12 = user["Income"].tail(12).sum()
    avg_12 = user["Income"].tail(12).mean()
    stability = 100 - min(100, user["Income"].tail(6).std()/max(1,avg_12)*100)

    st.markdown(f"#### مرحباً {uname} — تخصص: {specialty}")

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">آخر دخل شهري</div>'
                    f'<div class="metric-val">{last.Income:,.0f}</div><div class="metric-label">ريال</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card gold"><div class="metric-label">متوسط دخل 12 شهر</div>'
                    f'<div class="metric-val">{avg_12:,.0f}</div><div class="metric-label">ريال/شهر</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">صندوق الطوارئ</div>'
                    f'<div class="metric-val">{last.Emergency_Fund:,.0f}</div><div class="metric-label">ريال</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card gold"><div class="metric-label">مؤشر استقرار الدخل</div>'
                    f'<div class="metric-val">{stability:.0f}%</div><div class="metric-label">كلما زاد كان أفضل</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cc1, cc2 = st.columns([1,2])
    with cc1:
        st.markdown(f"""
        <div class="salary-box">
          <div style="font-size:16px;opacity:.9">راتبك الاصطناعي الثابت</div>
          <div class="num">{avg_income:,.0f}</div>
          <div style="font-size:15px;opacity:.9">ريال / شهر &nbsp;·&nbsp; محسوب من آخر {window} أشهر</div>
        </div>""", unsafe_allow_html=True)
        st.info("💡 هذا هو المبلغ الثابت الذي يصرفه لك «صندوق التسوية» كل بداية شهر — "
                "والفائض يوزّع تلقائياً: 50% طوارئ · 30% مشاريع · 20% ادخار.")

    with cc2:
        # رسم الدخل الفعلي مقابل الراتب الاصطناعي
        user["label"] = user["Month"].map(MONTH_NAMES) + " " + user["Year"].astype(str).str[-2:]
        user["smoothed"] = user["Income"].rolling(window, min_periods=1).mean()
        fig = go.Figure()
        fig.add_bar(x=user["label"], y=user["Income"], name="الدخل الفعلي",
                    marker_color="#bcd9c7")
        fig.add_scatter(x=user["label"], y=user["smoothed"], name="الراتب الاصطناعي",
                        mode="lines+markers", line=dict(color=PRIMARY,width=4))
        fig.update_layout(height=340, margin=dict(t=30,b=70,l=10,r=10),
                          legend=dict(orientation="h"), plot_bgcolor="white",
                          xaxis=dict(tickangle=-45),
                          font=dict(family="Tajawal"))
        st.plotly_chart(fig, use_container_width=True)

    # وثيقة الدخل الموثق
    st.markdown("---")
    st.markdown("#### 📄 وثيقة الدخل الموثق (بديل شهادة الراتب للبنك)")
    d1,d2,d3 = st.columns(3)
    d1.metric("متوسط الدخل (12 شهر)", f"{avg_12:,.0f} ر")
    d2.metric("إجمالي دخل السنة", f"{total_income_12:,.0f} ر")
    d3.metric("استقرار التدفق", f"{stability:.0f}%")
    st.caption("في النسخة الكاملة: زر واحد يولّد PDF رسمي بهذه البيانات — مقبول لدى البنوك ويعزّز ملف SIMAH.")

# ───────────────────────── تبويب 2 : الجفاف ─────────────────────
with tab2:
    st.markdown("#### 🌵 نظام التنبؤ بشهر الجفاف")
    st.caption("الموديل يحلل نمطك التاريخي ويتوقع إن كان الشهر القادم شهر جفاف (دخل منخفض) قبل وقوعه.")

    u = user.copy()
    u["Income_lag1"]  = u["Income"].shift(1)
    u["Income_roll3"] = u["Income"].rolling(3,min_periods=1).mean()
    nextmonth = (int(last.Month)%12)+1
    row = pd.DataFrame([{
        "Income_lag1": last.Income,
        "Income_roll3": u["Income"].tail(3).mean(),
        "Number_of_Projects": last.Number_of_Projects,
        "Total_Expenses": last.Total_Expenses,
        "Month": nextmonth,
        "Payment_Delay_Days": last.Payment_Delay_Days,
    }])
    proba = clf.predict_proba(row[F1])[0][1]
    pred_income = reg_income.predict(pd.DataFrame([{
        "Income": last.Income, "Income_lag1": u["Income"].iloc[-2] if len(u)>1 else last.Income,
        "Income_roll3": u["Income"].tail(3).mean(),
        "Number_of_Projects": last.Number_of_Projects, "Month": nextmonth }])[F2])[0]

    cL, cR = st.columns([1,1])
    with cL:
        if proba >= 0.5:
            st.markdown(f"""<div class="alert-dry">
            ⚠️ <b>تحذير: {MONTH_NAMES[nextmonth]} قد يكون شهر جفاف</b><br><br>
            احتمال انخفاض الدخل: <b>{proba*100:.0f}%</b><br>
            الدخل المتوقع: <b>{pred_income:,.0f} ريال</b><br><br>
            🔻 ننصح بخفض راتبك الاصطناعي 20% هذا الشهر، وتفعيل صندوق الطوارئ.<br>
            📞 الآن وقت مناسب لتجديد العقود أو البحث عن عميل جديد.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="alert-safe">
            ✅ <b>{MONTH_NAMES[nextmonth]} يبدو شهراً آمناً</b><br><br>
            احتمال الجفاف منخفض: <b>{proba*100:.0f}%</b><br>
            الدخل المتوقع: <b>{pred_income:,.0f} ريال</b><br><br>
            استمر على خطتك — وهذا وقت جيد لتعزيز صندوق الطوارئ.
            </div>""", unsafe_allow_html=True)
    with cR:
        # عداد احتمال الجفاف
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=proba*100,
            number={"suffix":"%","font":{"size":40}},
            title={"text":"احتمال شهر الجفاف"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":DRY_RED if proba>=0.5 else PRIMARY},
                   "steps":[{"range":[0,50],"color":"#d8f0e0"},
                            {"range":[50,100],"color":"#fbdcdf"}]}))
        gauge.update_layout(height=300, font=dict(family="Tajawal"))
        st.plotly_chart(gauge, use_container_width=True)

    # تمييز أشهر الجفاف التاريخية
    st.markdown("##### أشهر الجفاف في سجلك")
    u["label"] = u["Month"].map(MONTH_NAMES)+" "+u["Year"].astype(str).str[-2:]
    colors = [DRY_RED if x=="Yes" else "#bcd9c7" for x in u["Dry_Month_Label"]]
    figd = go.Figure(go.Bar(x=u["label"], y=u["Income"], marker_color=colors))
    figd.update_layout(height=340, margin=dict(t=50,b=80,l=10,r=10), plot_bgcolor="white",
                       font=dict(family="Tajawal"),
                       xaxis=dict(tickangle=-45),
                       title="🔴 أحمر = شهر جفاف   |   🟢 أخضر = شهر عادي")
    st.plotly_chart(figd, use_container_width=True)

# ───────────────────────── تبويب 3 : المشاريع ─────────────────────
with tab3:
    st.markdown("#### 📊 محفظة المشاريع المنفصلة")
    pj = projects[projects["Freelancer_ID"]==fid].copy()
    if len(pj)==0:
        st.warning("لا توجد مشاريع مسجلة لهذا الحساب.")
    else:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("عدد المشاريع", len(pj))
        m2.metric("إجمالي قيمة المشاريع", f"{pj['Project_Value'].sum():,.0f} ر")
        m3.metric("متوسط قيمة المشروع", f"{pj['Project_Value'].mean():,.0f} ر")
        m4.metric("أعلى مشروع", f"{pj['Project_Value'].max():,.0f} ر")

        cA, cB = st.columns(2)
        with cA:
            by_client = pj.groupby("Client_Type")["Project_Value"].sum().reset_index()
            figp = px.pie(by_client, names="Client_Type", values="Project_Value",
                          title="توزيع الدخل حسب نوع العميل",
                          color_discrete_sequence=px.colors.sequential.Greens_r)
            figp.update_layout(font=dict(family="Tajawal"))
            st.plotly_chart(figp, use_container_width=True)
        with cB:
            pj["ربحية/ساعة"] = pj["Project_Value"]/pj["Estimated_Hours"]
            figh = px.bar(pj.sort_values("ربحية/ساعة",ascending=False).head(10),
                          x="Project_ID", y="ربحية/ساعة",
                          title="أعلى 10 مشاريع ربحية بالساعة",
                          color="ربحية/ساعة", color_continuous_scale="Greens")
            figh.update_layout(font=dict(family="Tajawal"))
            st.plotly_chart(figh, use_container_width=True)

        st.markdown("##### تفاصيل المشاريع")
        show = pj[["Project_ID","Client_Type","Project_Duration_Days",
                   "Estimated_Hours","Project_Value"]].copy()
        show.columns = ["رقم المشروع","نوع العميل","المدة (يوم)","الساعات","القيمة (ريال)"]
        st.dataframe(show, use_container_width=True, hide_index=True)

# ───────────────────────── تبويب 4 : التسعير ─────────────────────
with tab4:
    st.markdown("#### 💰 حاسبة التسعير الذكية")
    st.caption("أدخل تفاصيل مشروع جديد، والموديل يقترح لك السعر العادل بناءً على بيانات السوق.")

    c1,c2,c3 = st.columns(3)
    with c1:
        in_spec = st.selectbox("التخصص", sorted(projects["Specialty"].unique()),
                               index=sorted(projects["Specialty"].unique()).index(specialty)
                               if specialty in projects["Specialty"].unique() else 0)
        in_client = st.selectbox("نوع العميل", sorted(projects["Client_Type"].unique()))
    with c2:
        in_duration = st.number_input("مدة المشروع (أيام)", 1, 120, 14)
        in_hours = st.number_input("ساعات العمل المقدّرة", 1.0, 400.0, 60.0)
    with c3:
        in_complexity = st.slider("تعقيد المشروع", 0.7, 1.5, 1.0, 0.05)
        in_rate = st.number_input("سعر ساعتك الحالي (ريال)", 50, 500,
                                   int(projects[projects["Specialty"]==in_spec]["Hourly_Rate"].iloc[0]))

    if st.button("🔮 احسب السعر المقترح"):
        # بناء صف بنفس أعمدة التدريب
        base = pd.DataFrame([{
            "Project_Duration_Days": in_duration, "Estimated_Hours": in_hours,
            "Complexity": in_complexity, "Hourly_Rate": in_rate }])
        for c in projects["Specialty"].unique():
            base[f"Specialty_{c}"] = 1 if c==in_spec else 0
        for c in projects["Client_Type"].unique():
            base[f"Client_Type_{c}"] = 1 if c==in_client else 0
        for col in F3:
            if col not in base: base[col]=0
        price = reg_price.predict(base[F3])[0]

        st.markdown(f"""
        <div class="salary-box" style="margin-top:14px">
          <div style="font-size:16px;opacity:.9">السعر المقترح لهذا المشروع</div>
          <div class="num">{price:,.0f}</div>
          <div style="font-size:15px;opacity:.9">ريال &nbsp;·&nbsp; نطاق عادل: {price*0.85:,.0f} – {price*1.15:,.0f}</div>
        </div>""", unsafe_allow_html=True)
        st.success(f"💡 بهذا السعر، ربحية ساعتك ≈ {price/in_hours:,.0f} ريال/ساعة")

st.markdown("---")
st.markdown(f"<center style='color:#6b7c72'>مستقل · Mustaqil — هاكاثون أمد · "
            f"مصرف الإنماء × أكاديمية طويق</center>", unsafe_allow_html=True)
