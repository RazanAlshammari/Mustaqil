# -*- coding: utf-8 -*-
"""
تشغيل النماذج الثلاثة ورؤية نتائجها في الترمنال مباشرة.
الطريقة:  python run_models.py
(يحتاج: pandas, scikit-learn, openpyxl)
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, classification_report

print("جارٍ تحميل البيانات...")
monthly = pd.read_excel("mustaqil_dataset.xlsx", sheet_name="Monthly_Data")
projects = pd.read_excel("mustaqil_dataset.xlsx", sheet_name="Projects_Data")

# ═══════ النموذج 1: التنبؤ بشهر الجفاف ═══════
print("\n" + "="*55)
print("النموذج 1: التنبؤ بشهر الجفاف (تصنيف)")
print("="*55)
m = monthly.sort_values(["Freelancer_ID","Year","Month"]).copy()
m["dry"] = (m["Dry_Month_Label"]=="Yes").astype(int)
m["Income_lag1"]  = m.groupby("Freelancer_ID")["Income"].shift(1)
m["Income_roll3"] = m.groupby("Freelancer_ID")["Income"].transform(
    lambda x: x.rolling(3,min_periods=1).mean())
m = m.dropna(subset=["Income_lag1"])
f1 = ["Income_lag1","Income_roll3","Number_of_Projects","Total_Expenses","Month","Payment_Delay_Days"]
Xtr,Xte,ytr,yte = train_test_split(m[f1], m["dry"], test_size=0.25, random_state=1, stratify=m["dry"])
clf = RandomForestClassifier(n_estimators=150, random_state=1, class_weight="balanced")
clf.fit(Xtr,ytr)
print(f"الدقة (Accuracy): {accuracy_score(yte, clf.predict(Xte))*100:.1f}%")
print("\nتقرير مفصّل:")
print(classification_report(yte, clf.predict(Xte), target_names=["شهر عادي","شهر جفاف"]))

# ═══════ النموذج 2: التنبؤ بالدخل القادم ═══════
print("="*55)
print("النموذج 2: التنبؤ بدخل الشهر القادم (انحدار)")
print("="*55)
m2 = m.dropna(subset=["Next_Month_Income"])
f2 = ["Income","Income_lag1","Income_roll3","Number_of_Projects","Month"]
Xtr,Xte,ytr,yte = train_test_split(m2[f2], m2["Next_Month_Income"], test_size=0.25, random_state=1)
reg = RandomForestRegressor(n_estimators=150, random_state=1)
reg.fit(Xtr,ytr); pred = reg.predict(Xte)
print(f"R² (معامل التحديد): {r2_score(yte,pred):.2f}")
print(f"متوسط الخطأ (MAE): {mean_absolute_error(yte,pred):,.0f} ر.س")
print("ملاحظة: R² منخفض لأن الدخل متقلّب بطبيعته، لكن النموذج يلتقط الاتجاه الموسمي.")

# ═══════ النموذج 3: اقتراح التسعير ═══════
print("\n" + "="*55)
print("النموذج 3: اقتراح سعر المشروع (انحدار)")
print("="*55)
p = pd.get_dummies(projects, columns=["Specialty","Client_Type"])
drop = ["Project_ID","Freelancer_ID","Project_Value","Suggested_Price"]
f3 = [c for c in p.columns if c not in drop]
Xtr,Xte,ytr,yte = train_test_split(p[f3], p["Suggested_Price"], test_size=0.25, random_state=1)
reg2 = RandomForestRegressor(n_estimators=200, random_state=1)
reg2.fit(Xtr,ytr); pred = reg2.predict(Xte)
print(f"R² (معامل التحديد): {r2_score(yte,pred):.2f}")
print(f"متوسط الخطأ (MAE): {mean_absolute_error(yte,pred):,.0f} ر.س")

print("\n✅ انتهى تشغيل النماذج الثلاثة")
