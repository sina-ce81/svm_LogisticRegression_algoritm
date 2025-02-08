import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# خواندن فایل اکسل
data = pd.read_excel('feshar.xlsx')

# حذف ستون‌های غیرضروری
data = data.drop(columns=['Unnamed: 0', 'drugs'], errors='ignore')

# تبدیل ستون‌های دسته‌ای به مقادیر عددی
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    data[col] = data[col].fillna('نامشخص')  # پر کردن مقادیر گمشده
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# پر کردن مقادیر گمشده ستون‌های عددی با میانگین
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].apply(lambda x: x.fillna(x.mean()))

# استانداردسازی داده‌های عددی
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# انتخاب ستون 'a' به‌عنوان برچسب (Label)
X = data.drop(columns=['a'])  # ویژگی‌ها
y = data['a']                 # برچسب

# تبدیل برچسب‌های پیوسته به 3 دسته (Low, Medium, High)
try:
    y_binned = pd.qcut(y, q=3, labels=[0, 1, 2], duplicates='drop')
except ValueError:
    num_bins = len(pd.qcut(y, q=3, duplicates='drop').unique()) - 1
    y_binned = pd.qcut(y, q=num_bins, labels=list(range(num_bins)), duplicates='drop')

# بررسی تعداد نمونه‌های هر کلاس
print("Class distribution before filtering:")
print(y_binned.value_counts())

# حذف کلاس‌هایی که کمتر از 2 نمونه دارند
class_counts = y_binned.value_counts()
valid_classes = class_counts[class_counts >= 2].index
filtered_data = data[y_binned.isin(valid_classes)]

# به‌روزرسانی X و y پس از فیلتر کردن داده‌ها
X = filtered_data.drop(columns=['a'])
y = y_binned[y_binned.isin(valid_classes)]

# بررسی کلاس‌های موجود در داده‌های فیلترشده
print("Class distribution after filtering:")
print(y.value_counts())

# تقسیم داده‌ها به مجموعه‌های آموزش و آزمون (حفظ تعادل کلاس‌ها)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# آموزش مدل Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# پیش‌بینی روی داده‌های آزمون
y_pred = logistic_model.predict(X_test)

# ارزیابی مدل
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# نمایش نتایج
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
