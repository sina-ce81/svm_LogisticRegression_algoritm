import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# خواندن فایل اکسل
data = pd.read_excel('feshar.xlsx')

# حذف ستون‌های غیرضروری
data = data.drop(columns=['Unnamed: 0', 'drugs'], errors='ignore')

# تبدیل ستون‌های دسته‌ای به مقادیر عددی
categorical_columns = ['بارداری', 'سابقه خانوادگی']
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])
    data[col] = data[col].astype('category').cat.codes

# پر کردن مقادیر گمشده ستون‌های عددی با میانگین
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].apply(lambda x: x.fillna(x.mean()))

# استانداردسازی داده‌های عددی
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# انتخاب ستون 'a' به‌عنوان برچسب (Label) و تبدیل به مقادیر عددی
X = data.drop(columns=['a'])  # ویژگی‌ها
y = data['a']                 # برچسب

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# پیش‌بینی روی داده‌های آزمون
y_pred = svm_model.predict(X_test)

# ارزیابی مدل
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# نمایش نتایج
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
