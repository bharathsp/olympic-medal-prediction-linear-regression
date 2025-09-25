# 🏅 Olympic Medal Prediction using Linear Regression  

*A Machine Learning project to predict the number of medals a country might win in a given Olympic year.*  

---

## 📌 Project Overview  
This project leverages **Linear Regression** to predict medal counts based on historical Olympic data.  
The workflow involves **data preprocessing, outlier detection, feature engineering, model training, and evaluation**.  

The final trained model is saved as a **Pickle file** for deployment and future use.

---

## 📂 Datasets  

1. **athletes_events.csv**  
   Contains details of athletes across Olympic events.  
   - `ID, Name, Sex, Age, Height, Weight, Team, NOC, Games, Year, Season, City, Sport, Event, Medal`  

2. **NOC_regions.csv**  
   Contains mapping of **NOC (National Olympic Committee)** codes to regions.  
   - `NOC, region, notes`  

---

## 🛠️ Data Exploration & Preprocessing  

✔️ Handle Missing Values  
- Drop rows with missing values in athletes & NOC dataset  
- Impute missing values in `Age`, `Height`, `Weight` with mean  
- Fill missing values in `Medal` with `"None"`  

✔️ Merge Datasets  
- Merge **athletes dataset** with **NOC regions dataset**  
- Replace `NOC` with `region`  
- Combine `Team` + `NOC` → **Combined_Team**  
- Standardize text fields (`Combined_Team`, `City`, `Sport`, `Event`)  
- Remove duplicate rows  

✔️ Visualize & Detect Outliers  
- Box plots 📦 → `Age`, `Height`, `Weight`  
- Histograms 📊 → `Age`, `Height`, `Weight`  
- Z-score method to remove outliers  

---

## 🔧 Feature Engineering  

- **One-Hot Encode** → `Sex` column  
- **Group data** → by `Year` and `Combined_Team`  
- Create **Previous Medal Count** feature  
- Correlation Matrix 🔗 → drop low-correlation features  

---

## 🤖 Model Building  

- Features (`X`) and Target (`y`: medal count)  
- Split → **80% train / 20% test**  
- Apply **Feature Scaling**  
- Train **Linear Regression Model**  

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
````

---

## 📈 Model Evaluation

* Predictions added to test set
* Metrics:

```python
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R² Score:", r2)
```

---

## 🚀 Model Deployment

✔️ Save Model & Scaler

```python
import pickle

with open("olympic_medal_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

✔️ Load & Use Model

```python
with open("olympic_medal_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)

# Example Prediction
scaled_input = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(scaled_input)
```

---

## 📊 Tech Stack

* 🐍 Python
* 📚 Pandas, Numpy
* 📊 Matplotlib, Seaborn
* 🤖 Scikit-learn
* 🗄️ Pickle

---

## ✅ Results

* Successfully predicted **medal counts** per country
* Achieved meaningful R² score with Linear Regression
* Built an **end-to-end ML pipeline** ready for deployment

---

## 📌 Future Enhancements

* Try **Ridge / Lasso Regression** for better generalization
* Use **XGBoost / Random Forest** for improved accuracy 🌲
* Deploy via **Streamlit or Flask API** for real-time predictions
