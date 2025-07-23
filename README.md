# 🚑 Drug User Profiling & Forecasting API

A FastAPI-based web service for analyzing, profiling, and forecasting drug user data using machine learning models such as **LSTM**, **RandomForest**, and **SHAP**.

---

## 📦 Features

- 🔮 Forecast future drug usage trends using LSTM
- 📊 Analyze feature relevance using SHAP & Variance
- 👤 Generate demographic summaries of drug users
- 🖼️ Serve plots as PNG or JSON (base64-encoded)
- 🧪 Swagger UI for testing API endpoints interactively

---

## ⚙️ Setup Instructions (Step-by-Step Tutorial)

### 🧾 1. Clone the Repository

```bash
git clone https://github.com/Alayers2804/Visualization-and-Prediction-Web-of-Drugs-User-using-LSTM-and-Feature-Importance.git
cd Visualization-and-Prediction-Web-of-Drugs-User-using-LSTM-and-Feature-Importance
```

---

### 🐍 2. Create and Activate Virtual Environment

#### On **Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On **Linux/macOS**:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 📦 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, you can manually install:

```bash
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib seaborn shap openpyxl tensorflow
```

---

### 🧪 4. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

You should see output like:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### 🌐 5. Explore API via Swagger

Once running, open your browser:

- **Swagger UI** 👉 http://127.0.0.1:8000/docs
- **ReDoc UI** 👉 http://127.0.0.1:8000/redoc

---

## 🔁 Example API Endpoints

| Method | Endpoint                     | Description                          |
|--------|------------------------------|--------------------------------------|
| GET    | `/predict`                   | Predict drug case numbers (LSTM)     |
| GET    | `/predict/plot`              | Plot forecast as PNG                 |
| GET    | `/predict/plot/json`         | Plot forecast as base64 JSON         |
| GET    | `/features`                  | Get feature importance (variance)    |
| GET    | `/features/plot`             | Visualize feature importance         |
| GET    | `/profile-summary`           | Show drug user demographics          |
| GET    | `/profile-summary/plot`      | Plot user distribution per feature   |

Use query parameters such as:

- `n_months=3` → Number of months to predict
- `features=["USIA", "PEKERJAAN"]` → Features to analyze
- `return_base64=true` → Get image as base64 string instead of PNG

---

## 📁 Suggested Project Structure

```
.
├── app/
│   ├── api.py                  # API routes (FastAPI)
│   ├── visualizer.py           # Plotting logic
│   ├── model_utils.py          # LSTM, SHAP, forecasting
│   ├── preprocess.py           # Feature engineering, cleaning
├── data/
│   └── dataset_tat_rehab.xlsx  # Your dataset file
├── static/                     # Auto-generated plots
├── requirements.txt
└── README.md
```

---

## 🛠️ Notes

- The dataset must be available at: `data/dataset_tat_rehab.xlsx`
- If you use TensorFlow and encounter errors, ensure you're using Python 3.8–3.10 for compatibility.
- For production, consider switching from `--reload` to a production server like Gunicorn.

---

## 🤝 Contribution

Feel free to fork the repo, open issues or pull requests to improve the project!

---
