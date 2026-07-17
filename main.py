from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Energy Prediction Dashboard")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the pipeline
pipeline = joblib.load('pipeline.joblib')
scaler = pipeline['scaler']
pca = pipeline['pca']
model = pipeline['model']
print("✅ Pipeline loaded successfully!")

# Load feature names
X_train = pd.read_csv('X_train.csv')
FEATURE_NAMES = list(X_train.columns)
print(f"✅ Loaded {len(FEATURE_NAMES)} features")

# ============================================
# BEAUTIFUL HOME PAGE
# ============================================
HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 30px;
            padding: 50px 60px;
            max-width: 900px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: slideUp 0.8s ease-out;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .logo {
            background: linear-gradient(135deg, #667eea, #764ba2);
            width: 80px;
            height: 80px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 18px;
            font-weight: 400;
            margin-bottom: 30px;
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 16px;
        }
        .feature-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .feature-item span { font-size: 20px; }
        .feature-item p { font-size: 13px; color: #333; font-weight: 500; }
        .nav-buttons {
            display: flex;
            gap: 20px;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        .btn-primary, .btn-secondary {
            padding: 16px 40px;
            border: none;
            border-radius: 14px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
        }
        .btn-secondary {
            background: #f0f2f5;
            color: #333;
        }
        .btn-secondary:hover {
            background: #e4e6e9;
            transform: translateY(-3px);
        }
        .stats {
            display: flex;
            gap: 30px;
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid #f0f2f5;
        }
        .stat-item { text-align: center; }
        .stat-item .number { font-size: 28px; font-weight: 700; color: #667eea; }
        .stat-item .label { font-size: 14px; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">⚡</div>
        <h1>Energy Consumption Prediction</h1>
        <p class="subtitle">Real-time prediction using PCA &amp; Machine Learning</p>
        
        <div class="features-grid">
            <div class="feature-item"><span>🌡️</span><p>Temperature</p></div>
            <div class="feature-item"><span>💧</span><p>Humidity</p></div>
            <div class="feature-item"><span>🕐</span><p>Hour of Day</p></div>
            <div class="feature-item"><span>📅</span><p>Day of Week</p></div>
            <div class="feature-item"><span>🏭</span><p>Load Type</p></div>
            <div class="feature-item"><span>📆</span><p>Weekend/Holiday</p></div>
        </div>
        
        <div class="nav-buttons">
            <a href="/dashboard" class="btn-primary">📊 View Dashboard</a>
            <a href="/predict" class="btn-secondary">🔮 Make Prediction</a>
        </div>
        
        <div class="stats">
            <div class="stat-item"><div class="number">10</div><div class="label">Features</div></div>
            <div class="stat-item"><div class="number">95%</div><div class="label">Variance Captured</div></div>
            <div class="stat-item"><div class="number">Real-time</div><div class="label">Predictions</div></div>
        </div>
    </div>
</body>
</html>
"""

# ============================================
# BEAUTIFUL DASHBOARD PAGE
# ============================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Energy Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: #f0f2f5;
            min-height: 100vh;
        }
        .navbar {
            background: white;
            padding: 18px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 20px rgba(0,0,0,0.08);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar .brand {
            font-size: 22px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .navbar .nav-links a {
            text-decoration: none;
            color: #555;
            margin: 0 20px;
            font-weight: 500;
            transition: color 0.3s;
            padding: 8px 16px;
            border-radius: 8px;
        }
        .navbar .nav-links a:hover {
            color: #667eea;
            background: #f0f2ff;
        }
        .navbar .nav-links a.active {
            color: #667eea;
            background: #f0f2ff;
        }
        .container {
            max-width: 1300px;
            margin: 40px auto;
            padding: 0 30px;
        }
        .page-header {
            margin-bottom: 30px;
        }
        .page-header h1 {
            font-size: 36px;
            font-weight: 800;
            color: #1a1a2e;
        }
        .page-header p {
            color: #666;
            font-size: 16px;
            margin-top: 5px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        @media (max-width: 1024px) {
            .grid { grid-template-columns: 1fr; }
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 40px rgba(0,0,0,0.1);
        }
        .card.full-width {
            grid-column: 1 / -1;
        }
        .card h3 {
            font-size: 20px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card .card-subtitle {
            color: #888;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .card img {
            width: 100%;
            height: auto;
            border-radius: 12px;
        }
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="brand">⚡ Energy Dashboard</div>
        <div class="nav-links">
            <a href="/">🏠 Home</a>
            <a href="/dashboard" class="active">📊 Dashboard</a>
            <a href="/predict">🔮 Predict</a>
        </div>
    </nav>
    
    <div class="container">
        <div class="page-header">
            <h1>📊 PCA Analysis Dashboard</h1>
            <p>Interactive visualizations showing the results of Principal Component Analysis</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📈 Scree Plot</h3>
                <div class="card-subtitle">Explained variance by each principal component</div>
                <img src="/static/scree_plot.png" alt="Scree Plot">
            </div>
            
            <div class="card">
                <h3>📊 Cumulative Variance</h3>
                <div class="card-subtitle">Components needed to capture 95% variance</div>
                <img src="/static/cumulative_variance.png" alt="Cumulative Variance">
            </div>
            
            <div class="card full-width">
                <h3>🔥 Feature Loading Heatmap</h3>
                <div class="card-subtitle">Contribution of original features to principal components</div>
                <img src="/static/loading_heatmap.png" alt="Loading Heatmap">
            </div>
        </div>
    </div>
</body>
</html>
"""

# ============================================
# DYNAMIC PREDICT PAGE (FIXED)
# ============================================
def get_predict_html(features, prediction=None, form_data=None):
    if form_data is None:
        form_data = {}
    
    form_fields = ""
    for feature in features:
        value = form_data.get(feature, "")
        form_fields += f'''
            <div class="form-group">
                <label for="{feature}">{feature}:</label>
                <input type="number" step="any" id="{feature}" name="{feature}" 
                       value="{value}" placeholder="Enter {feature}" required>
            </div>
        '''
    
    result_html = ""
    if prediction is not None:
        result_html = f'''
            <div class="result-card">
                <div class="result-icon">✅</div>
                <div class="result-content">
                    <h3>Prediction Complete</h3>
                    <div class="prediction-value">{prediction} <span>kWh</span></div>
                    <p>Estimated energy consumption based on the provided inputs</p>
                </div>
            </div>
        '''
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predict - Energy Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Inter', sans-serif;
                background: #f0f2f5;
                min-height: 100vh;
            }}
            .navbar {{
                background: white;
                padding: 18px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            .navbar .brand {{
                font-size: 22px;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .navbar .nav-links a {{
                text-decoration: none;
                color: #555;
                margin: 0 20px;
                font-weight: 500;
                transition: color 0.3s;
                padding: 8px 16px;
                border-radius: 8px;
            }}
            .navbar .nav-links a:hover {{
                color: #667eea;
                background: #f0f2ff;
            }}
            .navbar .nav-links a.active {{
                color: #667eea;
                background: #f0f2ff;
            }}
            .container {{
                max-width: 900px;
                margin: 40px auto;
                padding: 0 30px;
            }}
            .main-card {{
                background: white;
                border-radius: 24px;
                padding: 50px;
                box-shadow: 0 4px 30px rgba(0,0,0,0.08);
            }}
            .main-card h1 {{
                font-size: 34px;
                font-weight: 800;
                color: #1a1a2e;
                margin-bottom: 8px;
            }}
            .main-card .subtitle {{
                color: #888;
                font-size: 16px;
                margin-bottom: 30px;
            }}
            .form-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            @media (max-width: 700px) {{
                .form-grid {{ grid-template-columns: 1fr; }}
                .main-card {{ padding: 30px; }}
            }}
            .form-group {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}
            .form-group label {{
                font-weight: 600;
                color: #333;
                font-size: 14px;
            }}
            .form-group input {{
                padding: 12px 16px;
                border: 2px solid #e8eaed;
                border-radius: 12px;
                font-size: 15px;
                font-family: 'Inter', sans-serif;
                transition: border-color 0.3s, box-shadow 0.3s;
                background: #fafbfc;
            }}
            .form-group input:focus {{
                border-color: #667eea;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
                outline: none;
                background: white;
            }}
            .form-group input::placeholder {{
                color: #aaa;
                font-size: 13px;
            }}
            .btn-predict {{
                width: 100%;
                padding: 16px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 20px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 20px;
                font-family: 'Inter', sans-serif;
                box-shadow: 0 8px 30px rgba(102, 126, 234, 0.35);
            }}
            .btn-predict:hover {{
                transform: translateY(-3px);
                box-shadow: 0 12px 40px rgba(102, 126, 234, 0.45);
            }}
            .result-card {{
                margin-top: 30px;
                padding: 30px;
                background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
                border-radius: 16px;
                display: flex;
                align-items: center;
                gap: 25px;
                animation: fadeIn 0.6s ease;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .result-icon {{
                font-size: 50px;
                background: #4caf50;
                width: 70px;
                height: 70px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                flex-shrink: 0;
            }}
            .result-content h3 {{
                color: #1a1a2e;
                font-size: 18px;
                margin-bottom: 4px;
            }}
            .result-content .prediction-value {{
                font-size: 42px;
                font-weight: 800;
                color: #2e7d32;
            }}
            .result-content .prediction-value span {{
                font-size: 20px;
                font-weight: 400;
                color: #555;
            }}
            .result-content p {{
                color: #555;
                font-size: 14px;
            }}
            .footer-note {{
                text-align: center;
                margin-top: 20px;
                color: #aaa;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="brand">⚡ Energy Dashboard</div>
            <div class="nav-links">
                <a href="/">🏠 Home</a>
                <a href="/dashboard">📊 Dashboard</a>
                <a href="/predict" class="active">🔮 Predict</a>
            </div>
        </nav>
        
        <div class="container">
            <div class="main-card">
                <h1>🔮 Predict Energy Consumption</h1>
                <p class="subtitle">Enter values for all features to get a real-time prediction</p>
                
                <form method="POST" action="/predict">
                    <div class="form-grid">
                        {form_fields}
                    </div>
                    <button type="submit" class="btn-predict">🚀 Get Prediction</button>
                </form>
                
                {result_html}
            </div>
            <div class="footer-note">Powered by PCA &amp; Machine Learning</div>
        </div>
    </body>
    </html>
    """

# ============================================
# ROUTES - FIXED PREDICT POST
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HOME_HTML

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML

@app.get("/predict", response_class=HTMLResponse)
async def predict_get():
    return get_predict_html(FEATURE_NAMES)

@app.post("/predict", response_class=HTMLResponse)
async def predict_post(request: Request):
    # Get form data from request
    form_data = await request.form()
    
    # Convert form data to dictionary
    data_dict = dict(form_data)
    
    input_values = []
    for feature in FEATURE_NAMES:
        value = data_dict.get(feature, "0")
        try:
            input_values.append(float(value))
        except ValueError:
            input_values.append(0.0)
    
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)[0]
    
    return get_predict_html(FEATURE_NAMES, round(float(prediction), 2), data_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)