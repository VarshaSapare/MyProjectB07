from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# ECG page
@app.route('/ecg', methods=['GET', 'POST'])
def ecg_diagnosis():
    if request.method == 'POST':
        file = request.files.get('ecg_image')
        filename = None

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Dummy ECG prediction (frontend demo only)
        prediction = {
            'code': 'F',
            'name': 'Atrial Fibrillation',
            'confidence': 0.92,
            'risk': 'High',
            'severity': 4,
            'risk_description': 'Irregular heart rhythm detected',
            'implications': 'Increased risk of stroke and heart failure',
            'color': '#dc3545'
        }

        return render_template('ecg_result.html', prediction=prediction, image=filename)

    return render_template('ecg_diagnosis.html')

# Health page
@app.route('/health', methods=['GET', 'POST'])
def health_input():
    if request.method == 'POST':
        data = request.form.to_dict()

        # Dummy health prediction (frontend demo only)
        prediction = {
            'cluster': 3,
            'label': 'High-Risk Multimorbidity',
            'description': 'Critical health status with multiple comorbidities.',
            'confidence': 0.88,
            'risk_level': 'Critical',
            'severity': 5,
            'characteristics': [
                'Multiple disorders (heart, cancer, diabetes, depression)',
                'Older age groups (65+ years)',
                'Poor lifestyle habits'
            ],
            'recommendations': [
                'Immediate medical consultation',
                'Comprehensive health management plan',
                'Regular specialist follow-ups'
            ],
            'color': '#dc3545'
        }

        return render_template(
            'health_result.html',
            prediction=prediction,
            data=data,
            cluster_num=prediction['cluster'],
            cluster_title=prediction['label'],
            cluster_desc=prediction['description']
        )

    return render_template('health_input.html')

# Combined assessment page
@app.route('/combined-assessment', methods=['GET', 'POST'])
def combined_assessment():
    if request.method == 'POST':
        data = request.form.to_dict()

        file = request.files.get('ecg_image')
        filename = None
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Dummy ECG prediction
        ecg_prediction = {
            'code': 'F',
            'name': 'Atrial Fibrillation',
            'confidence': 0.92,
            'risk': 'High',
            'severity': 4,
            'risk_description': 'Irregular heart rhythm detected',
            'implications': 'Increased risk of stroke and heart failure',
            'color': '#dc3545'
        }

        # Dummy Health prediction
        health_prediction = {
            'cluster': 3,
            'label': 'High-Risk Multimorbidity',
            'description': 'Critical health status with multiple comorbidities.',
            'confidence': 0.88,
            'risk_level': 'Critical',
            'severity': 5,
            'characteristics': [
                'Multiple disorders (heart, cancer, diabetes, depression)',
                'Older age groups (65+ years)',
                'Poor lifestyle habits'
            ],
            'recommendations': [
                'Immediate medical consultation',
                'Comprehensive health management plan',
                'Regular specialist follow-ups'
            ],
            'color': '#dc3545'
        }

        # Dummy combined result
        combined_result = {
            'final_risk': 'Critical Risk',
            'risk_score': 6,
            'risk_level': 'Critical',
            'risk_description': 'Critical risk, emergency care required',
            'risk_badge': 'Critical Risk (Score: 6/7)',
            'severity_color': '#dc3545',
            'explanations': [
                'ECG Finding: Atrial Fibrillation (High Risk)',
                'Health Profile: High-Risk Multimorbidity (Critical Risk)',
                'Combined Assessment: Severe cardiac and health risk detected'
            ],
            'recommendations': [
                'Immediate cardiology consultation',
                'Comprehensive cardiac evaluation',
                'Regular specialist follow-up',
                'Lifestyle intervention program'
            ],
            'medical_urgency': 'Emergency care',
            'follow_up_time': 'Immediate',
            'confidence': 'High',
            'ecg_code': 'F',
            'ecg_risk': 'High',
            'health_cluster': 3,
            'health_risk': 'Critical',
            'matrix_description': 'Severe cardiac and health risk detected'
        }

        enumerated_recommendations = list(enumerate(combined_result['recommendations'], start=1))

        return render_template(
            'combined_result.html',
            ecg_prediction=ecg_prediction,
            health_prediction=health_prediction,
            combined_result=combined_result,
            data=data,
            ecg_image=filename if filename else None,
            enumerated_recommendations=enumerated_recommendations
        )

    return render_template('combined_assessment.html')

# History pages (demo)
@app.route('/history/ecg')
def history_ecg():
    predictions = []
    return render_template('history_ecg.html', predictions=predictions)

@app.route('/history/health')
def history_health():
    predictions = []
    return render_template('history_health.html', predictions=predictions)

# Login/Register demo pages (optional frontend only)
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/logout')
def logout():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)