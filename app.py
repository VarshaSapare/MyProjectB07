from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import datetime
import numpy as np
from PIL import Image
import mysql.connector
from mysql.connector import Error
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'cardiac_health'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ECG Class Mapping
ECG_CLASSES = {
    0: ('F', 'Atrial Fibrillation'),
    1: ('M', 'Murmur'),
    2: ('N', 'Normal'),
    3: ('Q', 'Premature Ventricular Contraction'),
    4: ('S', 'Supraventricular Tachycardia'),
    5: ('V', 'Ventricular Tachycardia'),
    6: ('I', 'Irrelevant / Not an ECG')
}

# ENHANCED ECG Risk Mapping with Detailed Descriptions
ECG_RISK_MAPPING = {
    'N': {
        'risk': 'Low', 
        'description': 'Normal ECG pattern - regular rhythm and normal intervals',
        'severity': 1,
        'implications': 'No immediate cardiac abnormalities detected',
        'color': '#28a745'
    },
    'F': {
        'risk': 'High', 
        'description': 'Atrial Fibrillation detected - irregular heart rhythm',
        'severity': 4,
        'implications': 'Increased risk of stroke and heart failure',
        'color': '#dc3545'
    },
    'M': {
        'risk': 'Moderate', 
        'description': 'Heart Murmur detected - abnormal heart sound',
        'severity': 2,
        'implications': 'Possible valve problems or structural heart issues',
        'color': '#fd7e14'
    },
    'Q': {
        'risk': 'High', 
        'description': 'Premature Ventricular Contraction - early heartbeat',
        'severity': 3,
        'implications': 'Potential for arrhythmias and cardiac events',
        'color': '#dc3545'
    },
    'S': {
        'risk': 'High', 
        'description': 'Supraventricular Tachycardia - rapid heart rate',
        'severity': 3,
        'implications': 'Risk of palpitations, dizziness, and fainting',
        'color': '#dc3545'
    },
    'V': {
        'risk': 'Critical', 
        'description': 'Ventricular Tachycardia - life-threatening rhythm',
        'severity': 5,
        'implications': 'Immediate medical attention required, risk of cardiac arrest',
        'color': '#dc3545'
    },
    'I': {
        'risk': 'Unknown', 
        'description': 'Image not recognized as valid ECG or poor quality',
        'severity': 0,
        'implications': 'Cannot assess cardiac status from provided image',
        'color': '#6c757d'
    },
    'ERR': {
        'risk': 'Error', 
        'description': 'Model error in processing ECG image',
        'severity': 0,
        'implications': 'Technical issue with ECG analysis',
        'color': '#6c757d'
    }
}

# Mapping dictionaries
yes_no_map = {'No': 0, 'Yes': 1}
sex_map = {'Female': 0, 'Male': 1}

general_health_map = {
    'Poor': 3, 'Fair': 1, 'Good': 2, 'Very Good': 4, 'Excellent': 0
}

checkup_map = {
    'Within the past year': 4,
    'Within the past 2 years': 2,
    'Within the past 5 years': 3,
    '5 or more years ago': 0,
    'Never': 1
}

diabetes_map = {
    'No': 0,
    'No, pre-diabetes or borderline diabetes': 1,
    'Yes': 2,
    'Yes, but female told only during pregnancy': 3
}

# Age category mapping
age_category_map = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3,
    '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7,
    '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11,
    '80+': 12
}

# Enhanced General Health Clusters with Detailed Information
GENERAL_HEALTH_CLUSTERS = {
    0: {
        'name': 'Optimal Health ',
        #'description': 'Excellent health conditions with predominantly female population. High fruit/vegetable consumption, minimal fried foods, regular exercise. All major health disorders show negative correlation.',
        'risk_level': 'Low',
        'severity': 1,
        'characteristics': [
            'Female majority (75%)',
            'Highest fruit & vegetable intake',
            'Lowest fried potato consumption',
            'Regular exercise habits',
            'Minimal smoking/alcohol'
        ],
        'recommendations': [
            'Maintain current healthy lifestyle',
            'Annual comprehensive health checkup',
            'Continue balanced diet',
            'Monitor bone health for arthritis prevention'
        ],
        'color': '#28a745'
    },
    1: {
        'name': 'Smoking & Arthritis Risk',
        'description': 'High-risk group with significant smoking history and arthritis prevalence. Heart disease rate 3x higher than average. Cancer risk elevated. Predominantly middle-aged males.',
        'risk_level': 'High',
        'severity': 4,
        'characteristics': [
            'High smoking prevalence (68%)',
            'Arthritis rate 12% higher than average',
            'Heart disease risk 3x normal',
            'Elevated cancer correlation',
            'Moderate alcohol consumption'
        ],
        'recommendations': [
            'Immediate smoking cessation program',
            'Cardiology consultation within 2 weeks',
            'Regular cancer screenings',
            'Anti-inflammatory diet for arthritis'
        ],
        'color': '#dc3545'
    },
    2: {
        'name': 'Moderate Lifestyle (Male Dominant)',
        'description': 'Moderate health risks with unhealthy lifestyle patterns. High fried food and alcohol consumption. Mostly male population with sedentary tendencies.',
        'risk_level': 'Moderate',
        'severity': 2,
        'characteristics': [
            'Male majority (82%)',
            'High fried potato consumption',
            'Elevated alcohol intake',
            'Irregular exercise patterns',
            'Average fruit/vegetable intake'
        ],
        'recommendations': [
            'Reduce fried food consumption',
            'Limit alcohol to moderate levels',
            'Increase physical activity',
            'Improve dietary balance'
        ],
        'color': '#ffc107'
    },
    3: {
        'name': 'High-Risk Multimorbidity',
        'description': 'Critical health status with multiple comorbidities. High prevalence of heart disease, cancer, diabetes, and depression. Older age group with poor lifestyle habits.',
        'risk_level': 'Critical',
        'severity': 5,
        'characteristics': [
            'Multiple disorders (heart, cancer, diabetes, depression)',
            'Older age groups (65+ years)',
            'Poor lifestyle habits',
            'Low fruit/vegetable consumption',
            'High medication usage'
        ],
        'recommendations': [
            'Immediate medical consultation',
            'Comprehensive health management plan',
            'Regular specialist follow-ups',
            'Lifestyle intervention program'
        ],
        'color': '#dc3545'
    },
    4: {
        'name': 'Young & Healthy ',
        'description': 'Very good health in younger population. Mild male predominance. All disorders negatively correlated. Active lifestyle with balanced nutrition.',
        'risk_level': 'Low',
        'severity': 1,
        'characteristics': [
            'Younger age groups (18-35)',
            'Mild male predominance (55%)',
            'Good exercise habits',
            'Balanced diet',
            'Minimal health issues'
        ],
        'recommendations': [
            'Continue healthy habits',
            'Regular preventive checkups',
            'Maintain active lifestyle',
            'Monitor health as age increases'
        ],
        'color': '#20c997'
    }
}

# COMPREHENSIVE COMBINED RISK MATRIX
# Format: COMBINED_RISK_MATRIX[ecg_code][health_cluster]
COMBINED_RISK_MATRIX = {
    # NORMAL ECG (N) - Best Case Scenario
    'N': {
        0: {  # Optimal Health + Normal ECG
            'final_risk': 'Very Low Risk',
            'description': 'Excellent overall health with normal cardiac function',
            'score': 1,
            'recommendations': [
                'Continue current healthy lifestyle',
                'Annual comprehensive health checkup',
                'Regular moderate exercise',
                'Maintain balanced diet'
            ],
            'medical_urgency': 'Routine follow-up',
            'follow_up_time': '12 months',
            'confidence': 'High'
        },
        1: {  # Smoking/Arthritis Risk + Normal ECG
            'final_risk': 'Moderate Risk',
            'description': 'Normal ECG but significant lifestyle risk factors present',
            'score': 3,
            'recommendations': [
                'Smoking cessation program immediately',
                'Cardiology consultation within 1 month',
                'Arthritis management plan',
                'Regular cancer screenings'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1 month',
            'confidence': 'Medium'
        },
        2: {  # Moderate Lifestyle + Normal ECG
            'final_risk': 'Low-Moderate Risk',
            'description': 'Normal cardiac function with unhealthy lifestyle habits',
            'score': 2,
            'recommendations': [
                'Reduce fried food consumption',
                'Limit alcohol intake',
                'Increase physical activity',
                'Improve dietary habits'
            ],
            'medical_urgency': 'Lifestyle modification',
            'follow_up_time': '3-6 months',
            'confidence': 'Medium'
        },
        3: {  # High-Risk Multimorbidity + Normal ECG
            'final_risk': 'High Risk',
            'description': 'Normal ECG but multiple comorbidities require attention',
            'score': 4,
            'recommendations': [
                'Immediate specialist consultations',
                'Comprehensive disease management',
                'Regular medication review',
                'Lifestyle intervention program'
            ],
            'medical_urgency': 'Urgent consultation',
            'follow_up_time': '2 weeks',
            'confidence': 'High'
        },
        4: {  # Young & Healthy + Normal ECG
            'final_risk': 'Very Low Risk',
            'description': 'Excellent health in young individual with normal heart function',
            'score': 1,
            'recommendations': [
                'Continue preventive health measures',
                'Annual health screening',
                'Maintain active lifestyle',
                'Regular health education'
            ],
            'medical_urgency': 'Routine follow-up',
            'follow_up_time': '12-24 months',
            'confidence': 'High'
        }
    },
    
    # ATRIAL FIBRILLATION (F) - High Risk ECG
    'F': {
        0: {  # Optimal Health + Atrial Fibrillation
            'final_risk': 'High Risk',
            'description': 'Atrial fibrillation detected despite otherwise good health',
            'score': 4,
            'recommendations': [
                'Immediate cardiology consultation',
                'ECG monitoring (Holter if needed)',
                'Stroke risk assessment',
                'Consider anticoagulation therapy'
            ],
            'medical_urgency': 'Urgent consultation',
            'follow_up_time': '1-2 weeks',
            'confidence': 'High'
        },
        1: {  # Smoking/Arthritis + Atrial Fibrillation
            'final_risk': 'Very High Risk',
            'description': 'Atrial fibrillation with multiple additional risk factors',
            'score': 5,
            'recommendations': [
                'Immediate medical attention',
                'Comprehensive cardiac evaluation',
                'Smoking cessation program',
                'Anticoagulation therapy likely needed'
            ],
            'medical_urgency': 'Immediate attention',
            'follow_up_time': '1 week',
            'confidence': 'High'
        },
        2: {  # Moderate Lifestyle + Atrial Fibrillation
            'final_risk': 'High Risk',
            'description': 'Atrial fibrillation with unhealthy lifestyle patterns',
            'score': 4,
            'recommendations': [
                'Cardiology consultation within 2 weeks',
                'Lifestyle modification program',
                'Alcohol reduction',
                'Regular ECG monitoring'
            ],
            'medical_urgency': 'Urgent consultation',
            'follow_up_time': '2 weeks',
            'confidence': 'High'
        },
        3: {  # High-Risk Multimorbidity + Atrial Fibrillation
            'final_risk': 'Critical Risk',
            'description': 'Atrial fibrillation with multiple comorbidities - highest risk category',
            'score': 6,
            'recommendations': [
                'Emergency medical evaluation',
                'Hospital admission consideration',
                'Multidisciplinary care team',
                'Intensive monitoring required'
            ],
            'medical_urgency': 'Emergency care',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        },
        4: {  # Young & Healthy + Atrial Fibrillation
            'final_risk': 'Moderate-High Risk',
            'description': 'Atrial fibrillation in young otherwise healthy individual',
            'score': 3,
            'recommendations': [
                'Cardiology consultation within 2 weeks',
                'Thyroid function tests',
                'Echocardiogram recommended',
                'Lifestyle assessment'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '2-4 weeks',
            'confidence': 'Medium'
        }
    },
    
    # HEART MURMUR (M) - Moderate Risk ECG
    'M': {
        0: {  # Optimal Health + Heart Murmur
            'final_risk': 'Low-Moderate Risk',
            'description': 'Heart murmur detected in otherwise healthy individual',
            'score': 2,
            'recommendations': [
                'Cardiology consultation',
                'Echocardiogram recommended',
                'Regular follow-up',
                'Maintain healthy lifestyle'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1-3 months',
            'confidence': 'Medium'
        },
        1: {  # Smoking/Arthritis + Heart Murmur
            'final_risk': 'Moderate-High Risk',
            'description': 'Heart murmur with additional cardiovascular risk factors',
            'score': 3,
            'recommendations': [
                'Cardiology consultation within 1 month',
                'Echocardiogram required',
                'Smoking cessation',
                'Regular cardiac monitoring'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1 month',
            'confidence': 'Medium'
        },
        2: {  # Moderate Lifestyle + Heart Murmur
            'final_risk': 'Moderate Risk',
            'description': 'Heart murmur with lifestyle risk factors',
            'score': 3,
            'recommendations': [
                'Cardiology evaluation',
                'Lifestyle modification',
                'Echocardiogram recommended',
                'Regular follow-up'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1-2 months',
            'confidence': 'Medium'
        },
        3: {  # High-Risk Multimorbidity + Heart Murmur
            'final_risk': 'High Risk',
            'description': 'Heart murmur in patient with multiple comorbidities',
            'score': 4,
            'recommendations': [
                'Urgent cardiology consultation',
                'Comprehensive cardiac workup',
                'Multidisciplinary management',
                'Close monitoring required'
            ],
            'medical_urgency': 'Urgent consultation',
            'follow_up_time': '2 weeks',
            'confidence': 'High'
        },
        4: {  # Young & Healthy + Heart Murmur
            'final_risk': 'Low Risk',
            'description': 'Likely innocent murmur in young healthy individual',
            'score': 1,
            'recommendations': [
                'Cardiology evaluation for confirmation',
                'Echocardiogram if symptoms present',
                'Regular health monitoring',
                'Maintain active lifestyle'
            ],
            'medical_urgency': 'Routine follow-up',
            'follow_up_time': '6-12 months',
            'confidence': 'Medium'
        }
    },
    
    # PREMATURE VENTRICULAR CONTRACTION (Q) - High Risk ECG
    'Q': {
        0: {  # Optimal Health + PVC
            'final_risk': 'Moderate Risk',
            'description': 'PVCs detected in otherwise healthy individual',
            'score': 3,
            'recommendations': [
                'Cardiology consultation',
                'Holter monitoring recommended',
                'Stress test consideration',
                'Lifestyle assessment'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1 month',
            'confidence': 'Medium'
        },
        1: {  # Smoking/Arthritis + PVC
            'final_risk': 'High Risk',
            'description': 'PVCs with significant additional risk factors',
            'score': 4,
            'recommendations': [
                'Urgent cardiology consultation',
                'Comprehensive cardiac evaluation',
                'Smoking cessation immediately',
                'Regular cardiac monitoring'
            ],
            'medical_urgency': 'Urgent consultation',
            'follow_up_time': '2 weeks',
            'confidence': 'High'
        },
        2: {  # Moderate Lifestyle + PVC
            'final_risk': 'Moderate-High Risk',
            'description': 'PVCs with lifestyle risk factors',
            'score': 3,
            'recommendations': [
                'Cardiology consultation within 1 month',
                'Lifestyle modification',
                'Holter monitoring',
                'Reduce stimulant intake'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1 month',
            'confidence': 'Medium'
        },
        3: {  # High-Risk Multimorbidity + PVC
            'final_risk': 'Very High Risk',
            'description': 'PVCs in patient with multiple comorbidities',
            'score': 5,
            'recommendations': [
                'Emergency cardiology evaluation',
                'Hospital admission consideration',
                'Intensive monitoring',
                'Medication review'
            ],
            'medical_urgency': 'Emergency care',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        },
        4: {  # Young & Healthy + PVC
            'final_risk': 'Low-Moderate Risk',
            'description': 'Occasional PVCs in young healthy individual',
            'score': 2,
            'recommendations': [
                'Cardiology evaluation',
                'Monitor frequency and symptoms',
                'Reduce caffeine/alcohol if frequent',
                'Regular follow-up'
            ],
            'medical_urgency': 'Routine follow-up',
            'follow_up_time': '3-6 months',
            'confidence': 'Medium'
        }
    },
    
    # SUPRAVENTRICULAR TACHYCARDIA (S) - High Risk ECG
    'S': {
        0: {  # Optimal Health + SVT
            'final_risk': 'Moderate-High Risk',
            'description': 'SVT detected in otherwise healthy individual',
            'score': 3,
            'recommendations': [
                'Cardiology consultation within 2 weeks',
                'ECG event monitoring',
                'Trigger identification',
                'Consider ablation therapy'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '2-4 weeks',
            'confidence': 'High'
        },
        1: {  # Smoking/Arthritis + SVT
            'final_risk': 'High Risk',
            'description': 'SVT with multiple additional risk factors',
            'score': 4,
            'recommendations': [
                'Urgent cardiology consultation',
                'Comprehensive electrophysiology evaluation',
                'Smoking cessation',
                'Medication management'
            ],
            'medical_urgency': 'Urgent consultation',
            'follow_up_time': '1-2 weeks',
            'confidence': 'High'
        },
        2: {  # Moderate Lifestyle + SVT
            'final_risk': 'Moderate-High Risk',
            'description': 'SVT with lifestyle triggers',
            'score': 3,
            'recommendations': [
                'Cardiology consultation within 1 month',
                'Lifestyle modification',
                'Trigger avoidance',
                'Regular monitoring'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1 month',
            'confidence': 'Medium'
        },
        3: {  # High-Risk Multimorbidity + SVT
            'final_risk': 'Very High Risk',
            'description': 'SVT in patient with multiple comorbidities',
            'score': 5,
            'recommendations': [
                'Emergency evaluation',
                'Hospital admission likely needed',
                'Intensive monitoring',
                'Multidisciplinary management'
            ],
            'medical_urgency': 'Emergency care',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        },
        4: {  # Young & Healthy + SVT
            'final_risk': 'Moderate Risk',
            'description': 'SVT in young otherwise healthy individual',
            'score': 3,
            'recommendations': [
                'Cardiology consultation',
                'Vagal maneuver education',
                'Trigger identification',
                'Regular follow-up'
            ],
            'medical_urgency': 'Scheduled consultation',
            'follow_up_time': '1-2 months',
            'confidence': 'Medium'
        }
    },
    
    # VENTRICULAR TACHYCARDIA (V) - Critical Risk ECG
    'V': {
        0: {  # Optimal Health + VT
            'final_risk': 'Critical Risk',
            'description': 'Life-threatening arrhythmia detected - requires immediate attention',
            'score': 6,
            'recommendations': [
                'EMERGENCY MEDICAL ATTENTION REQUIRED',
                'Immediate hospital admission',
                'Cardiac monitoring in CCU/ICU',
                'Consider ICD implantation'
            ],
            'medical_urgency': 'Emergency',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        },
        1: {  # Smoking/Arthritis + VT
            'final_risk': 'Critical Risk',
            'description': 'Life-threatening arrhythmia with additional risk factors',
            'score': 6,
            'recommendations': [
                'EMERGENCY MEDICAL ATTENTION REQUIRED',
                'Immediate hospitalization',
                'Comprehensive cardiac evaluation',
                'Long-term management planning'
            ],
            'medical_urgency': 'Emergency',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        },
        2: {  # Moderate Lifestyle + VT
            'final_risk': 'Critical Risk',
            'description': 'Life-threatening arrhythmia requiring emergency care',
            'score': 6,
            'recommendations': [
                'EMERGENCY MEDICAL ATTENTION REQUIRED',
                'Immediate cardiac care unit admission',
                'Aggressive risk factor management',
                'Long-term follow-up essential'
            ],
            'medical_urgency': 'Emergency',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        },
        3: {  # High-Risk Multimorbidity + VT
            'final_risk': 'Critical Risk',
            'description': 'Life-threatening arrhythmia in high-risk patient - extreme urgency',
            'score': 7,
            'recommendations': [
                'EMERGENCY MEDICAL ATTENTION REQUIRED',
                'Immediate intensive care',
                'Multidisciplinary emergency team',
                'Advanced life support measures'
            ],
            'medical_urgency': 'Emergency',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        },
        4: {  # Young & Healthy + VT
            'final_risk': 'Critical Risk',
            'description': 'Life-threatening arrhythmia in young individual - urgent care needed',
            'score': 6,
            'recommendations': [
                'EMERGENCY MEDICAL ATTENTION REQUIRED',
                'Immediate cardiac evaluation',
                'Genetic testing consideration',
                'Family screening recommended'
            ],
            'medical_urgency': 'Emergency',
            'follow_up_time': 'Immediate',
            'confidence': 'High'
        }
    },
    
    # IRRELEVANT/NOT ECG (I) - Unknown Risk
    'I': {
        0: {  # Optimal Health + Irrelevant ECG
            'final_risk': 'Assessment Needed',
            'description': 'ECG image quality insufficient for analysis',
            'score': 0,
            'recommendations': [
                'Retake clear ECG image',
                'Consult healthcare provider for proper ECG',
                'Continue healthy lifestyle',
                'Regular health monitoring'
            ],
            'medical_urgency': 'Quality improvement',
            'follow_up_time': 'When clear ECG available',
            'confidence': 'Low'
        },
        1: {  # Smoking/Arthritis + Irrelevant ECG
            'final_risk': 'Assessment Needed',
            'description': 'ECG unclear, but significant health risks present',
            'score': 0,
            'recommendations': [
                'Obtain clear ECG for proper assessment',
                'Address smoking and arthritis risks',
                'Regular health screenings',
                'Lifestyle modification'
            ],
            'medical_urgency': 'Quality improvement',
            'follow_up_time': 'When clear ECG available',
            'confidence': 'Low'
        },
        2: {  # Moderate Lifestyle + Irrelevant ECG
            'final_risk': 'Assessment Needed',
            'description': 'ECG quality insufficient, lifestyle risks noted',
            'score': 0,
            'recommendations': [
                'Retake clear ECG image',
                'Improve lifestyle habits',
                'Regular health checkups',
                'Dietary improvements'
            ],
            'medical_urgency': 'Quality improvement',
            'follow_up_time': 'When clear ECG available',
            'confidence': 'Low'
        },
        3: {  # High-Risk Multimorbidity + Irrelevant ECG
            'final_risk': 'Assessment Needed',
            'description': 'ECG unclear, but multiple comorbidities require attention',
            'score': 0,
            'recommendations': [
                'Obtain proper ECG urgently',
                'Continue medical management',
                'Regular specialist follow-ups',
                'Close health monitoring'
            ],
            'medical_urgency': 'Quality improvement',
            'follow_up_time': 'When clear ECG available',
            'confidence': 'Low'
        },
        4: {  # Young & Healthy + Irrelevant ECG
            'final_risk': 'Assessment Needed',
            'description': 'ECG image quality needs improvement for assessment',
            'score': 0,
            'recommendations': [
                'Retake clear ECG image',
                'Continue preventive health measures',
                'Maintain healthy lifestyle',
                'Regular health education'
            ],
            'medical_urgency': 'Quality improvement',
            'follow_up_time': 'When clear ECG available',
            'confidence': 'Low'
        }
    },
    
    # ERROR (ERR) - Technical Issues
    'ERR': {
        0: {  # Optimal Health + Error
            'final_risk': 'Technical Issue',
            'description': 'ECG analysis error in otherwise healthy individual',
            'score': 0,
            'recommendations': [
                'Retry ECG analysis with different image',
                'Consult healthcare provider if symptoms',
                'Continue healthy lifestyle',
                'Regular health monitoring'
            ],
            'medical_urgency': 'Technical retry',
            'follow_up_time': 'When analysis successful',
            'confidence': 'Low'
        },
        1: {  # Smoking/Arthritis + Error
            'final_risk': 'Technical Issue',
            'description': 'ECG analysis error with existing health risks',
            'score': 0,
            'recommendations': [
                'Retry ECG analysis',
                'Address smoking and arthritis risks',
                'Consider professional ECG',
                'Regular health screenings'
            ],
            'medical_urgency': 'Technical retry',
            'follow_up_time': 'When analysis successful',
            'confidence': 'Low'
        },
        2: {  # Moderate Lifestyle + Error
            'final_risk': 'Technical Issue',
            'description': 'ECG analysis error with lifestyle risks',
            'score': 0,
            'recommendations': [
                'Retry ECG analysis',
                'Improve lifestyle habits',
                'Regular health checkups',
                'Dietary improvements'
            ],
            'medical_urgency': 'Technical retry',
            'follow_up_time': 'When analysis successful',
            'confidence': 'Low'
        },
        3: {  # High-Risk Multimorbidity + Error
            'final_risk': 'Technical Issue',
            'description': 'ECG analysis error in high-risk patient',
            'score': 0,
            'recommendations': [
                'Professional ECG evaluation recommended',
                'Continue medical management',
                'Regular specialist follow-ups',
                'Close health monitoring'
            ],
            'medical_urgency': 'Technical retry',
            'follow_up_time': 'When analysis successful',
            'confidence': 'Low'
        },
        4: {  # Young & Healthy + Error
            'final_risk': 'Technical Issue',
            'description': 'ECG analysis error in young healthy individual',
            'score': 0,
            'recommendations': [
                'Retry ECG analysis',
                'Continue preventive health measures',
                'Maintain healthy lifestyle',
                'Regular health education'
            ],
            'medical_urgency': 'Technical retry',
            'follow_up_time': 'When analysis successful',
            'confidence': 'Low'
        }
    }
}

# Risk Score Interpretation
RISK_SCORE_INTERPRETATION = {
    0: {'level': 'Unassessed', 'color': '#6c757d', 'description': 'Risk assessment not possible'},
    1: {'level': 'Very Low', 'color': '#28a745', 'description': 'Minimal risk, excellent health'},
    2: {'level': 'Low', 'color': '#20c997', 'description': 'Low risk, good health'},
    3: {'level': 'Moderate', 'color': '#ffc107', 'description': 'Moderate risk, needs attention'},
    4: {'level': 'High', 'color': '#fd7e14', 'description': 'High risk, requires intervention'},
    5: {'level': 'Very High', 'color': '#dc3545', 'description': 'Very high risk, urgent attention needed'},
    6: {'level': 'Critical', 'color': '#dc3545', 'description': 'Critical risk, emergency care required'},
    7: {'level': 'Extreme', 'color': '#8b0000', 'description': 'Extreme risk, immediate emergency care'}
}

# Load Models
try:
    ECG_MODEL = tf.keras.models.load_model('models/best_cnn_balanced_model.h5')
    print("✅ ECG CNN model loaded successfully")
except Exception as e:
    print(f"❌ ECG Model load error: {e}")
    ECG_MODEL = None

try:
    with open('models/random_forest_model.pkl', 'rb') as f:
        GENERAL_HEALTH_MODEL = joblib.load(f)
    print("✅ General Health random forest model loaded successfully")
except Exception as e:
    print(f"❌ General Health model error: {e}")
    GENERAL_HEALTH_MODEL = None

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        return filename
    return None

def get_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(e)
        return None

def predict_ecg(image_path):
    if not ECG_MODEL:
        return {'code': 'ERR', 'name': 'Model Not Loaded', 'confidence': 0.0, 'risk': 'Error'}
    
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        probabilities = ECG_MODEL.predict(img_array, verbose=0)[0]
        idx = np.argmax(probabilities)
        confidence = float(probabilities[idx])

        REJECTION_THRESHOLD = 0.65

        if confidence < REJECTION_THRESHOLD:
            final_class = 6
            final_confidence = min(0.98, 1.0 - confidence)
            name = "Not an ECG image / Irrelevant"
            code = "I"
        else:
            final_class = idx
            final_confidence = confidence
            code, name = ECG_CLASSES.get(idx, ('UNK', 'Unknown'))
        
        risk_info = ECG_RISK_MAPPING.get(code, ECG_RISK_MAPPING['ERR'])
        
        return {
            'code': code,
            'name': name,
            'confidence': final_confidence,
            'risk': risk_info['risk'],
            'severity': risk_info['severity'],
            'risk_description': risk_info['description'],
            'implications': risk_info['implications'],
            'color': risk_info['color'],
            'raw_probabilities': probabilities.tolist()
        }
    except Exception as e:
        print(f"ECG prediction error: {e}")
        return {'code': 'ERR', 'name': 'Prediction Error', 'confidence': 0.0, 'risk': 'Error', 'risk_description': 'Error processing ECG image'}

def predict_general_health(form_data):
    if not GENERAL_HEALTH_MODEL:
        return {'cluster': -1, 'label': 'Error', 'description': 'Model Not Loaded', 'confidence': 0.0, 'risk_level': 'Error'}
    
    required = ['general_health', 'checkup', 'exercise', 'heart_disease', 'skin_cancer',
                'other_cancer', 'depression', 'diabetes', 'arthritis', 'sex',
                'age_category', 'height_cm', 'weight_kg', 'bmi', 'smoking_history',
                'alcohol_consumption', 'fruit_consumption', 'green_vegetables_consumption',
                'friedpotato_consumption']
    
    missing = [f for f in required if f not in form_data]
    if missing:
        print(f"WARNING: Missing fields in form: {missing}")
    try:
        # Convert form data to model features
        features = [
                general_health_map.get(form_data['general_health'], 2),      # 0
                checkup_map.get(form_data['checkup'], 4),                    # 1
                yes_no_map.get(form_data['exercise'], 0),                    # 2
                yes_no_map.get(form_data['heart_disease'], 0),               # 3  ← ADD THIS
                yes_no_map.get(form_data['skin_cancer'], 0),                 # 4
                yes_no_map.get(form_data['other_cancer'], 0),                # 5
                yes_no_map.get(form_data['depression'], 0),                  # 6
                diabetes_map.get(form_data['diabetes'], 0),                  # 7
                yes_no_map.get(form_data['arthritis'], 0),                   # 8
                sex_map.get(form_data['sex'], 1),                            # 9
                age_category_map.get(form_data['age_category'], 0),          # 10
                float(form_data['height_cm']),                               # 11
                float(form_data['weight_kg']),                               # 12
                float(form_data['bmi']),                                     # 13
                yes_no_map.get(form_data['smoking_history'], 0),             # 14
                int(form_data['alcohol_consumption']),                       # 15
                int(form_data['fruit_consumption']),                         # 16
                int(form_data['green_vegetables_consumption']),              # 17
                int(form_data['friedpotato_consumption'])                    # 18
            ]

        X = np.array(features).reshape(1, -1)
        
        # Predict with random forest
        cluster = int(GENERAL_HEALTH_MODEL.predict(X)[0])
        probs = GENERAL_HEALTH_MODEL.predict_proba(X)[0]
        confidence = float(probs[cluster])
        
        cluster_info = GENERAL_HEALTH_CLUSTERS.get(cluster, {
            'name': 'Unknown',
            'description': 'No information available',
            'risk_level': 'Unknown',
            'severity': 0,
            'recommendations': ['Consult healthcare provider'],
            'color': '#6c757d'
        })
        
        return {
            'cluster': cluster,
            'label': cluster_info['name'],
            'description': cluster_info['description'],
            'confidence': confidence,
            'risk_level': cluster_info['risk_level'],
            'severity': cluster_info['severity'],
            'characteristics': cluster_info.get('characteristics', []),
            'recommendations': cluster_info['recommendations'],
            'color': cluster_info['color'],
            'all_probabilities': probs.tolist()
        }
    except Exception as e:
        print(f"Health prediction error: {e}")
        return {'cluster': -1, 'label': 'Error', 'description': str(e), 'confidence': 0.0, 'risk_level': 'Error'}

def calculate_combined_risk(ecg_prediction, health_prediction):
    """Calculate comprehensive combined risk assessment"""
    
    ecg_code = ecg_prediction.get('code', 'ERR')
    health_cluster = health_prediction.get('cluster', -1)
    
    # Get combined risk details from matrix
    combined_info = COMBINED_RISK_MATRIX.get(ecg_code, {}).get(health_cluster, {})
    
    if not combined_info:
        # Default if combination not found
        combined_info = {
            'final_risk': 'Assessment Needed',
            'description': 'Combination not in risk matrix',
            'score': 0,
            'recommendations': ['Consult healthcare provider for assessment'],
            'medical_urgency': 'Consultation needed',
            'follow_up_time': 'As soon as possible',
            'confidence': 'Low'
        }
    
    # Get risk score interpretation
    score = combined_info.get('score', 0)
    score_info = RISK_SCORE_INTERPRETATION.get(score, RISK_SCORE_INTERPRETATION[0])
    
    # Generate comprehensive explanations
    explanations = []
    
    # ECG explanation
    ecg_name = ecg_prediction.get('name', 'Unknown ECG')
    ecg_risk = ecg_prediction.get('risk', 'Unknown')
    ecg_desc = ecg_prediction.get('risk_description', '')
    
    explanations.append(f"ECG Finding: {ecg_name} ({ecg_risk} Risk)")
    explanations.append(f"ECG Implication: {ecg_desc}")
    
    # Health explanation
    health_label = health_prediction.get('label', 'Unknown Health Profile')
    health_risk = health_prediction.get('risk_level', 'Unknown')
    
    explanations.append(f"Health Profile: {health_label} ({health_risk} Risk)")
    
    # Combined explanation
    explanations.append(f"Combined Assessment: {combined_info['description']}")
    
    # Medical urgency explanation
    urgency = combined_info.get('medical_urgency', 'Consultation needed')
    follow_up = combined_info.get('follow_up_time', 'As soon as possible')
    
    explanations.append(f"Medical Urgency: {urgency} (Follow-up: {follow_up})")
    
    # Combine recommendations
    all_recommendations = []
    
    # Add ECG-specific recommendations
    if ecg_code != 'I' and ecg_code != 'ERR':
        all_recommendations.append(f"ECG-specific: {ecg_prediction.get('implications', 'Consult cardiologist')}")
    
    # Add health cluster recommendations
    health_recs = health_prediction.get('recommendations', [])
    if isinstance(health_recs, list):
        all_recommendations.extend([f"Health: {rec}" for rec in health_recs])
    
    # Add combined recommendations
    combined_recs = combined_info.get('recommendations', [])
    if isinstance(combined_recs, list):
        all_recommendations.extend([f"Combined: {rec}" for rec in combined_recs])
    
    # Remove duplicates while preserving order
    unique_recommendations = []
    seen = set()
    for rec in all_recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    # Determine severity color
    severity_color = score_info['color']
    
    # Generate risk badge text
    risk_badge = f"{score_info['level']} Risk (Score: {score}/7)"
    
    return {
        'final_risk': combined_info['final_risk'],
        'risk_score': score,
        'risk_level': score_info['level'],
        'risk_description': score_info['description'],
        'risk_badge': risk_badge,
        'severity_color': severity_color,
        'explanations': explanations,
        'recommendations': unique_recommendations,
        'medical_urgency': urgency,
        'follow_up_time': follow_up,
        'confidence': combined_info.get('confidence', 'Medium'),
        'ecg_code': ecg_code,
        'ecg_risk': ecg_risk,
        'health_cluster': health_cluster,
        'health_risk': health_risk,
        'matrix_description': combined_info['description']
    }



@app.route('/combined-assessment', methods=['GET', 'POST'])
def combined_assessment():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get form data
        data = request.form.to_dict()
        
        # Handle ECG image upload
        ecg_file = request.files.get('ecg_image')
        ecg_filename = None
        ecg_prediction = None
        print("DEBUG: Form fields:", list(request.form.keys()))
        print("DEBUG: Files received:", list(request.files.keys()))
        if ecg_file and allowed_file(ecg_file.filename):
            ecg_filename = save_file(ecg_file)
            if ecg_filename:
                path = os.path.join(app.config['UPLOAD_FOLDER'], ecg_filename)
                ecg_prediction = predict_ecg(path)
                print("DEBUG: ECG file name:", ecg_file.filename)
                print("DEBUG: ECG file size:", ecg_file.content_length)
                print("DEBUG: ECG file type:", ecg_file.content_type)
            else:
                flash('Invalid ECG image file', 'danger')
                return redirect(request.url)
        else:
            # If no ECG file uploaded
            ecg_prediction = {
                'code': 'I',
                'name': 'No ECG Uploaded',
                'confidence': 0.0,
                'risk': 'Unknown',
                'risk_description': 'ECG image not provided'
            }
            flash('No ECG image uploaded. Health assessment only.', 'info')
        
        # Get health prediction
        health_prediction = predict_general_health(data)
        
        # Calculate combined risk
        # combined_result = calculate_combined_risk(ecg_prediction, health_prediction)
        combined_result = calculate_combined_risk(ecg_prediction, health_prediction)
        
        # Prepare enumerated recommendations **only here**
        enumerated_recommendations = list(enumerate(combined_result['recommendations'], start=1))
        # Save to database
        conn = get_db()
        if conn:
            cur = conn.cursor()
            try:
                # Save ECG prediction
                if ecg_filename:
                    cur.execute("""
                        INSERT INTO ecg_predictions 
                        (user_id, image_filename, prediction_code, prediction_name, confidence)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (session['user_id'], ecg_filename, 
                          ecg_prediction.get('code'), 
                          ecg_prediction.get('name'), 
                          ecg_prediction.get('confidence')))
                
                # Save health prediction
                cur.execute("""
                    INSERT INTO health_predictions
                    (user_id, age_category, bmi, smoking, alcohol, exercise,
                     skin_cancer, other_cancer, depression, diabetes, arthritis, sex,
                     cluster, cluster_name, confidence, general_health, height_cm, weight_kg,
                     smoking_history, fruit_consumption, green_vegetables_consumption, 
                     friedpotato_consumption, final_risk, risk_score, medical_urgency)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    session['user_id'],
                    data.get('age_category'),
                    float(data.get('bmi', 0)),
                    yes_no_map.get(data.get('smoking_history'), 0),
                    int(data.get('alcohol_consumption', 0)),
                    yes_no_map.get(data.get('exercise'), 0),
                    yes_no_map.get(data.get('skin_cancer'), 0),
                    yes_no_map.get(data.get('other_cancer'), 0),
                    yes_no_map.get(data.get('depression'), 0),
                    data.get('diabetes'),
                    yes_no_map.get(data.get('arthritis'), 0),
                    data.get('sex'),
                    health_prediction.get('cluster', -1),
                    health_prediction.get('label', 'Unknown'),
                    health_prediction.get('confidence', 0),
                    data.get('general_health'),
                    float(data.get('height_cm', 0)),
                    float(data.get('weight_kg', 0)),
                    yes_no_map.get(data.get('smoking_history'), 0),
                    int(data.get('fruit_consumption', 0)),
                    int(data.get('green_vegetables_consumption', 0)),
                    int(data.get('friedpotato_consumption', 0)),
                    combined_result.get('final_risk', 'Unknown'),
                    combined_result.get('risk_score', 0),
                    combined_result.get('medical_urgency', 'Unknown')
                ))
                
                conn.commit()
                
            except Exception as e:
                print(f"Database insert error: {e}")
                conn.rollback()
                # flash('Error saving prediction data', 'danger')
            finally:
                cur.close()
                conn.close()
        
        # Render combined results
        return render_template('combined_result.html',
                               ecg_prediction=ecg_prediction,
                               health_prediction=health_prediction,
                               combined_result=combined_result,
                               data=data,
                               ecg_image=ecg_filename if ecg_filename else None,
                               clusters_info=GENERAL_HEALTH_CLUSTERS,
                               ecg_risk_mapping=ECG_RISK_MAPPING,
                               risk_score_interpretation=RISK_SCORE_INTERPRETATION,
                               combined_matrix=COMBINED_RISK_MATRIX,
                               enumerated_recommendations=enumerated_recommendations)

    # GET request - show the combined form
    return render_template('combined_assessment.html')

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if request.form['password'] != request.form['confirm_password']:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        hashed = generate_password_hash(password)
        conn = get_db()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                            (username, email, hashed))
                conn.commit()
                flash('Registered successfully! Please login.', 'success')
                return redirect(url_for('login'))
            except mysql.connector.IntegrityError:
                flash('Username or email already exists!', 'danger')
            finally:
                cur.close()
                conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        if conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE username=%s", (username,))
            user = cur.fetchone()
            cur.close()
            conn.close()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('home'))

@app.route('/ecg', methods=['GET', 'POST'])
def ecg_diagnosis():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        file = request.files['ecg_image']
        filename = save_file(file)
        if not filename:
            flash('Invalid file type', 'danger')
            return redirect(request.url)
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        prediction = predict_ecg(path)

        conn = get_db()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ecg_predictions 
                (user_id, image_filename, prediction_code, prediction_name, confidence)
                VALUES (%s, %s, %s, %s, %s)
            """, (session['user_id'], filename, prediction['code'], prediction['name'], prediction['confidence']))
            conn.commit()
            cur.close()
            conn.close()

        return render_template('ecg_result.html', prediction=prediction, image=filename)
    
    return render_template('ecg_diagnosis.html')

@app.route('/health', methods=['GET', 'POST'])
def health_input():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
   
    if request.method == 'POST':
        data = request.form.to_dict()
        prediction = predict_general_health(data)
        conn = get_db()
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO health_predictions
                    (user_id, age_category, bmi, smoking, alcohol, exercise,
                     heart_disease, skin_cancer, other_cancer, depression, diabetes, arthritis, sex,
                     prediction_result, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    session['user_id'],
                    data['age_category'],
                    float(data['bmi']),
                    int(yes_no_map.get(data['smoking_history'], 0)),
                    int(data['alcohol_consumption']),
                    int(yes_no_map.get(data['exercise'], 0)),
                    int(yes_no_map.get(data['heart_disease'], 0)),  # Input heart_disease
                    int(yes_no_map.get(data['skin_cancer'], 0)),
                    int(yes_no_map.get(data['other_cancer'], 0)),
                    int(yes_no_map.get(data['depression'], 0)),
                    data['diabetes'],
                    int(yes_no_map.get(data['arthritis'], 0)),
                    data['sex'],
                    prediction['label'],  # e.g., 'High Risk (Major Disorders, Unhealthy Lifestyle)'
                    prediction['confidence']
                ))
                conn.commit()
            except Exception as e:
                print(f"Database insert error: {e}")
                flash('Error saving prediction', 'danger')
            finally:
                cur.close()
                conn.close()
        cluster_num = prediction['cluster']
        cluster_title = prediction['label']
        cluster_desc  = prediction['description']

        return render_template(
            'health_result.html',
            prediction=prediction,
            data=data,
            cluster_num=cluster_num,
            cluster_title=cluster_title,
            cluster_desc=cluster_desc
        )
   
    return render_template('health_input.html')

@app.route('/history/ecg')
def history_ecg():
    if 'user_id' not in session: return redirect(url_for('login'))
    conn = get_db()
    predictions = []
    if conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM ecg_predictions WHERE user_id=%s ORDER BY prediction_date DESC", (session['user_id'],))
        predictions = cur.fetchall()
        cur.close()
        conn.close()
    return render_template('history_ecg.html', predictions=predictions)

@app.route('/history/health')
def history_health():
    if 'user_id' not in session: return redirect(url_for('login'))
    conn = get_db()
    predictions = []
    if conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM health_predictions WHERE user_id=%s ORDER BY prediction_date DESC", (session['user_id'],))
        predictions = cur.fetchall()
        cur.close()
        conn.close()
    return render_template('history_health.html', predictions=predictions)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)