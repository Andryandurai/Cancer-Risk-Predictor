from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'ahbahrir'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

def generate_next_health_id():
    last_user = User.query.order_by(User.health_id.desc()).first()
    if last_user is None:
        return 10000
    return last_user.health_id + 1

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    health_id = db.Column(db.Integer, unique=True, nullable=False)
    user_symptom = db.relationship('UserSymptom', uselist=False, backref='user')
    risk_results = db.relationship('RiskResult', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Symptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptom_name = db.Column(db.String(120), unique=True, nullable=False)
    category = db.Column(db.String(20))

class UserSymptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    general_symptoms = db.Column(db.String)  
    breast_symptoms = db.Column(db.String)   
    skin_symptoms = db.Column(db.String)     
    kidney_symptoms = db.Column(db.String)   

class RiskResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    risk_percentage = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    recommendation = db.Column(db.String(255))
    date = db.Column(db.DateTime, default=datetime.utcnow)


SYMPTOMS_GROUPED = {
    'Lung Cancer': [
        'Persistent_cough_greater_than_2weeks', 'Blood_in_sputum', 'Chest_pain_tightness', 'Shortness_of_breath_wheezing',
        'Hoarse_voice', 'Recurring_chest_infections', 'Unexplained_weight_loss', 'Fatigue'
    ],
    'Breast Cancer': [
        'Breast_lump', 'Breast_asymmetry', 'Skin_changes_in_breast', 'Nipple_discharge', 'Nipple_inversion',
        'Breast_pain_non_menstrual', 'Visible_breast_veins', 'Swelling'
    ],
    'Skin Cancer': [
        'New_mole_spot', 'Mole_shape_change', 'Lesion_bleeds_oozes', 'Sore_not_healing_greater_than_3weeks',
        'Rough_scaly_patch', 'Itchy_painful_lesion', 'Dark_streaks_under_nails', 'Persistent_red_patches'
    ],
    'Kidney Cancer': [
        'Blood_in_urine', 'Persistent_side_flank_pain', 'Abdominal_lump', 'Swelling_ankles_legs_abdomen',
        'Loss_of_appetite', 'Fever_non_infection', 'Hypertension', 'Night_sweats'
    ]
}

def prettify_symptom(symptom):
    return symptom.replace('_greater_than_', ' greater than ').replace('_', ' ')

app.jinja_env.filters['prettify_symptom'] = prettify_symptom

MODEL_FILE = 'cancer_risk_model.pkl'
FEATURES_FILE = 'model_features.pkl'

def generate_recommendation(risk_score_percent):
    if risk_score_percent < 20:
        risk_level = "Low"
        recommendation = "Maintain a healthy lifestyle. Make sure to continue routine checkups."
    elif 20 <= risk_score_percent < 50:
        risk_level = "Medium"
        recommendation = "Consult a Family Doctor or Family Physician for a check-up and mention your symptoms."
    else:
        risk_level = "High"
        recommendation = "Immediate consultation with a Medical specialist is strongly recommended."
    return {
        'risk_percentage': f"{risk_score_percent:.2f}%",
        'risk_level': risk_level,
        'recommendation': recommendation
    }

def populate_symptoms():
    if Symptom.query.count() == 0:
        for cat, syms in SYMPTOMS_GROUPED.items():
            for symp in syms:
                db.session.add(Symptom(symptom_name=symp, category=cat))
        db.session.commit()

@app.route('/register', methods=['GET', 'POST'])
def register():
    errors = {}
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        age = request.form['age']
        email = request.form['email']
        phone = request.form['phone']

        if User.query.filter_by(email=email).first():
            errors['email'] = "Email already registered."
        if User.query.filter_by(username=username).first():
            errors['username'] = "Username already taken."

        try:
            age_int = int(age)
            if age_int < 0 or age_int > 125:
                errors['age'] = "Age must be between 0 and 125."
        except ValueError:
            errors['age'] = "Please enter a valid age."

        if not (phone.isdigit() and len(phone) == 10):
            errors['phone'] = "Phone number must be exactly 10 digits."

        if len(password) < 6:
            errors['password'] = "Password must be at least 6 characters."

        if errors:
            return render_template('register.html', errors=errors, form=request.form)

        health_id = generate_next_health_id()
        user = User(username=username, age=age_int, email=email, phone=phone, health_id=health_id)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        session['health_id'] = user.health_id
        return redirect(url_for('index'))
    return render_template('register.html', errors=errors)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            error = "Please enter both username and password."
            return render_template('login.html', error=error)

        user = User.query.filter_by(username=username).first()

        if user is None:
            error = "User with that username does not exist."
            return render_template('login.html', error=error)

        if not user.check_password(password):
            error = "Incorrect password. Please try again."
            return render_template('login.html', error=error)

        session['user_id'] = user.id
        session['health_id'] = user.health_id
        flash('Login successful!')
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    gender_options = ['Male', 'Female', 'Other']
    smoking_options = ['Yes', 'No', 'Former']
    return render_template('index.html',
                           symptom_groups=SYMPTOMS_GROUPED,
                           gender_options=gender_options,
                           smoking_options=smoking_options,
                           health_id=session.get('health_id'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('login'))

    if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
        return render_template('result.html', result={
            'risk_level': "Error",
            'recommendation': "ML model files not found. Please run the original training script first."
        })

    model = joblib.load(MODEL_FILE)
    all_features_list = joblib.load(FEATURES_FILE)
    form_data = request.form

    age = int(form_data.get('age', user.age))
    gender = form_data.get('gender', 'Other')
    smoking = form_data.get('smoking_history', 'No')
    family_history = int(form_data.get('family_history', 0))

    input_data = {
        'Age': age,
        'Family_History_Cancer': family_history,
        'Gender': gender,
        'Smoking_History': smoking
    }

    for cat_symptoms in SYMPTOMS_GROUPED.values():
        for symp in cat_symptoms:
            input_data[symp] = 0

    checked_symptoms = {cat: [] for cat in SYMPTOMS_GROUPED.keys()}
    for cat, syms in SYMPTOMS_GROUPED.items():
        for symp in syms:
            if form_data.get(symp) == 'on':
                input_data[symp] = 1
                checked_symptoms[cat].append(symp)

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['Gender', 'Smoking_History'], drop_first=True)
    final_input = pd.Series(0, index=all_features_list)
    for col in final_input.index:
        if col in input_df.columns:
            final_input[col] = input_df[col].iloc[0]

    X_new_user = pd.DataFrame([final_input])
    risk_prob = model.predict_proba(X_new_user)[0][1]
    risk_percent = risk_prob * 100
    result = generate_recommendation(risk_percent)

    rr = RiskResult(user_id=user.id, risk_percentage=risk_percent,
                    risk_level=result['risk_level'], recommendation=result['recommendation'])
    db.session.add(rr)

    us = UserSymptom.query.filter_by(user_id=user.id).first()
    if not us:
        us = UserSymptom(user_id=user.id)
    us.general_symptoms = ', '.join(checked_symptoms.get('Lung Cancer', []))
    us.breast_symptoms = ', '.join(checked_symptoms.get('Breast Cancer', []))
    us.skin_symptoms = ', '.join(checked_symptoms.get('Skin Cancer', []))
    us.kidney_symptoms = ', '.join(checked_symptoms.get('Kidney Cancer', []))
    db.session.add(us)
    db.session.commit()

    return render_template('result.html', result=result, health_id=user.health_id)

@app.route('/database')
def show_database():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    users = User.query.all()
    symptoms = Symptom.query.all()
    user_symptoms = UserSymptom.query.all()
    risk_results = RiskResult.query.order_by(RiskResult.date.desc()).all()

    return render_template('database.html',
                           users=users,
                           symptoms=symptoms,
                           user_symptoms=user_symptoms,
                           risk_results=risk_results,
                           prettify_symptom=prettify_symptom)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        populate_symptoms()
    app.run(debug=True)