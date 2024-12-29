from flask import Flask, render_template, redirect, url_for, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from forms import LoginForm, RegistrationForm
from models import db, User
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:  # Ensure to use hashed password in production
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check your username and password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if the username already exists
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('register.html', form=form)  # Render the registration template again
        
        # Create a new user if the username is unique
        new_user = User(username=form.username.data, password=form.password.data)  # Make sure to hash the password!
        db.session.add(new_user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    common_diseases = {
        'Disease A': 'Low',
        'Disease B': 'Medium',
        'Disease C': 'High'
    }
    return render_template('dashboard.html', common_diseases=common_diseases)

@app.route('/classify', methods=['GET', 'POST'])
@login_required
def classify():
    if request.method == 'POST':
        # Get user input
        age = float(request.form['age'])
        educational_level = request.form['educational_level']
        sex = request.form['sex']
        housing_stability = request.form['housing_stability']
        water_quality = request.form['water_quality']
        air_quality = request.form['air_quality']
        access_to_primary_care = request.form['access_to_primary_care']

        # Prepare input DataFrame for classification
        user_input = pd.DataFrame({
            'Age': [age],
            'Educational Level': [educational_level],
            'Sex': [sex],
            'Housing Stability': [housing_stability],
            'Water Quality': [water_quality],
            'Air Quality': [air_quality],
            'Access to Primary Care': [access_to_primary_care]
        })

        # Load the trained model
        model = load_model()
        if model is None:
            flash('Model could not be loaded. Please check the model file.', 'danger')
            return redirect(url_for('dashboard'))

        # Make predictions
        predictions = model.predict(user_input)
        
        disease, risk_level = predictions[0]
        return render_template('classify.html', result=(disease, risk_level))
    return render_template('classify.html')

@app.route('/analysis')
@login_required
def analysis():
    plot_analysis_graphs()
    return render_template('analysis.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

def load_model():
    model_path = 'model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print("Model file not found.")
        return None

def plot_analysis_graphs():
    data = pd.read_csv('Health.csv')
    plot_variable_contributions(data)
    plot_risk_levels(data)
    plot_correlation_heatmap(data)

def plot_variable_contributions(data):
    variable_counts = data['Disease'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.pie(variable_counts, labels=variable_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Variable Contributions to Disease Causation')
    plt.axis('equal')
    plt.savefig('static/disease_contributions.png')
    plt.close()

def plot_risk_levels(data):
    risk_counts = data['Risk Level'].value_counts()
    colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
    plt.figure(figsize=(8, 5))
    plt.pie(risk_counts, labels=risk_counts.index, colors=[colors[risk] for risk in risk_counts.index], autopct='%1.1f%%', startangle=90)
    plt.title('Classification Risk Levels')
    plt.axis('equal')
    plt.savefig('static/risk_levels.png')
    plt.close()

def plot_correlation_heatmap(data):
    numerical_data = pd.get_dummies(data, drop_first=True)
    plt.figure(figsize=(10, 8))
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Correlation Analysis Heatmap')
    plt.savefig('static/correlation_heatmap.png')
    plt.close()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)