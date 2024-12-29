from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Length

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=150)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=150)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Register')

class ClassificationForm(FlaskForm):
    age = StringField('Age', validators=[DataRequired()])
    educational_level = SelectField('Educational Level', choices=[('Not Applicable', 'Not Applicable'), ('Primary', 'Primary'), ('Secondary', 'Secondary'), ('Tertiary', 'Tertiary')])
    sex = SelectField('Sex', choices=[('Male', 'Male'), ('Female', 'Female')])
    housing_stability = SelectField('Housing Stability', choices=[('Stable', 'Stable'), ('Unstable', 'Unstable')])
    water_quality = SelectField('Water Quality', choices=[('Poor', 'Poor'), ('Fair', 'Fair'), ('Good', 'Good')])
    air_quality = SelectField('Air Quality', choices=[('Poor', 'Poor'), ('Fair', 'Fair'), ('Good', 'Good')])
    access_to_primary_care = SelectField('Access to Primary Care', choices=[('Yes', 'Yes'), ('No', 'No')])
    submit = SubmitField('Classify')