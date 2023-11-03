from flask import Flask, render_template, request, jsonify
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the password strength dataset from a CSV file
df = pd.read_csv(
    'D:/SLIIT/3 Year/Semester 2/IPS/PasswordClassifier-main/PasswordClassifier-main/PasswordClassifier/passwords.csv')
imputer = SimpleImputer(strategy='most_frequent')
df['password'] = imputer.fit_transform(df[['password']]).flatten()  # Flatten the result

# Convert the passwords into a matrix of TF-IDF features
vectorizer = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1, 3))
X_train = vectorizer.fit_transform(df['password'])
y_train = df['strength']

# Train a logistic regression model on the entire dataset
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

strength_labels = {
    1: "low",
    2: "medium",
    3: "high"
}


def is_valid_password(password):
    return len(password) >= 12


def has_special_characters(password):
    return any(c in string.punctuation for c in password)


def has_numbers(password):
    return any(c.isdigit() for c in password)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/password_strength', methods=['POST'])
def password_strength():
    password = request.form['password']

    if len(password) < 8:
        return "Invalid password. Password must be at least 8 characters long."

    # Check if the password contains special characters
    special_characters = has_special_characters(password)

    # Check if the password contains numbers
    numbers = has_numbers(password)

    # Use the model to predict the strength of the password
    X_new = vectorizer.transform([password])
    strength = model.predict(X_new)[0]

    return render_template('passwordstrength.html', password=password,
                           has_special_characters=special_characters,
                           has_numbers=numbers,
                           is_valid=True,  # Password length is already validated
                           strength=strength_labels[strength])


@app.route('/api/password_strength', methods=['POST'])
def api_password_strength():
    data = request.get_json()
    password = data.get('password')

    if len(password) < 8:
        response = {'message': 'Invalid password. Password must be at least 8 characters long.'}
        return jsonify(response), 400

    # Check if the password contains special characters
    special_characters = has_special_characters(password)

    # Check if the password contains numbers
    numbers = has_numbers(password)

    # Use the model to predict the strength of the password
    X_new = vectorizer.transform([password])
    strength = model.predict(X_new)[0]

    response = {
        'password': password,
        'has_special_characters': special_characters,
        'has_numbers': numbers,
        'is_valid': True,
        'strength': strength_labels[strength]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
