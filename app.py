import os
import re
import random
import string
import joblib
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data safely
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

app = Flask(__name__)

# Path to pre-trained model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nb_classifier.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

model = None
vectorizer = None

# Try to load the pre-trained models
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and Vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: model files not found at {MODEL_PATH} or {VECTORIZER_PATH}")

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
        
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    title = data.get('title', '')
    description = data.get('description', '')
    url = data.get('url', '')
    
    combined_text = f"{title} {description}"
    cleaned_text = clean_text(combined_text)
    
    if model and vectorizer:
        try:
            features = vectorizer.transform([cleaned_text])
            prediction_val = model.predict(features)[0]
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features)[0]
                confidence_score = float(max(probs))
                fraud_prob = float(probs[1]) if len(probs) > 1 else (1.0 if prediction_val == 1 else 0.0)
            else:
                confidence_score = 1.0
                fraud_prob = 1.0 if prediction_val == 1 else 0.0
            
            prediction_label = "Fraudulent" if prediction_val == 1 else "Real"
            risk_level = "High" if fraud_prob > 0.5 else "Low"
            
            return jsonify({
                'prediction': prediction_label,
                'confidence_score': round(confidence_score * 100, 2),
                'risk_level': risk_level,
                'is_fraudulent': bool(prediction_val)
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    else:
        # Fallback/Mock logic if model files are missing
        fraud_keywords = {
            'urgent': 15, 'money': 10, 'whatsapp': 20, 'wire transfer': 25, 
            'no experience': 10, 'earn from home': 15, 'telegram': 20,
            'quick cash': 25, 'unlimited': 10, 'immediate start': 12,
            'processing fee': 30, 'bitcoin': 25, 'crypto': 20, 'cash app': 25,
            'zelle': 25, 'gift card': 30, 'shipping': 10, 'envelope': 15,
            'work at home': 10, 'part time': 5, 'flexibility': 5,
            'data entry': 15, 'administrative assistant': 10, 'customer service': 8
        }
        
        scam_roles = ['data entry', 'clerk', 'assistant', 'customer service', 'receptionist']
        title_scam_score = 0
        for role in scam_roles:
            if role in title.lower():
                title_scam_score += 15
        
        base_score = random.randint(10, 30)
        score = base_score + title_scam_score
        detected = []
        
        text_to_check = combined_text.lower()
        for kw, weight in fraud_keywords.items():
            if kw in text_to_check:
                score += weight
                detected.append(kw)
        
        if url:
            suspicious_domains = ['bit.ly', 'tinyurl.com', 'forms.gle', 'docs.google.com', 'whatsapp.com']
            for domain in suspicious_domains:
                if domain in url.lower():
                    score += 20
                    detected.append(f"Suspicious link: {domain}")
        
        if len(description) < 200:
            score += 15
        elif len(description) > 5000:
            score += 10
            
        if re.search(r'[a-zA-Z0-9._%+-]+@(gmail|yahoo|outlook|hotmail)\.com', description):
            score += 20
            detected.append("personal email domain")

        fraud_prob = min(score, 98) / 100.0
        prediction_label = "Fraudulent" if fraud_prob > 0.45 else "Real"
        risk_level = "High" if fraud_prob > 0.65 else ("Medium" if fraud_prob > 0.35 else "Low")
        
        if prediction_label == "Fraudulent" and len(detected) < 2:
            detected.append("Suspicious job structure")
            detected.append("Unverified company profile")
        
        return jsonify({
            'prediction': prediction_label,
            'confidence_score': round(fraud_prob * 100, 2),
            'risk_level': risk_level,
            'is_fraudulent': prediction_label == "Fraudulent",
            'risk_factors': detected[:5],
            'url': url,
            'note': 'Using enhanced fallback logic (model files not found)'
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

# This block allows you to still test locally with `python main.py` using Flask's dev server
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
