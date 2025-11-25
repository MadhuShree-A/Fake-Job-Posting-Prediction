import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class JobPostingPreprocessor:
    def __init__(self):
        self.text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        self.cat_cols = ['employment_type', 'required_experience', 'required_education']
        self.binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == '':
            return 'emptytext'
            
        # Remove HTML
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Remove special characters and brackets
        text = re.sub(r'\[[^]]*\]', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = [word for word in text.split() if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)
    
    def extract_features(self, job_data):
        """Extract features from job posting data"""
        features = {}
        
        # Text-based features
        description = job_data.get('description', '')
        title = job_data.get('title', '')
        company_profile = job_data.get('company_profile', '')
        
        # Basic text features
        features['char_count'] = len(description)
        features['word_count'] = len(description.split())
        features['title_length'] = len(title)
        features['has_company_profile'] = 1 if company_profile.strip() else 0
        
        # Fraud indicators
        features['uppercase_ratio'] = sum(1 for c in description if c.isupper()) / len(description) if description else 0
        features['num_exclamations'] = description.count('!')
        features['num_questions'] = description.count('?')
        features['has_email'] = 1 if '@' in description else 0
        features['has_url'] = 1 if 'http' in description.lower() or 'www' in description.lower() else 0
        
        # Job type features
        employment_type = job_data.get('employment_type', 'Unknown')
        features['is_contract'] = 1 if 'contract' in str(employment_type).lower() else 0
        features['is_full_time'] = 1 if 'full' in str(employment_type).lower() else 0
        
        # Company features
        features['telecommuting'] = job_data.get('telecommuting', 0)
        features['has_company_logo'] = job_data.get('has_company_logo', 0)
        features['has_questions'] = job_data.get('has_questions', 0)
        
        # Experience level
        experience = job_data.get('required_experience', 'Unknown')
        features['requires_experience'] = 0 if 'not' in str(experience).lower() or 'un' in str(experience).lower() else 1
        
        return features
    
    def prepare_for_prediction(self, job_data):
        """Prepare job data for model prediction"""
        # Extract features
        features = self.extract_features(job_data)
        
        # Convert to array (simplified - in real app, use your actual feature transformation)
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        return feature_array, features