import pickle
import numpy as np
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self):
        self.models = {}
        self.feature_info = None
        self.loaded = False
    
    def load_models(self):
        """Load all pre-trained models"""
        try:
            # Load individual models
            model_files = {
                'lgb': 'models/best_lgb_model.pkl',
                'xgb': 'models/xgb_fraud_model.pkl', 
                'rf': 'models/rf_fraud_model.pkl',
                'svm': 'models/optimized_svm_fraud_model.pkl',
                'perceptron': 'models/manual_perceptron_model.pkl'
            }
            
            for name, path in model_files.items():
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"✅ Loaded {name}")
            
            # Load ensemble models
            with open('models/ensemble_models.pkl', 'rb') as f:
                ensemble_data = pickle.load(f)
                self.models.update(ensemble_data)
            
            # Load configuration and feature info
            with open('models/ensemble_config.pkl', 'rb') as f:
                self.config = pickle.load(f)
                
            with open('models/feature_names.pkl', 'rb') as f:
                self.feature_info = pickle.load(f)
                
            # Load preprocessors
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
                
            with open('models/feature_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open('models/feature_selector.pkl', 'rb') as f:
                self.selector = pickle.load(f)
                
            self.loaded = True
            print("🎉 All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.loaded = False
    
    def predict_single(self, model_type='ensemble', features=None):
        """Make prediction for a single job posting"""
        if not self.loaded:
            self.load_models()
            
        # For demo - in real app, you'd preprocess the features first
        # This is a simplified version
        if features is None:
            # Generate random features for demo
            features = np.random.rand(1, 100)
        
        if model_type == 'lgb':
            model = self.models.get('lgb', self.models.get('best_lgb_model'))
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[:, 1]
            else:
                proba = model.predict(features)
        elif model_type == 'xgb':
            model = self.models['xgb']
            proba = model.predict_proba(features)[:, 1]
        elif model_type == 'ensemble':
            # Use ensemble prediction
            lgb_proba = self.models['lgb'].predict_proba(features)[:, 1]
            xgb_proba = self.models['xgb'].predict_proba(features)[:, 1]
            rf_proba = self.models['rf'].predict_proba(features)[:, 1]
            
            weights = self.config.get('weights', [0.4, 0.4, 0.2])
            proba = (weights[0] * lgb_proba + 
                    weights[1] * xgb_proba + 
                    weights[2] * rf_proba)
        elif model_type == 'manual_perceptron':
            model = self.models['perceptron']
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
            else:
                proba = model.predict(features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        prediction = (proba > 0.5).astype(int)
        
        return {
            'prediction': prediction[0],
            'probability': float(proba[0]),
            'risk_level': self.get_risk_level(proba[0]),
            'model_type': model_type
        }
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def get_model_info(self):
        """Get information about available models"""
        return {
            'inbuilt_models': ['LightGBM', 'XGBoost', 'Random Forest', 'Ensemble', 'SVM'],
            'manual_models': ['Manual Perceptron'],
            'ensemble_weights': self.config.get('weights', [0.4, 0.4, 0.2]),
            'optimal_threshold': self.config.get('optimal_threshold', 0.5)
        }