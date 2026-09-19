import json
import numpy as np
import xgboost as xgb

class FraudPredictor:
    def __init__(self, model_path='models/model.json', config_path='models/config.json'):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        with open(config_path) as f:
            self.config = json.load(f)

        self.amount_mean = self.config['amount_mean']
        self.amount_scale = self.config['amount_scale']
        self.threshold = self.config['threshold']
        self.feature_order = self.config['feature_order']

    def preprocess(self, v_features: dict, amount: float, hour: float):
        amount_scaled = (amount - self.amount_mean) / self.amount_scale
        row = {**v_features, 'Hour': hour, 'Amount_scaled': amount_scaled}
        try:
            x = np.array([[row[f] for f in self.feature_order]])
        except KeyError as e:
            raise ValueError(f"Missing required feature: {e}")
        return x

    def predict(self, v_features: dict, amount: float, hour: float):
        x = self.preprocess(v_features, amount, hour)
        proba = self.model.predict_proba(x)[0, 1]
        is_fraud = bool(proba >= self.threshold)
        return {
            'fraud_probability': float(proba),
            'is_fraud': is_fraud,
            'threshold': self.threshold
        }