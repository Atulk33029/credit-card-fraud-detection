from predict import FraudPredictor

p = FraudPredictor(model_path='../models/model.json', config_path='../models/config.json')

v_features = {f'V{i}': 0.0 for i in range(1, 29)}
result = p.predict(v_features, amount=50.0, hour=14)
print(result)