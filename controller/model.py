import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 모델 불러오기
def load_model():
    try:
        # 모델 파일이 존재하는지 확인
        if not os.path.exists('./model.h5'):
            raise FileNotFoundError('Model does not exist')
        
        # 모델 로드
        model = tf.keras.models.load_model(
            './model.h5', 
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
    
        return model

    except Exception as e:
        print(f'Error loading model: {e}')
        return None

# 데이터 예측
def predict(result):
    # 데이터 scaling
    scaler = StandardScaler()
    scaler.fit(result)
    test_scaled = scaler.transform(result.reshape(-1, result.shape[-1])).reshape(result.shape)

    # 모델 로드
    model = load_model()

    # 모델 있으면 예측 진행
    # 모델 없으면 None 반환
    if model:
        predictions = model.predict(test_scaled)
        return predictions.flatten()
    else:
        print('Model could not be loaded.')
        return None

    