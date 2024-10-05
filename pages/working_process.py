import streamlit as st


def display_model_description():
    st.title("working process")

    st.header("Model Architecture")
    st.image("images/models/1dCNN.png", caption="1D-CNN Architecture Diagram")
    code_string = """
    # 1D-Model Definition
    def create_1d_cnn_model(input_shape):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1)  # Output Layer: Print Out Event Start Time
        ])
        return model
    """
    st.code(code_string, language="python")
    
    
# 페이지가 호출될 때 함수 실행
if __name__ == "__main__":
    display_model_description()