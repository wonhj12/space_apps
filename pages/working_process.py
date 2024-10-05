import streamlit as st


def display_sidebar_toc():
    st.markdown("""
    <style>
    /* Style the sidebar */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
    }
    
    /* Style TOC links in the sidebar */
    .toc-link {
        font-size: 20px;
        color: sky-blue;
        text-decoration: none;
        padding: 3px 0;
        display: block;
    }

    .toc-link:hover {
        color: #2c3e50;
        font-weight: bold;
        text-decoration: underline;
    }

    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Contents")
    st.sidebar.markdown("""
    1. <a class="toc-link" href="#data-processing"> Data Processing </a>
    2. <a class="toc-link" href="#model-architecture"> Model Architecture </a>
    """, unsafe_allow_html=True)


def Data_processing():
    st.title("Working Process")

    st.markdown("# Data Processing", unsafe_allow_html=True)

    # 두 개의 열 생성
    col1, col2 = st.columns([2.5, 1])  # 첫 번째 열이 이미지를, 두 번째 열이 텍스트를 담당 (비율 조정 가능)

    # 두 번째 열에 텍스트 배치
    with col1:
        st.write("""
        Our goal is to predict the meaningful data points where seismic events occur. 
        To do this effectively, we decided to use deep learning. 
        \nHowever, deep learning requires a large dataset to produce significant results. 
        When we looked at the Lunar data, we only had 77 samples available, which was not enough to predict event points using deep learning. 
        Even after adding additional data provided by NASA, we only managed to collect around 200 samples. 
        Therefore, we decided to augment the data ourselves.
        """)

    # 첫 번째 열에 이미지 배치
    with col2:
        st.image("images/Lunar.png", width=200)
        
    st.subheader("Slicing With OverLap")

    st.write("1. We loaded the existing CSV file and used the 'slicing' method along with overlap to generate new samples.")

    reading_code = """event_time_rel = label_row['time_rel(sec)'].values[0]"""

    st.code(reading_code,language="python")

    st.markdown("""
    2. Based on the file name, we extract the seismic event time `event_time_rel` and perform slicing starting 
    from `event_index - (time_step // 2)`. First, we decide how many data segments we will divide a single file (one day) 
    into with `time_step = 6000`, and store these segments in an array. Then, we use `event_time_rel` as the reference point and extract a fixed-length data segment centered around this event time.
    """)
    slice_code = """for start_index in range(max(0, event_index - time_step // 2), min(len(velocity) - time_step + 1, event_index + 1)):
                        end_index = start_index + time_step
    """

    st.code(reading_code,language="python")

    st.write("""3. After slicing, we shift the starting point by `overlap_step` (100) and repeatedly generate new samples. 
            Since overlapping points are included in the sampling process, we are able to create more samples from the same data.""")
    slice_code = """ for start_index in range(max(0, event_index - time_step // 2), min(len(velocity) - time_step + 1, event_index + 1), overlap_step):
                        end_index = start_index + time_step
    """

    st.code(reading_code,language="python")
    st.divider()
    col1, col2 = st.columns([2, 1]) 
    with col1:
        st.write("""
        We were able to successfully increase the dataset from 77 samples to around 230,000, completing the preparation process for effective deep learning training.""")

    with col2:
        st.code("""
        X_train, y_train
        (221336, 6000), (221336,)
        X_test, y_test
        (9096, 6000), (9096,)
        """)  


def display_model_description():
    st.markdown("# Model Architecture", unsafe_allow_html=True)
    
    st.subheader("Why 1D-CNN?")
    st.markdown("#### **1) Temporal Pattern Recognition**")
    st.write("""
    1D-CNNs are capable of learning key patterns and features in time series data. 
    By using convolution operations, they capture local correlations over time, making them effective in detecting changes in time series data.
    """)

    st.markdown("#### **2) Efficient Feature Extraction**")
    st.write("""
    1D-CNNs can automatically extract important features from time series data using filters (kernels). 
    This is more efficient than manually designing features and allows the model to learn various patterns.
    """)

    st.markdown("#### **3) Lightweight and Computationally Efficient**")
    st.write("""
    1D-CNNs are lightweight enough to be loaded and executed even using CPU resources, offering both low computational cost and efficiency. 
    In order to deploy the model through GCP, it needed to perform predictions quickly even with CPU resources. 
    Among the models used for time series analysis, this approach proved to be the most effective.
    """)
    
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
    st.image("images/models/1dCNN.png", caption="1D-CNN Architecture Diagram")
    
    st.divider()
    st.subheader("Model Structure")
    st.markdown("#### **Convolution Layer**")
    st.write("""
    - `Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape)`: 
    This 1D convolutional layer uses 64 filters to learn patterns from the input data. 
    The `kernel_size=3` means the layer processes three consecutive data points (representing 3 time intervals in the time series). 
    `input_shape` specifies the shape of the input, which includes the length and the number of channels of the time series data.
    """)

    st.markdown("#### **MaxPooling1D Layer**")
    st.write("""
    - `MaxPooling1D(pool_size=2)`: This max-pooling layer uses a pool size of 2, reducing the size of the input data by half. 
    This helps speed up learning and prevents overfitting.
    - The Conv + MaxPooling layers are repeated 3 times.
    - Activation function: **ReLU**
    """)

    st.markdown("#### **Flatten Layer**")
    st.write("""
    - `Flatten()`: This layer flattens the multidimensional output from the convolution layers into a 1D array to pass to the Dense layers.
    """)

    st.markdown("#### **Dense Layer**")
    st.write("""
    - `Dense(128, activation='relu')`: A fully connected layer with 128 nodes that processes the extracted features from the final convolutional layer 
    and learns higher-level patterns for prediction.
    """)

    st.markdown("#### **Dropout Layer**")
    st.write("""
    - `Dropout(0.5)`: To prevent overfitting, dropout is applied, randomly disabling 50% of the nodes during training.
    """)

    st.markdown("#### **Output Layer**")
    st.write("""
    - `Dense(1)`: The final output layer returns a single value, predicting the event's start time.
    """)


if __name__ == "__main__":
    display_sidebar_toc()
    Data_processing()
    st.divider()
    display_model_description()
