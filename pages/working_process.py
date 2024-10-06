import streamlit as st

# st.image("https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg",width=300)

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
        text-decoration: none;
        padding: 3px 0;
        display: block;
    }

    .toc-link:hover {
        color: #8b5543;
        font-weight: bold;
        text-decoration: underline;
    }

    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Contents")
    st.sidebar.markdown("""
    1. <a class="toc-link" href="#data-processing"> Data Processing </a>
    2. <a class="toc-link" href="#model-architecture"> Model Architecture </a>
    3. <a class="toc-link" href="#how-to-use-cnn"> How To Use CNN </a>
    """, unsafe_allow_html=True)


def Data_processing():
    # st.title("Working Process")
    # st.markdown("# Data Processing", unsafe_allow_html=True)
    # st.markdown("""
    # <div>
    #     <h1 style="color : #d38856; text-align: center">
    #         Working Process
    #     </h1>
    # </div>""", unsafe_allow_html=True)
    # st.write('###')
    st.markdown("""
    <div>
        <h1 style="color : #d38856">
            Data Processing
        </h1>
    </div>""", unsafe_allow_html=True)


    # 두 개의 열 생성
    col1, col2 = st.columns([2.5, 1])  # 첫 번째 열이 이미지를, 두 번째 열이 텍스트를 담당 (비율 조정 가능)

    # 두 번째 열에 텍스트 배치
    with col1:
        st.write("""
        Our goal is to predict meaningful data points where seismic events occur. 
        To efficiently achieve the goal, we decided to use deep learning. 
        \nHowever, deep learning requires large dataset to produce significant results. 
        Unfortunately, we only had 77 samples available, which was not sufficient enough for meaningful learning. 
        Even with additional data provided by IRIS, we only managed to collect around 200 samples.
        Therefore, we decided to augment the data ourselves.
        """)

    # 첫 번째 열에 이미지 배치
    with col2:
        st.image("images/astronaut.png", width=200, caption="Astronaut in moon")
        
    # Slicing with overlap
    st.markdown("""
    <div>
        <h2 style="color : #d38856">
            Slicing data with overlaps
        </h2>
    </div>""", unsafe_allow_html=True)

    st.write("""
    01. We loaded the data from MSEED file to generate new training samples. 
    We sliced and overlapped the points to maximize the accuracy.
    """)
    reading_code = """event_time_rel = label_row['time_rel(sec)'].values[0]"""
    st.code(reading_code,language="python")

    st.markdown("""
    02. We extracted the seismic event time `event_time_rel` from the catalog and began slicing the data starting from `event_index - (time_step // 2)`. 
    We decided to slice the data into intervals of 6000 steps, so we've set `time_step = 6000` to slice and save the data into an array. 
    Then, we used `event_time_rel` as the reference point to extract several points from the sliced data.
    """)
    slice_code = """
    for start_index in range(max(0, event_index - time_step // 2), min(len(velocity) - time_step + 1, event_index + 1)):
        end_index = start_index + time_step
    """
    st.code(slice_code,language="python")

    st.write("""
    03. After the first slice, we shifted the starting point by factor of `overlap_step = 100` and repeatedly generated new samples. 
    Since we overlapped the data points while slicing, we were able to create enough samples""")
    overlap_code = """
    for start_index in range(max(0, event_index - time_step // 2), min(len(velocity) - time_step + 1, event_index + 1), overlap_step):
        end_index = start_index + time_step
    """
    st.code(overlap_code,language="python")

    st.markdown("###", unsafe_allow_html=True)

    st.write("""
    We were able to successfully increase the dataset from 77 samples to around 230,000, 
    completing the preparation process for effective deep learning training.""")

    st.code("""
    X_train, y_train = (221336, 6000), (221336,)
    X_test, y_test = (9096, 6000), (9096,)
    """)  


def display_model_description():
    st.markdown("""
    <div>
        <h1 style="color : #d38856">
            Model Architecture
        </h1>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div>
        <h2 style="color : #d38856">
            Why 1D-CNN?
        </h2>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 1. Lightweight and computationally efficient")
    st.write("""
    1D-CNNs are lightweight and requires small CPU resource, offering both low computational cost and high efficiency. 
    The purpose of this challenge was to detect event times in order to reduce the energy cost of sending the data, so we decided to use the lightest model that we can find.
    In addition, to deploy the model through Google Cloud Platform we had to perform predictions quickly with CPU resources. 
    Among the models used for time series analysis, 1D-CNN approach proved to be the most effective.
    """)
    
    st.markdown('### 2. Temporal pattern recognition')
    
    st.write("""
    1D-CNNs are capable of learning key patterns and features in time series data. 
    By using convolution operations, it is possible to capture local correlations over time, 
    making it effective to detect changes in time series data.
    """)

    st.markdown("### 3. Efficient feature extraction")
    st.write("""
    1D-CNNs can automatically extract important features from time series data through filters (kernels). 
    This is more efficient than manually designing features and allows the model to learn various patterns.
    """)

    st.markdown("###", unsafe_allow_html=True)
    
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
    
    # Model structure
    st.markdown("""
    <div>
        <h2 style="color : #d38856">
            Model structure
        </h2>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("### Convolution layer")

    st.write("""
    - `Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape)`: 
    This 1D convolutional layer uses 64 filters to learn patterns from the input data. 
    The `kernel_size=3` shows that the layer processes three consecutive data points (representing 3 time intervals in the time series). 
    `input_shape` specifies the shape of the input, which includes the length and the number of channels of the time series data.
    """)

    st.markdown("### MaxPooling1D layer")
    st.write("""
    - `MaxPooling1D(pool_size=2)`: This max-pooling layer uses a pool size of 2, reducing the size of the input data by half. 
    This helps the model to speed up learning and prevents overfitting.
    - The Conv + MaxPooling layers are repeated 3 times.
    - Activation function: **ReLU**
    """)

    st.markdown("### Flatten layer")
    st.write("""
    - `Flatten()`: This layer flattens the multidimensional output from the convolution layers into a 1D array to passes to the Dense layers.
    """)

    st.markdown("### Dense layer")
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
    
def model_working_process():
    #how-to-use-cnn
    st.markdown("""
    <div>
        <h1 style="color : #d38856">
            How To Use CNN?
        </h1>
    </div>""", unsafe_allow_html=True)
    
    st.write("""
    A day’s worth of data (one day) consists of 570,000 sequence data points. However, since it was too large to train the model with all the data at once, we decided to first divide the data into 6000 sections and determine whether each section contains an event point. Afterward, the exact event point was identified in the sections predicted to have an event, thus achieving the final goal.
    """)
    
    st.markdown("""
    <div>
        <h2 style="color : #d38856">
            1. Classification
        </h2>
    </div>""", unsafe_allow_html=True)
    
    st.write("""
    This task involves classifying which of the 6000 sections contains an event point. It is a binary classification problem, and the goal is to predict whether a given section includes an event point.
    """)
    
    st.write("""
    - A 1D-CNN (1D Convolutional Neural Network) is used to learn the patterns in each section.
    Each section, as input, contains multiple sequences of data, and the 1D-CNN processes these sequences to extract important spatial features. 
    """)
    
    st.write("""
    - ReLU activation functions are used in each convolutional (conv) and pooling layer, adding non-linearity in the intermediate layers to learn complex patterns.
    The output of the CNN is passed through a fully connected layer and transformed into a binary classification problem. The final output layer produces the result without an activation function. 
    """)
    
    st.write("""
    - During the training process, the CrossEntropyLoss function is used to minimize the difference between the model’s predictions and the actual values, and the Adam optimizer is used to efficiently update the weights.
    """)
    
    st.write("""
    Ultimately, the model achieves **a Validation Loss of 0.0936** and **a Validation Accuracy of 0.9782**, demonstrating a very high accuracy in classifying whether or not a section contains an event point.
    """)

    st.markdown("""
    <div>
        <h3 style="color : #d38856">
            Post-Processing
        </h3>
    </div>""", unsafe_allow_html=True)

    st.write("""
    When we checked CNN's output, we found that it worked well for some data, but not for others, especially when the boundaries between sections were ambiguously spanned by event points, or when the data classified event points as being in too many sections. We developed algorithms to improve this.
    """)

    st.write("""
    The first is to predict the sections as before, and then predict them again by shifting the starting position by half the size of the section. This way, even if the event point spans an ambiguous location, the event point can still be found because it has been trained a second time by shifting half a space. If the range of the original section and the range of the section shifted by half a space overlapped, we chose the section that was further ahead because it was more likely to have the start of the seismic wave earlier.
    """)

    st.write("""
    The second method was to extract the largest velocity value from each section up to two spaces behind the section size, designate it as the max velocity for that section, and determine that only the top three sections with the largest velocity values contained event points. We used this method because when we analyzed the data, we found that most of the relatively large velocity values occurred after the start of the seismic waves, so we believed that the sections with the largest velocities were more likely to contain event points.
    """)

    st.write("""
    This is what the results looked like when using these algorithms:
    """)

    st.image("images/classification_1.png")
    st.write("before")
    st.image("images/classification_2.png")
    st.write("after")
    
    st.markdown("""
    <div>
        <h1 style="color : #d38856">
            2. Find the event point
        </h1>
    </div>""", unsafe_allow_html=True)

    
    st.write("""
    After predicting which sections contain an event point in the previous step, the next task is to identify the exact point where the event occurs within the predicted section.""")
    
    st.write("""
    - In this step, the predicted event sections are input into the CNN model to further refine the detection of the event point.
    """)
    
    st.write("""
    - The CNN learns patterns related to the event within the section and predicts the probability of the event occurring at each point. It then identifies the point with the highest probability.
    """)
    
    st.write("""
    The final results show **val_loss of 7391.5918** and **Val_Accuracy of 8.8421e-04**, demonstrating the model’s strong performance in accurately identifying the event point. This indicates that the model can successfully detect the precise location of the event point within the section where the event occurs.
    """)
    
    st.caption("""Accuracy Function Definition : Consider a prediction accurate if the difference between the predicted and actual value is less than or equal to 0.1""")
    
if __name__ == "__main__":
    display_sidebar_toc()
    Data_processing()
    st.divider()
    display_model_description()
    st.divider()
    model_working_process()
