# NASA Space Apps Challenge - Team 구구덕
[Streamlit Dashboard](http://guguduck.earth/)

## 1. Background
### Main Challenges to Solve
- **High Cost** : Transmitting the vast amounts of data collected by spacecraft from celestial bodies back to Earth is very costly, as it requires a significant amount of power to send the data, which is greatly affected by distance.
- **Signal Delay** : The distance between Earth and celestial bodies causes signal delays. Signal delays can lead to interference, increasing the likelihood of data loss or errors during transmission.

----
## 2. Working Process
### Data Source
[Seismic Detection Across the Solar System](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/?tab=resources)

### Data Processing
- **Slicing with overlaps** : To predict seismic events using deep learning, we initially had a small dataset of 77 samples, which increased to about 200 after adding IRIS data. This was still insufficient, so we augmented the data by slicing the seismic data into 6000-step intervals and overlapping points. Shifting the starting point by 100 steps generated multiple samples, ensuring enough data for training.
  
### Model Architecture
#### Why 1D-CNN?
- **Lightweight & efficient** : 1D-CNNs are computationally efficient and ideal for reducing energy costs while enabling fast predictions with minimal CPU resources, making them effective for detecting seismic event times.
- **Time series pattern recognition** : 1D-CNNs learn key patterns in time series data by capturing local correlations through convolution operations, making them effective at detecting changes over time.
- **Automatic feature extraction** : 1D-CNNs extract important features using filters, offering a more efficient approach than manual feature design, allowing the model to learn various patterns automatically.


### How To Use CNN
#### STEP 1) Binary Classification to Find the Section Containing the Event Point

- The task is to classify whether a 6000-step section contains an event point (binary classification).
A 1D-CNN processes the sequences to learn spatial features, using ReLU activations in the convolution and pooling layers. The output passes through a fully connected layer for binary classification. CrossEntropyLoss and the Adam optimizer are used for training.

<img width="1463" alt="image" src="https://github.com/user-attachments/assets/1a52f7eb-3843-4666-a7d2-7376270894e0">

![9c7f1391cda52e52ce10c3a69b0dc07308cb25b484705a2e89eac08f](https://github.com/user-attachments/assets/db5ccc30-761d-4402-bdcb-00383bb11921)


#### STEP 2) Regression to Find the Event Point
- After detecting sections with event points, the next step is to locate the exact event point within the section. A CNN is used to learn patterns and predict the probability of the event at each point, identifying the one with the highest probability. The model shows its strong capability to accurately pinpoint the event's location.
![9a4ce20021e2ed99294b82e42361e307d628fc02a1b0f8ddac1132fd](https://github.com/user-attachments/assets/ebcdc8fe-e4b2-4599-9944-610b85ba2952)

```
# 1D-CNN Model Definition
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
    return mode
```

#### Model Performance

| Model       | Metric              | Value     |
|-------------|---------------------|-----------|
| Classifier  | Validation Loss      | 0.0936    |
|             | Validation Accuracy  | 0.9782    |
| Regressor  | Validation Loss      | 7391.5918 |
|             | MSE                  | 2407.4464 |

---
## 3. Tech Spec

| Category        | Details                    |
|-----------------|----------------------------|
| Main Language   | Python                     |
| Dashboard       | Streamlit                  |
| Model           | PyTorch                    |
| Model Training  | KT Cloud, Jupyter Notebook |
| Server & DNS    | GCP VM Machine, Porkbun    |
| Collaboration   | Notion, GitHub             |


---
## 4. Installation & Execution
### Server
[Streamlit Dashboard](http://guguduck.earth/)

### How to Test Locally
```
# Execute virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run dashboard
streamlit run predict.py
```


--- 
## 5. Our Team
### Team Member
<table>
  <tbody>
    <tr>
      <td align="center"><img src="https://github.com/user-attachments/assets/054db3b9-01cc-4593-af45-a80c0c3f142c" width="100px;" alt=""/><br /><sub><b>Lead Scientist : 원하진</b></sub><br /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/147d16dc-6ff6-4810-bf60-ed9c2460aca9" width="100px;" alt=""/><br /><sub><b>Software Engineer : 양은서</b></sub><br /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/61900f80-f3ce-4ffb-b48e-b5a9a6a96029" width="100px;" alt=""/><br /><sub><b>Software Engineer : 최다영</b></sub><br /></td>
     <tr/>
      <td align="center"><img src="https://github.com/user-attachments/assets/0114a8c7-55a0-4203-84a1-983b6090fb60" width="100px;" alt=""/><br /><sub><b> Software Engineer : 김태관</b></sub><br /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/fa27ffea-702c-4c36-9767-e7bf8d04caf2" width="100px;" alt=""/><br /><sub><b>Software Engineer : 김태우</b></sub><br /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/d4aec8e2-d28e-4e6a-855d-cd77eb9cc8eb" width="100px;" alt=""/><br /><sub><b>Software Engineer : 이원준</b></sub><br /></td>
    </tr>
  </tbody>
</table>


