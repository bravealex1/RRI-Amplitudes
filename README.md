# Sleep Apnea Detection Using Random Forest and Deep Learning

## **Introduction**
The primary objective of the project was to analyze ECG data, extract meaningful features, and build predictive models for apnea detection. To achieve this, methodologies using both traditional machine learning (Random Forest) and deep learning approaches (PyTorch with hyperparameter tuning using Optuna) were implemented.

---

## **Methodologies**

### **Data Acquisition and Preprocessing**
- **Dataset:** The dataset utilized was the Apnea-ECG Database, retrieved using the KaggleHub API.
- **Preprocessing Steps:**
  - ECG signals were cleaned using NeuroKit2’s `ecg_clean` function.
  - R-peaks were detected for feature extraction.
  - Features such as RR intervals (mean and standard deviation) and R-peak amplitudes were derived.
  - Labels indicating apnea ('A') or normal ('N') conditions were mapped to binary values (1 for apnea, 0 for normal).
- **Dataset Splitting:**
  - The dataset was divided into training, validation, and testing sets, ensuring a balanced representation of both classes.

### **Feature Engineering**
Key features extracted included:
- **RR Intervals:** Time differences between successive R-peaks.
- **R-Peak Amplitudes:** Heights of the detected R-peaks.
These features encapsulate the variability and rhythm in ECG signals critical for apnea detection.

### **Modeling Approaches**

#### **Random Forest Classifier**
- A traditional machine learning approach was used to benchmark performance.
- Achieved an accuracy of approximately **94.06%** on the test set.

#### **Deep Learning with PyTorch**
- An enhanced neural network was designed, incorporating:
  - Two hidden layers with ReLU activations.
  - Dropout for regularization.
  - Weighted cross-entropy loss to handle class imbalances effectively.

#### **Hyperparameter Optimization with Optuna**
- Optuna was used to optimize hyperparameters, including:
  - **Learning Rate:** 0.0021
  - **Hidden Layer Sizes:** 65 and 47 nodes.
  - **Dropout Rate:** 0.3059
  - **Batch Size:** 64

### **Training and Validation**
- **Optimizer:** Adam optimizer.
- **Learning Rate Scheduler:** Dynamically adjusted the learning rate over epochs for optimal convergence.
- **Performance Metrics:**
  - **Loss:** Monitored for both training and validation phases to detect overfitting or underfitting.
  - **Accuracy:** Calculated for validation and test datasets to measure overall performance.
  - **Classification Report:** Provided precision, recall, F1-score, and support for each class.

---

## **Results**

### **Random Forest Benchmark**
- Achieved an accuracy of **94.06%**, demonstrating its effectiveness as a baseline for the problem.

### **Deep Learning Approach**
- Validation Accuracy: **65.56%**
- Test Accuracy: **59.85%**
- While dropout regularization improved robustness, overall accuracy remained suboptimal ([Chung et al., 2022](https://arrhythmia.biomedcentral.com/articles/10.1186/s42444-022-00075-x)).

### **Hyperparameter Tuning**
- Optuna failed to significantly enhance performance, suggesting a need for architectural changes or additional data ([Hong et al., 2020](https://www.sciencedirect.com/science/article/abs/pii/S0010482520303466)).

---

## **Discussion**

### **Advantages of PyTorch**
1. **Automated Feature Extraction:**
   - Deep learning models learned features directly from raw data, reducing reliance on manual engineering ([Nurmaini et al., 2019](https://www.mdpi.com/2076-3417/9/14/2921)).
2. **Improved Robustness:**
   - Dropout regularization mitigated overfitting and showcased the potential of deep learning for complex temporal patterns ([Niu et al., 2020](https://www.mdpi.com/2227-9032/8/4/437)).

### **Potential Limitations**
1. **Dataset Constraints:**
   - Limited dataset size restricted generalization capabilities. Future work could leverage data augmentation or transfer learning ([Hong et al., 2020](https://www.sciencedirect.com/science/article/abs/pii/S0010482520303466)).
2. **Architectural Exploration:**
   - Models like Long Short-Term Memory (LSTMs) and Transformers could enhance temporal dependency recognition ([Chang et al., 2018](https://ieeexplore.ieee.org/document/8500853)).
3. **Hyperparameter Optimization:**
   - Optuna's optimization highlighted the importance of tuning but failed to significantly improve accuracy without redesigning the training schema ([Hong et al., 2020](https://www.sciencedirect.com/science/article/abs/pii/S0010482520303466)).

---

## **References**
- Chang, Y. C., Wu, S. H., Tseng, L. M., Chao, H. L., & Ko, C. H. (2018). *AF detection by exploiting the spectral and temporal characteristics of ECG signals with the LSTM model*. [IEEE Explore](https://ieeexplore.ieee.org/document/8500853).
- Chung, C. T., et al. (2022). *Clinical significance, challenges, and limitations in using artificial intelligence for electrocardiography-based diagnosis*. [Arrhythmia Journal](https://arrhythmia.biomedcentral.com/articles/10.1186/s42444-022-00075-x).
- Hong, S., et al. (2020). *Deep learning methods for ECG data: A systematic review*. [Computers in Biology and Medicine](https://www.sciencedirect.com/science/article/abs/pii/S0010482520303466).
- Nurmaini, S., et al. (2019). *Automated ECG Beat Classification with Deep Neural Networks*. [MDPI](https://www.mdpi.com/2076-3417/9/14/2921).
- Niu, L., et al. (2020). *Deep-Learning ECG Classification Using Adversarial Domain Adaptation*. [MDPI](https://www.mdpi.com/2227-9032/8/4/437).
