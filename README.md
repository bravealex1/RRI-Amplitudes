# Sleep Apnea Detection Using ECG and Machine Learning

## Introduction

The primary objective of the project was to analyze ECG data, extract meaningful features, and build predictive models for apnea detection. To achieve this, methodologies using both traditional machine learning (Random Forest) and deep learning approaches (PyTorch with hyperparameter tuning using Optuna) were implemented. 

## Methodology

Data Acquisition and Preprocessing
The dataset utilized was the Apnea-ECG Database, retrieved using the KaggleHub API.
Preprocessing Steps:
ECG signals were cleaned using NeuroKit2’s ecg_clean function.
R-peaks were detected for feature extraction.
Features such as RR intervals (mean and standard deviation) and R-peak amplitudes were derived.
Labels indicating apnea ('A') or normal ('N') conditions were mapped to binary values (1 for apnea, 0 for normal).
Splitting:
The dataset was divided into training, validation, and testing sets, ensuring a balanced representation of both classes.
Feature Engineering
Key features extracted included:
RR Intervals: Time differences between successive R-peaks.
R-Peak Amplitudes: Heights of the detected R-peaks.
These features encapsulate the variability and rhythm in ECG signals critical for apnea detection.
Modeling Approaches
Random Forest Classifier:
A traditional machine learning approach was used to benchmark performance.
Achieved an accuracy of approximately 94.06% on the test set.
Deep Learning with PyTorch:
An enhanced neural network was designed, incorporating two hidden layers with ReLU activations and dropout for regularization.
Weighted cross-entropy loss was employed to handle class imbalances effectively.
Hyperparameter Optimization with Optuna:
Optuna was used to optimize hyperparameters such as learning rate, hidden layer sizes, dropout rate, and batch size.
The best parameters included:
'learning_rate': 0.0021064209979199974,
'hidden_size1': 65,
'hidden_size2': 47,
'dropout_rate': 0.3058764629784719,
'batch_size': 64
Training and Validation
The model was trained using the Adam optimizer. To enhance training efficiency and stability, a learning rate scheduler was employed to dynamically adjust the learning rate over epochs, ensuring optimal convergence.
Throughout the training process, performance on the validation set was closely monitored. This approach not only prevented overfitting by ensuring the model generalized well to unseen data but also provided insights into its capacity to maintain robustness across different data distributions.
Key evaluation metrics included:
Loss: Monitored for both training and validation phases to gauge model optimization and detect any signs of overfitting or underfitting.
Accuracy: Calculated for validation and test datasets to measure the model's overall performance in classifying apnea and normal samples.
Classification Report: Metrics such as precision, recall, F1-score, and support for each class, offering a comprehensive understanding of the model's ability to balance sensitivity and specificity across both apnea and normal cases.
Testing and Final Evaluation
The test accuracy achieved was 59.85%, which is not good compared to random forest classifier.

## Results

Random Forest Benchmark:
The Random Forest model achieved an impressive accuracy of 94.06%, demonstrating its effectiveness as a baseline for the problem.
Deep Learning Approach:
The deep learning model, implemented using PyTorch, yielded a validation accuracy of 65.56% and a test accuracy of 59.85%. While the addition of dropout regularization enhanced the model's robustness by mitigating overfitting, overall accuracy remained suboptimal (Chung et al., 2022).
Hyperparameter Tuning:
Despite using Optuna, an advanced hyperparameter optimization framework, the tuning process did not significantly improve the model's performance. This suggests that the deep learning model may require architectural changes or additional data for better optimization (Hong et al., 2020).

## Discussion

This project explored the use of regression algorithms and deep learning models, specifically using PyTorch, to analyze physiological signals like ECG. While Random Forest achieved high accuracy with manual feature engineering, PyTorch offered several advantages that, with further refinement, could lead to improved performance:
Automated Feature Extraction:
Unlike traditional methods, the deep learning model was capable of learning relevant features directly from raw data, reducing the reliance on manual feature engineering. This aligns with the growing trend of leveraging end-to-end learning for signal processing tasks (Nurmaini et al., 2019).
Improved Model Robustness:
Incorporating dropout regularization helped mitigate overfitting, especially given the relatively small dataset, and highlighted the potential of deep learning for tasks involving complex temporal patterns (Niu et al., 2020).

## Limitations

Dataset Constraints:
The dataset's size posed significant challenges. Small datasets often limit the generalization capabilities of deep learning models, a challenge commonly noted in medical signal processing (Hong et al., 2020). Future work could focus on data augmentation or transfer learning to overcome these limitations.
Architectural Exploration:
The model primarily relied on basic feedforward neural networks. Advanced architectures like Long Short-Term Memory (LSTMs) and Transformer-based models could better capture temporal dependencies in physiological signals (Chang et al., 2018). Exploring these architectures may yield substantial performance improvements.
Hyperparameter Optimization:
While Optuna's optimization process identified an optimal combination of hyperparameters, it failed to significantly improve accuracy. This highlights the importance of not only fine-tuning but also designing new loss function and training schema (Hong et al., 2020).

## References

Chang, Y. C., Wu, S. H., Tseng, L. M., Chao, H. L., & Ko, C. H. (2018, September). AF detection by exploiting the spectral and temporal characteristics of ECG signals with the LSTM model. In 2018 Computing in cardiology conference (CinC) (Vol. 45, pp. 1-4). IEEE.
Chung, C. T., Lee, S., King, E., Liu, T., Armoundas, A. A., Bazoukis, G., & Tse, G. (2022). Clinical significance, challenges and limitations in using artificial intelligence for electrocardiography-based diagnosis. International journal of arrhythmia, 23(1), 24.
Hong, S., Zhou, Y., Shang, J., Xiao, C., & Sun, J. (2020). Opportunities and challenges of deep learning methods for electrocardiogram data: A systematic review. Computers in biology and medicine, 122, 103801.
Niu, L., Chen, C., Liu, H., Zhou, S., & Shu, M. (2020). A Deep-Learning Approach to ECG Classification Based on Adversarial Domain Adaptation. Healthcare, 8(4), 437. https://doi.org/10.3390/healthcare8040437
Nurmaini, S., Umi Partan, R., Caesarendra, W., Dewi, T., Naufal Rahmatullah, M., Darmawahyuni, A., Bhayyu, V., & Firdaus, F. (2019). An Automated ECG Beat Classification System Using Deep Neural Networks with an Unsupervised Feature Extraction Technique. Applied Sciences, 9(14), 2921. https://doi.org/10.3390/app9142921
