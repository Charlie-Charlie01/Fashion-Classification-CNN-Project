# Fashion Class Classification using CNN

A deep learning project that classifies fashion items using Convolutional Neural Networks (CNN) with 91% accuracy.

## Project Overview

This project implements a CNN-based image classification system to identify different types of fashion items from the Fashion-MNIST dataset. The model successfully distinguishes between various clothing categories with high precision.

## Dataset

- **Source**: Kaggle Fashion-MNIST Dataset
- **Training Images**: 60,000 grayscale images
- **Test Images**: 10,000 grayscale images
- **Image Dimensions**: 28x28 pixels
- **Classes**: 10 fashion categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## Technologies Used

- **Python**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Model evaluation metrics

## Project Workflow

### 1. Data Loading and Exploration
- Imported training and testing datasets from Kaggle
- Performed initial data exploration using:
  - `.head()` - View first few samples
  - `.tail()` - View last few samples
  - `.shape` - Understand data dimensions

### 2. Data Preprocessing
- **Image Reshaping**: Converted images to 28×28×1 format (grayscale)
- **Normalization**: Scaled pixel values to range [0,1]

### 3. Exploratory Data Analysis
- Created comprehensive visualizations using **Matplotlib** and **Seaborn**
- Displayed sample images from each category
- Generated statistical insights about the dataset

### 4. CNN Architecture Design

The model consists of the following layers:

#### Feature Extraction Layers:
- **Convolutional Layers**: Feature detectors with various filter sizes
- **ReLU Activation**: Non-linear activation function
- **MaxPooling Layers**: Spatial dimension reduction and feature selection
- **Dropout Layers**: Regularization to prevent overfitting

#### Classification Layers:
- **Flattening Layer**: Convert 2D feature maps to 1D vector
- **Dense Layers**: Fully connected layers for final classification
- **Output Layer**: 10 neurons with softmax activation

### 5. Model Training
- Compiled model with appropriate optimizer and loss function
- Trained on 60,000 fashion images
- Monitored training progress with validation metrics

### 6. Model Evaluation
- **Test Accuracy**: 91% (improved from initial 90%)
- **Confusion Matrix**: Detailed performance analysis across all classes
- **Classification Report**: Precision, recall, and F1-score for each category

### 7. Model Optimization
- **Dropout Integration**: Added dropout layers to combat overfitting
- **Performance Improvement**: Accuracy increased from 90% to 91%
- **Hyperparameter Tuning**: Optimized learning rates and batch sizes

## Results

| Metric | Initial Model | Optimized Model |
|--------|---------------|-----------------|
| Test Accuracy | 90% | **91%** |
| Training Time | Standard | Slightly Increased |
| Overfitting | Moderate | **Reduced** |

## 🚀 Key Achievements

- Successfully implemented CNN architecture from scratch
- Achieved 91% classification accuracy
- Improved model performance through dropout regularization
- Comprehensive evaluation with confusion matrix and classification reports
- Effective data visualization and analysis

## Future Improvements

- Implement data augmentation techniques
- Experiment with transfer learning approaches
- Try advanced architectures (ResNet, EfficientNet)
- Deploy model as a web application
- Add real-time image classification capabilities

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Author**: Ojo Gbenga Charles  
**Date**: June 2025  
**Contact**: gbe01nga@gmail.com
