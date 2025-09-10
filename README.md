# Medical Diagnosis ML Project

## Dual Machine Learning Models for Disease Risk Assessment

A comprehensive machine learning web application that predicts disease risk based on patient medical parameters using Logistic Regression and Random Forest algorithms. This project provides an interactive Streamlit dashboard for real-time disease risk assessment with confidence scores, enabling healthcare professionals and researchers to compare different ML approaches for medical diagnosis.

## Results
![Results](https://github.com/Torajabu/medical-diagnosis-ML-project/blob/main/2025-06-2816-11-56-ezgif.com-video-to-gif-converter.gif)

## Data Flow Diagram and System Architecture 
![DFD](https://github.com/Torajabu/medical-diagnosis-ML-project/blob/main/arch.svg)

## Requirements

- Python 3.8+
- Streamlit - Web application framework
- Scikit-learn - Machine learning models and evaluation
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing operations
- Imbalanced-learn - SMOTE for class balancing
- Joblib - Model persistence and serialization

## Installation

```bash
git clone https://github.com/Torajabu/medical-diagnosis-ML-project.git
cd medical-diagnosis-ML-project
pip install streamlit pandas scikit-learn imbalanced-learn joblib numpy
```

## Quick Start

1. Clone this repository or download the script.
2. Install all required dependencies using the pip command above.
3. Generate the sample medical dataset:

```python
python3 create_data.py
```

4. Train the machine learning models:

```python
python3 train_models.py
```

5. Execute the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The web application will automatically open in your browser at http://localhost:8501, providing an interactive interface for disease risk prediction.

## Output

The program generates:
- **Real-time Risk Predictions** with percentage confidence scores for both Logistic Regression and Random Forest models
- **Interactive Web Dashboard** with slider controls for medical parameter input
- **Model Comparison Interface** allowing users to switch between different ML algorithms
- **Patient Data Summary** displaying all input parameters for verification
- **Confidence Scoring System** providing prediction probability for informed decision making

## How It Works

1. **Data Generation**: Creates synthetic medical records with realistic patient parameters including age, gender, blood pressure, cholesterol, and sugar levels
2. **Data Preprocessing**: Applies SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance in the dataset
3. **Model Training**: Trains both Logistic Regression and Random Forest classifiers using stratified train-test split (80/20)
4. **Model Persistence**: Saves trained models as pickle files for rapid loading during prediction
5. **Web Interface**: Provides Streamlit-based dashboard for real-time interaction and prediction visualization
6. **Risk Assessment**: Delivers instant disease risk predictions with confidence scores for clinical decision support

## Algorithm Theory Behind the Implementation

The project implements two complementary machine learning approaches for medical diagnosis prediction:

**Logistic Regression**: Uses sigmoid function to model probability of binary outcomes
```
P(y=1|x) = 1 / (1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))
```

**Random Forest**: Ensemble method combining multiple decision trees for robust predictions

**Key Concepts:**
- **SMOTE Balancing**: Generates synthetic minority samples to prevent model bias toward majority class
- **Stratified Splitting**: Maintains class distribution proportions in training and testing sets
- **Cross-Validation**: Ensures model generalization and prevents overfitting through multiple validation rounds

## Performance Metrics

- **Processing Time**: Sub-second prediction response for real-time clinical use
- **Model Accuracy**: Both models achieve balanced performance on synthetic medical data
- **Success Rate**: High reliability in distinguishing between risk categories
- **Memory Efficiency**: Lightweight models suitable for deployment in resource-constrained environments

## Usage Tips

- Input realistic medical parameter values within specified ranges for accurate predictions
- Compare results between both models to understand prediction consistency
- Use confidence scores to gauge prediction reliability before clinical interpretation
- Review patient data summary to verify input accuracy before diagnosis

## Troubleshooting

- **ModuleNotFoundError**: Install missing packages using pip install command with all required dependencies
- **File not found errors**: Ensure all project files are in the correct directory structure
- **Model loading errors**: Run train_models.py first to generate required pickle model files
- **Streamlit command not found**: Add ~/.local/bin to your system PATH environment variable

## File Structure

```
medical-diagnosis-ML-project/
├── README.md
├── create_data.py        # Generate synthetic medical dataset
├── train_models.py       # Train and save ML models
├── streamlit_app.py      # Main web application interface
├── medical_records.csv   # Generated patient dataset
├── lr_model.pkl         # Trained Logistic Regression model
└── rf_model.pkl         # Trained Random Forest model
```

## Educational Value

This project demonstrates:
- **Machine Learning Pipeline**: Complete workflow from data generation to model deployment and web interface
- **Healthcare AI Applications**: Practical implementation of ML algorithms in medical diagnosis scenarios
- **Model Comparison Methodology**: Side-by-side evaluation of different algorithms for same prediction task
- **Web Application Development**: Integration of ML models with interactive user interfaces using Streamlit
- **Data Balancing Techniques**: Implementation of SMOTE for addressing class imbalance in medical datasets

## Applications

- **Medical Education**: Training healthcare professionals in AI-assisted diagnosis interpretation
- **Research Prototyping**: Foundation for developing more sophisticated medical prediction systems
- **Clinical Decision Support**: Supplementary tool for risk assessment in healthcare settings
- **Algorithm Benchmarking**: Framework for comparing different ML approaches in healthcare contexts

## Post Mortem Notes

### What Worked Well
- **Streamlit Integration**: The web framework provided excellent rapid prototyping capabilities with minimal frontend development overhead
- **Model Comparison Framework**: Side-by-side algorithm comparison offers valuable insights into prediction consistency and reliability
- **SMOTE Implementation**: Effective handling of class imbalance resulted in more robust and fair model predictions across risk categories

### Challenges Encountered
- **Synthetic Data Limitations**: Generated medical records may not capture complex real-world medical correlations and patient variations
- **Model Interpretability**: Random Forest predictions lack the transparency of Logistic Regression for clinical decision explanation
- **Scalability Constraints**: Current architecture requires model retraining for new medical parameters or expanded feature sets

### Lessons Learned
- **User Interface Simplicity**: Medical applications benefit from intuitive interfaces that minimize complexity for healthcare professionals
- **Model Validation Importance**: Cross-validation and stratified splitting are crucial for reliable performance assessment in medical ML applications
- **Confidence Communication**: Presenting prediction uncertainty through confidence scores enhances clinical utility and responsible AI implementation

### Future Improvements
- **Real Medical Data Integration**: Incorporate anonymized clinical datasets for more realistic model training and validation
- **Advanced Feature Engineering**: Add interaction terms and derived medical indicators to improve prediction accuracy
- **Model Explainability Tools**: Implement SHAP or LIME for transparent prediction reasoning in clinical contexts
- **Multi-Class Disease Prediction**: Extend beyond binary classification to predict multiple disease categories simultaneously

### Performance Insights
- **Memory Efficiency**: Both models maintain small footprints suitable for edge deployment in clinical environments
- **Prediction Speed**: Sub-second response times enable real-time clinical workflow integration
- **Cross-Model Consistency**: High agreement between Logistic Regression and Random Forest predictions indicates robust feature selection

### Technical Debt
- **Hard-Coded Parameters**: Medical parameter ranges and model hyperparameters should be externalized to configuration files
- **Error Handling**: Limited exception handling for edge cases in medical parameter input validation
- **Testing Coverage**: Insufficient automated testing for model prediction accuracy and web interface functionality

## Important Notes

- This tool is designed for demonstration and educational purposes only and should not replace professional medical consultation
- All medical parameters have defined realistic ranges to ensure clinically meaningful predictions
- The application processes all data locally without external transmission, ensuring patient privacy protection
- Synthetic dataset generation provides consistent training conditions but may not reflect complex real-world medical scenarios

## References

- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
- SMOTE: Synthetic Minority Oversampling Technique, Chawla et al., Journal of Artificial Intelligence Research 16 (2002) 321-357
- Streamlit Documentation: https://docs.streamlit.io/
