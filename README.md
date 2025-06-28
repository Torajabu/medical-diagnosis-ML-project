# 🏥 Medical Disease Diagnosis using AI

A machine learning web application that predicts disease risk based on patient medical parameters using Logistic Regression and Random Forest algorithms.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/sklearn-v1.3+-green.svg)

## Preview
![Medical Diagnosis Demo](https://raw.githubusercontent.com/Torajabu/medical-diagnosis-ML-project/main/2025-06-2816-11-56-ezgif.com-video-to-gif-converter.gif)

![Screenshot](https://raw.githubusercontent.com/Torajabu/medical-diagnosis-ML-project/main/Screenshot%20from%202025-06-28%2016-23-33.png)

how to display in readme

## 🌟 Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Dual ML Models**: Compare Logistic Regression vs Random Forest predictions
- **Real-time Predictions**: Instant disease risk assessment
- **Confidence Scores**: Prediction probability for better decision making
- **Medical Disclaimer**: Responsible AI with proper medical guidance
- **Balanced Dataset**: Uses SMOTE for handling class imbalance

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Imbalanced-learn** - SMOTE for class balancing
- **Joblib** - Model persistence

## 📊 Dataset Features

The model uses the following medical parameters:
- **Age**: Patient age (18-90 years)
- **Gender**: Male/Female
- **Blood Pressure**: Systolic BP (80-200 mmHg)
- **Cholesterol**: Total cholesterol level (100-400 mg/dL)
- **Sugar Level**: Blood glucose (50-300 mg/dL)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Torajabu/medical-diagnosis-ML-project.git
cd medical-diagnosis-ML-project
```

### 2. Install Dependencies
```bash
pip install streamlit pandas scikit-learn imbalanced-learn joblib numpy
```

### 3. Generate Sample Data
```bash
python3 create_data.py
```

### 4. Train Models
```bash
python3 train_models.py
```

### 5. Run the Web Application
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
medical-diagnosis-ML-project/
├── README.md
├── create_data.py          # Generate sample medical data
├── train_models.py         # Train ML models
├── streamlit_app.py        # Web application
├── medical_records.csv     # Generated dataset
├── lr_model.pkl           # Trained Logistic Regression model
└── rf_model.pkl           # Trained Random Forest model
```

## 📈 Model Performance

The application trains two models and displays their performance:

- **Logistic Regression**: Linear approach, fast and interpretable
- **Random Forest**: Ensemble method, handles non-linear relationships

Both models use:
- **SMOTE** for handling class imbalance
- **Stratified train-test split** (80/20)
- **Cross-validation** for robust evaluation

## 🖥️ Usage

1. **Launch the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Input patient data**:
   - Adjust sliders for Age, Blood Pressure, Cholesterol, Sugar Level
   - Select Gender using radio buttons
   - Choose ML model (Logistic Regression or Random Forest)

3. **Get prediction**:
   - Click "🔍 Diagnose" button
   - View risk assessment with confidence score
   - Review input summary

## 👨‍💻 Created by Rajab

This application was developed by Rajab as a demonstration of machine learning capabilities in healthcare prediction.

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This tool is for demonstration and educational purposes
- **Not Medical Advice**: Results should not replace professional medical consultation
- **Consult Healthcare Professionals**: Always seek proper medical advice for health concerns

## 🛡️ Data Privacy

- No personal data is stored or transmitted
- All computations happen locally
- Sample data is synthetically generated

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Install missing packages using pip
2. **File not found errors**: Ensure all files are in the correct directory
3. **Model loading errors**: Run `train_models.py` first to generate model files
4. **Streamlit command not found**: Add `~/.local/bin` to your PATH

### Getting Help:

- Check the [Issues](https://github.com/Torajabu/medical-diagnosis-ML-project/issues) page
- Create a new issue with detailed error description
- Include Python version and operating system information

## 🙏 Acknowledgments

- Scikit-learn community for excellent ML tools
- Streamlit team for the amazing web framework
- Healthcare professionals for inspiration

## 📞 Contact

Rajab - Developer

Project Link: [https://github.com/Torajabu/medical-diagnosis-ML-project](https://github.com/Torajabu/medical-diagnosis-ML-project)

---

⭐ **Star this repo if you found it helpful!** ⭐
