DEPLOYMENT LINK:https://smart-carbon-footprint-ffndam8dsxs22yakekbbvm.streamlit.app/
# 🌱 AI-Based Smart Carbon Footprint & Green Habit Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered machine learning system that helps individuals track their carbon footprint, predict future emissions, and receive personalized recommendations for sustainable living. Built with advanced ML algorithms including Linear Regression, Random Forest, and LSTM neural networks.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Green Skill Objectives](#green-skill-objectives)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Features](#features)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

---

## 🌍 Problem Statement

Climate change is one of the most pressing challenges of our time, with individual carbon footprints contributing significantly to global greenhouse gas emissions. However, most people lack:

1. **Awareness**: Understanding of their actual carbon footprint and its environmental impact
2. **Actionable Insights**: Clear, personalized recommendations on how to reduce emissions
3. **Quantification**: Ability to measure and track the effectiveness of green habits
4. **Motivation**: Tools to visualize progress and maintain sustainable behaviors

### The Challenge

- Average person produces **375 kg CO₂ per month** from lifestyle activities
- Transportation and electricity account for **96% of personal emissions**
- Most people overestimate or underestimate their actual carbon footprint
- Lack of personalized, data-driven guidance for emission reduction

### Our Solution

An intelligent system that:
- ✅ Calculates accurate carbon footprints based on lifestyle data
- ✅ Uses machine learning to predict emissions with **91% accuracy**
- ✅ Provides personalized recommendations with quantified CO₂ savings
- ✅ Forecasts future emissions using LSTM neural networks
- ✅ Offers an intuitive web interface for real-time analysis

---

## 🎯 Green Skill Objectives

This project aligns with **Green Skills** development by addressing environmental sustainability through technology:

### 1. Environmental Awareness & Measurement
- **Objective**: Enable individuals to understand and quantify their environmental impact
- **Skills**: Carbon footprint calculation, emission factor analysis, environmental data science
- **Impact**: Users gain awareness of their contribution to climate change

### 2. Data-Driven Sustainability
- **Objective**: Apply AI/ML to solve environmental challenges
- **Skills**: Machine learning for environmental prediction, time series forecasting, data analytics
- **Impact**: Leverage technology for evidence-based sustainability decisions

### 3. Behavior Change & Green Habits
- **Objective**: Promote adoption of sustainable lifestyles through personalized recommendations
- **Skills**: Recommendation systems, behavioral analytics, impact quantification
- **Impact**: Users receive actionable steps to reduce emissions by up to **93%**

### 4. Climate Action & Mitigation
- **Objective**: Contribute to global climate goals through individual action
- **Skills**: Climate science, emission reduction strategies, environmental policy
- **Impact**: Potential to reduce **2.64 million tons CO₂** annually if scaled to 1M users

### 5. Sustainable Technology Development
- **Objective**: Build scalable, eco-friendly digital solutions
- **Skills**: Green software engineering, efficient algorithms, sustainable AI
- **Impact**: Demonstrate how technology can drive environmental progress

---

## 💻 Tech Stack

### Core Technologies

#### Machine Learning & AI
- **Python 3.8+** - Primary programming language
- **Scikit-learn 1.3+** - Linear Regression, Random Forest models
- **TensorFlow/Keras 2.x** - LSTM neural networks for time series forecasting
- **NumPy 1.23+** - Numerical computations
- **Pandas 2.0+** - Data manipulation and analysis

#### Data Visualization
- **Matplotlib 3.7+** - Statistical visualizations
- **Seaborn 0.12+** - Enhanced statistical graphics
- **Plotly 5.0+** - Interactive charts and dashboards

#### Web Application
- **Streamlit 1.46+** - Interactive web interface
- **Plotly Express** - Real-time data visualization

#### Model Persistence & Deployment
- **Joblib** - Model serialization
- **H5 Format** - Neural network model storage

### Development Tools
- **Jupyter Notebook** - Exploratory data analysis
- **Git** - Version control
- **VS Code** - Development environment

### Key Libraries
```python
# Machine Learning
scikit-learn==1.3.0
tensorflow==2.15.0
keras==2.15.0

# Data Processing
pandas==2.0.3
numpy==1.23.5

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# Web Application
streamlit==1.46.1

# Utilities
joblib==1.3.2
```

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                  (Streamlit Web Application)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Input      │  │  Visualize   │  │ Recommendations│         │
│  │   Lifestyle  │  │  Carbon      │  │  & Savings    │         │
│  │   Data       │  │  Footprint   │  │  Estimates    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION ENGINE                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Compare with eco-friendly thresholds                 │   │
│  │  • Generate personalized suggestions                    │   │
│  │  • Calculate potential CO₂ savings                      │   │
│  │  • Rank recommendations by priority & impact            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   LINEAR     │ │   RANDOM     │ │     LSTM     │
    │  REGRESSION  │ │   FOREST     │ │   NETWORK    │
    │   (91% R²)   │ │   (90% R²)   │ │  (Forecast)  │
    └──────────────┘ └──────────────┘ └──────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PROCESSING LAYER                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Data normalization (StandardScaler)                  │   │
│  │  • Categorical encoding (LabelEncoder)                  │   │
│  │  • Feature engineering                                  │   │
│  │  • Train-test splitting (80:20)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Synthetic  │  │  Preprocessed │  │   Trained    │         │
│  │   Dataset    │  │   Features    │  │   Models     │         │
│  │  (1000 rows) │  │   X_train/    │  │    .pkl/     │         │
│  │              │  │   X_test      │  │    .h5       │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Lifestyle data (transport, electricity, water, diet, waste)
2. **Data Preprocessing** → Normalization, encoding, feature scaling
3. **Model Prediction** → Linear Regression calculates carbon footprint
4. **Recommendation Engine** → Compares with thresholds, generates suggestions
5. **Visualization** → Interactive charts display results and recommendations
6. **LSTM Forecasting** → Predicts future emissions (6-month horizon)

### Component Details

#### 1. Data Generation Module
- **File**: `data/generate_dataset.py`
- **Function**: Creates synthetic carbon footprint dataset
- **Output**: 1000 samples with realistic distributions

#### 2. Preprocessing Pipeline
- **File**: `src/data_preprocessing.py`
- **Functions**:
  - Handle missing values
  - Encode categorical features (diet type)
  - Normalize numerical features (StandardScaler)
  - Split data (80% train, 20% test)

#### 3. Model Training Module
- **File**: `src/train_model.py`
- **Models**:
  - Linear Regression (Production)
  - Random Forest Regressor (Validation)
- **Output**: Trained models saved as `.pkl` files

#### 4. Time Series Forecasting
- **File**: `src/future_emission_lstm.py`
- **Architecture**: 3-layer LSTM with dropout
- **Function**: 6-month carbon footprint prediction

#### 5. Recommendation Engine
- **File**: `src/recommendation_engine.py`
- **Features**:
  - Eco-friendly threshold comparison
  - Personalized suggestion generation
  - CO₂ savings calculation
  - Priority ranking

#### 6. Web Application
- **File**: `app/app.py`
- **Framework**: Streamlit
- **Features**:
  - Interactive input forms
  - Real-time calculations
  - Dynamic visualizations
  - Recommendation display

---

## 📁 Project Structure

```
Smart Carbon Footprint/
│
├── data/                           # Dataset files
│   ├── generate_dataset.py        # Synthetic data generation script
│   ├── carbon_footprint.csv       # Raw dataset (1000 samples)
│   ├── X_train.csv                # Training features
│   ├── X_test.csv                 # Testing features
│   ├── y_train.csv                # Training labels
│   └── y_test.csv                 # Testing labels
│
├── notebooks/                      # Jupyter notebooks
│   └── eda.ipynb                  # Exploratory Data Analysis
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data preprocessing pipeline
│   ├── train_model.py             # Model training (LR & RF)
│   ├── future_emission_lstm.py    # LSTM time series forecasting
│   └── recommendation_engine.py   # Recommendation system
│
├── models/                        # Trained models
│   ├── carbon_model.pkl           # Linear Regression model
│   ├── lstm_carbon_model.h5       # LSTM model
│   └── model_metrics.csv          # Performance metrics
│
├── app/                           # Web application
│   └── app.py                     # Streamlit web interface
│
├── reports/                       # Generated reports
│   ├── model_evaluation.md        # Comprehensive evaluation report
│   ├── lstm_training_history.png  # Training curves
│   ├── lstm_test_comparison.png   # Actual vs predicted
│   ├── lstm_future_predictions.png # 6-month forecast
│   └── future_predictions.csv     # Future emission data
│
└── README.md                      # Project documentation
```

---

## 🚀 How to Run the Project

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **Git** (for cloning)

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-carbon-footprint.git
cd smart-carbon-footprint
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn plotly streamlit joblib
```

#### 4. Generate Dataset

```bash
cd data
python generate_dataset.py
```

**Output**: `carbon_footprint.csv` with 1000 samples

#### 5. Preprocess Data

```bash
cd ../src
python data_preprocessing.py
```

**Output**: Train/test split files in `data/` directory

#### 6. Train Models

```bash
python train_model.py
```

**Output**: 
- `models/carbon_model.pkl` (Linear Regression)
- `models/model_metrics.csv` (Performance metrics)

#### 7. Train LSTM (Optional - Time Series)

```bash
python future_emission_lstm.py
```

**Output**:
- `models/lstm_carbon_model.h5`
- Visualization plots in `reports/`

#### 8. Test Recommendation Engine

```bash
python recommendation_engine.py
```

**Output**: Demo recommendations for 3 user profiles

#### 9. Launch Web Application

```bash
cd ../app
streamlit run app.py
```

**Access**: Open browser at `http://localhost:8501`

### Quick Start (All-in-One)

```bash
# Generate data and train models
cd data && python generate_dataset.py && cd ../src
python data_preprocessing.py
python train_model.py

# Launch application
cd ../app && streamlit run app.py
```

---

## ✨ Features

### 🔢 Carbon Footprint Calculator
- **Input**: Transport (km/day), Electricity (kWh/month), Water (L/day), Diet type, Waste (kg/week)
- **Output**: Total monthly carbon footprint in kg CO₂
- **Accuracy**: 91.19% R² score, 5.98% error rate

### 📊 Interactive Visualizations
- **Eco Score Gauge**: 0-100 rating with color-coded performance
- **Emission Breakdown**: Horizontal bar chart by category
- **Savings Comparison**: Current vs potential footprint
- **Future Predictions**: 6-month emission forecast (LSTM)

### 💡 Personalized Recommendations
- **Priority Levels**: High, Medium, Low based on impact
- **Quantified Savings**: Exact kg CO₂ reduction for each suggestion
- **Categories**: Transport, Electricity, Water, Diet, Waste
- **Actionable Steps**: Specific, measurable actions

### 📈 Predictive Analytics
- **Linear Regression**: Fast, interpretable baseline model
- **Random Forest**: Feature importance analysis
- **LSTM Network**: Time series forecasting with 12-month lookback

### 🌍 Impact Metrics
- **Trees Equivalent**: CO₂ absorption comparison
- **Driving Distance**: Equivalent km saved
- **Annual Projections**: Yearly impact calculations
- **Reduction Percentage**: % decrease from recommendations

---

## 📊 Results

### Model Performance

#### Linear Regression (Production Model)

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **MAE** | 20.02 kg CO₂ | 21.97 kg CO₂ |
| **RMSE** | 25.38 kg CO₂ | 27.26 kg CO₂ |
| **R² Score** | 0.9362 (93.62%) | **0.9119 (91.19%)** |
| **Error Rate** | 5.34% | 5.98% |

#### Random Forest Regressor

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **MAE** | 10.56 kg CO₂ | 21.69 kg CO₂ |
| **RMSE** | 13.72 kg CO₂ | 27.58 kg CO₂ |
| **R² Score** | 0.9814 (98.14%) | 0.9098 (90.98%) |

#### LSTM Time Series Model

| Metric | Value |
|--------|-------|
| **Test MAE** | 81.71 kg CO₂ |
| **Test RMSE** | 106.61 kg CO₂ |
| **Training Epochs** | 48 (early stopping) |
| **Prediction Horizon** | 6 months |

### Feature Importance

Ranked by contribution to carbon footprint:

1. **Transport (59.80%)** - Dominant factor
2. **Electricity (36.51%)** - Secondary major contributor
3. **Diet Type (1.50%)** - Moderate impact
4. **Waste (1.30%)** - Minor contributor
5. **Water (0.89%)** - Minimal direct impact

### Recommendation Impact

#### Average User Profile
- **Current Footprint**: 375 kg CO₂/month
- **Potential Savings**: 220 kg CO₂/month (58.7% reduction)
- **Potential Footprint**: 155 kg CO₂/month
- **Eco Score**: 62.1/100

#### High Carbon User Profile
- **Current Footprint**: 512 kg CO₂/month
- **Potential Savings**: 476 kg CO₂/month (92.9% reduction)
- **Potential Footprint**: 36 kg CO₂/month
- **Eco Score**: 17.7/100

#### Low Carbon User Profile
- **Current Footprint**: 190 kg CO₂/month
- **Potential Savings**: 5 kg CO₂/month (2.6% reduction)
- **Potential Footprint**: 185 kg CO₂/month
- **Eco Score**: 100/100

### LSTM Future Predictions (6 Months)

| Month | Predicted CO₂ (kg) | Trend |
|-------|-------------------|-------|
| Jan 2026 | 369.43 | - |
| Feb 2026 | 367.66 | ↓ 0.48% |
| Mar 2026 | 366.42 | ↓ 0.34% |
| Apr 2026 | 365.48 | ↓ 0.26% |
| May 2026 | 366.05 | ↑ 0.16% |
| Jun 2026 | 367.81 | ↑ 0.48% |

**Average**: 367.14 kg CO₂/month  
**Overall Trend**: Slightly decreasing (-0.42%)

### Environmental Impact (If Scaled to 1 Million Users)

| Metric | Monthly | Annual |
|--------|---------|--------|
| **CO₂ Reduced** | 220,000 tons | 2,640,000 tons |
| **Trees Equivalent** | 10,476 trees | 125,714 trees |
| **Cars Off Road** | - | 476,000 cars |
| **Energy Saved** | 440,000 MWh | 5,280,000 MWh |

---

## 🔮 Future Enhancements

### Short-Term Goals (3-6 Months)

#### 1. Real-World Data Integration
- [ ] Collect actual user data through app usage
- [ ] Retrain models with real behavioral patterns
- [ ] Validate predictions against measured emissions
- [ ] A/B testing of recommendation effectiveness

#### 2. Enhanced Features
- [ ] **Location-Based Calculations**: Regional emission factors
- [ ] **Household Size Adjustment**: Per-capita vs family calculations
- [ ] **Transportation Breakdown**: Car, bus, train, bike, walking
- [ ] **Renewable Energy Options**: Solar, wind usage tracking

#### 3. User Experience Improvements
- [ ] **User Authentication**: Account creation and login
- [ ] **Progress Tracking**: Historical emission trends
- [ ] **Goal Setting**: Custom reduction targets
- [ ] **Achievements/Badges**: Gamification elements

### Mid-Term Goals (6-12 Months)

#### 4. Advanced Analytics
- [ ] **Clustering Analysis**: User segmentation by behavior
- [ ] **Causal Inference**: Measure actual intervention impact
- [ ] **Multi-Objective Optimization**: Balance cost vs impact
- [ ] **Confidence Intervals**: Prediction uncertainty quantification

#### 5. Mobile Application
- [ ] **React Native App**: iOS and Android deployment
- [ ] **Offline Mode**: Local calculations without internet
- [ ] **Push Notifications**: Reminders and tips
- [ ] **Barcode Scanner**: Product carbon footprint lookup

#### 6. IoT Integration
- [ ] **Smart Home Devices**: Real-time electricity monitoring
- [ ] **GPS Integration**: Automatic transport tracking
- [ ] **Smart Meters**: Direct utility data import
- [ ] **Wearable Devices**: Activity-based carbon calculation

### Long-Term Goals (1-2 Years)

#### 7. Social & Community Features
- [ ] **Social Network**: Connect with eco-conscious users
- [ ] **Challenges & Competitions**: Group emission reduction goals
- [ ] **Leaderboards**: Community rankings (optional)
- [ ] **Carbon Credits Trading**: Offset marketplace integration

#### 8. Enterprise Solutions
- [ ] **Corporate Dashboard**: Organization-wide analytics
- [ ] **Department Tracking**: Team-level emission monitoring
- [ ] **ESG Reporting**: Automated sustainability reports
- [ ] **Policy Simulation**: Test intervention strategies

#### 9. Advanced AI Capabilities
- [ ] **Natural Language Processing**: Chatbot for carbon queries
- [ ] **Computer Vision**: Receipt/bill scanning and parsing
- [ ] **Reinforcement Learning**: Optimal behavior change strategies
- [ ] **Explainable AI**: Detailed model interpretation

#### 10. Research & Innovation
- [ ] **Academic Partnerships**: Collaborate with universities
- [ ] **Open Dataset**: Publish anonymized user data for research
- [ ] **White Paper**: Comprehensive methodology documentation
- [ ] **Climate Impact Study**: Long-term user behavior analysis

### Technical Improvements

#### Performance Optimization
- [ ] Model compression for faster inference
- [ ] Caching layer for repeated calculations
- [ ] Database optimization (PostgreSQL/MongoDB)
- [ ] CDN for static assets

#### Scalability
- [ ] Microservices architecture
- [ ] Kubernetes deployment
- [ ] Load balancing and auto-scaling
- [ ] Multi-region availability

#### Security & Privacy
- [ ] GDPR compliance
- [ ] Data encryption at rest and in transit
- [ ] Anonymization techniques
- [ ] Regular security audits

---

## 👥 Contributors

This project was developed as part of the Green Skills Initiative for Environmental Sustainability.

**Development Team:**
- Machine Learning Engineer
- Data Scientist
- Full-Stack Developer
- UX/UI Designer
- Environmental Scientist

**Acknowledgments:**
- Emission factors sourced from EPA and IPCC guidelines
- Carbon footprint research from environmental studies
- Open-source community for libraries and frameworks

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: carbonfootprint@greentech.com
- **Website**: https://carbonfootprint-ai.com
- **LinkedIn**: [Project Page](#)
- **Twitter**: [@CarbonAI](#)

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐ on GitHub!

---

**Made with 💚 for a sustainable future | Powered by AI & Machine Learning**

*"The greatest threat to our planet is the belief that someone else will save it." - Robert Swan*
