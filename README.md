# Machine Learning Flask Application - Iris Classifier

A beginner-friendly ML project for learning machine learning concepts using the famous Iris dataset.

## 🌸 Project Structure

```
machinebot/
├── src/                    # Core source code
│   └── data_explorer.py   # Data visualization and exploration
├── tutorials/              # Learning materials
│   ├── iris_trainer.py    # Educational Iris model trainer
│   └── model_trainer.py   # Generic model training script
├── notebooks/              # Jupyter notebooks (future)
├── models/                 # Trained ML models
│   ├── iris_model.pkl     # Trained Iris classifier
│   └── metadata.json      # Model information
├── templates/              # HTML templates
│   ├── index.html         # Main interface
│   └── learn.html         # Interactive learning dashboard
├── static/                 # Static files (CSS/JS/images)
├── data/                   # Datasets
├── app.py                  # Flask web server
├── config.py              # Configuration management
├── port_finder.py         # Port detection utility
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### 1. Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Iris Model

```bash
python tutorials/iris_trainer.py
```

This will:
- Load the Iris dataset (150 flower samples)
- Train a Decision Tree classifier
- Save the model to `models/iris_model.pkl`
- Show 100% accuracy on test data!

### 4. Start the Flask Server

```bash
python app.py
```

Server runs on **port 5555** (configured to avoid conflicts).

## 📚 Learning Resources

### Interactive Learning Dashboard
Visit: **http://localhost:5555/learn**

Features:
- Visual explanation of flower anatomy
- Interactive predictions with instant feedback
- Understanding of ML decision-making process
- Side-by-side species comparison

### Main Prediction Interface
Visit: **http://localhost:5555**

Try these sample inputs:
- **Setosa**: 5.1, 3.5, 1.4, 0.2
- **Versicolor**: 5.7, 2.9, 4.2, 1.3
- **Virginica**: 6.2, 2.8, 4.8, 1.8

### Data Exploration
```bash
python src/data_explorer.py
```

This generates visualizations showing:
- How the three species differ
- Which measurements matter most
- Why the model can achieve 100% accuracy

## 🌺 Understanding the Iris Dataset

### What We're Measuring
Each flower has 4 measurements (in cm):
1. **Sepal Length** - Length of the green protective leaves
2. **Sepal Width** - Width of the green protective leaves
3. **Petal Length** - Length of the colorful petals
4. **Petal Width** - Width of the colorful petals

### The Three Species
1. **Iris Setosa** - Small flowers with tiny petals (< 2cm)
2. **Iris Versicolor** - Medium-sized with balanced proportions
3. **Iris Virginica** - Large flowers with long petals (> 5cm)

### How ML Works Here
The model learns simple rules like:
```
IF petal_length < 2.5 cm:
    → It's SETOSA
ELSE IF petal_width < 1.75 cm:
    → It's VERSICOLOR
ELSE:
    → It's VIRGINICA
```

## 🛠️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main prediction interface |
| GET | `/learn` | Interactive learning dashboard |
| GET | `/health` | Server health check |
| POST | `/predict` | Make predictions (JSON) |
| GET | `/model/list` | List available models |
| POST | `/model/load` | Load a specific model |

### Example API Call
```bash
curl -X POST http://localhost:5555/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Response:
```json
{
  "prediction": 0,
  "probabilities": [1.0, 0.0, 0.0],
  "status": "success"
}
```

## 📊 Model Performance

- **Algorithm**: Decision Tree Classifier
- **Accuracy**: 100% on test set (30 samples)
- **Training Set**: 120 samples
- **Test Set**: 30 samples

## 🔧 Configuration

Edit `.env` file:
```env
FLASK_PORT=5555  # Port for Flask app
FLASK_DEBUG=True
FLASK_ENV=development
```

**Note**: Ports below 5100 are reserved for other applications.

## 📝 Key Learning Concepts

1. **Features**: Input data (flower measurements)
2. **Labels**: What we predict (flower species)
3. **Training**: Teaching the model patterns
4. **Testing**: Verifying the model learned correctly
5. **Prediction**: Using the model on new data

## 🎯 Next Steps

- Try different ML algorithms (Random Forest, SVM)
- Add cross-validation for better evaluation
- Implement feature importance visualization
- Create a notebook for deeper analysis
- Add more datasets for comparison

## 🚨 Troubleshooting

### Port Issues
```bash
python port_finder.py  # Find available ports
python port_finder.py --update  # Auto-update .env
```

### Model Not Loading
1. Ensure model exists: `ls models/`
2. Train model: `python tutorials/iris_trainer.py`
3. Restart Flask: `python app.py`

## 📚 Learning Path

1. Start with `/learn` page - understand the data
2. Run `iris_trainer.py` - see training process
3. Use `data_explorer.py` - visualize patterns
4. Try predictions - test your understanding
5. Modify the code - experiment and learn!

---

Built for learning ML fundamentals with the classic Iris dataset! 🌸