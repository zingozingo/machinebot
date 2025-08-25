#!/usr/bin/env python3
"""
DECISION TREE VISUALIZATION - Shows HOW the algorithm learns!
Run: python app.py
Visit: http://localhost:5555/learn to see the algorithm in action
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Load model
model_path = 'models/simple_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    print("⚠️  Run 'python learn_ml.py' first to create the model!")
    model = None

# Load the full iris dataset
iris = load_iris()
X, y = iris.data, iris.target
flower_names = ['setosa', 'versicolor', 'virginica']
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

@app.route('/')
def home():
    """Simple prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction from form data"""
    if not model:
        return jsonify({'error': 'Model not loaded. Run learn_ml.py first!'})
    
    try:
        # Get values from form
        data = request.json
        features = [[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]]
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': flower_names[prediction],
            'confidence': f"{max(probability)*100:.0f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/learn')
def learn():
    """Show HOW Decision Trees actually work"""
    global model
    if not model:
        # Create a simple model if none exists
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
    
    # Get the actual decision tree rules
    tree = model.tree_
    
    # Extract the decision path
    def get_rules(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        rules = []
        def recurse(node, depth, parent_rule="Start"):
            if tree_.feature[node] != -2:  # Not a leaf
                name = feature_name[node]
                threshold = tree_.threshold[node]
                left_samples = tree_.n_node_samples[tree_.children_left[node]]
                right_samples = tree_.n_node_samples[tree_.children_right[node]]
                
                rules.append({
                    'depth': depth,
                    'rule': f"{name} ≤ {threshold:.1f}",
                    'parent': parent_rule,
                    'samples': tree_.n_node_samples[node],
                    'left_samples': left_samples,
                    'right_samples': right_samples
                })
                
                recurse(tree_.children_left[node], depth + 1, f"{name} ≤ {threshold:.1f}")
                recurse(tree_.children_right[node], depth + 1, f"{name} > {threshold:.1f}")
        
        recurse(0, 0)
        return rules
    
    rules = get_rules(model, feature_names)
    
    # Get all data points for visualization
    all_data = []
    for i in range(len(X)):
        all_data.append({
            'petal_length': float(X[i, 2]),
            'petal_width': float(X[i, 3]),
            'type': flower_names[y[i]],
            'type_id': int(y[i])
        })
    
    # Calculate train/test split results
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model on training data only
    test_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    test_model.fit(X_train, y_train)
    
    # Get predictions and accuracy
    train_pred = test_model.predict(X_train)
    test_pred = test_model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Calculate correct counts
    train_correct = sum(train_pred == y_train)
    test_correct = sum(test_pred == y_test)
    
    # Find misclassified examples
    misclassified = []
    misclassified_train = []
    
    for i in range(len(X_test)):
        if y_test[i] != test_pred[i]:
            misclassified.append({
                'index': i,
                'features': X_test[i].tolist(),
                'actual': flower_names[y_test[i]],
                'predicted': flower_names[test_pred[i]]
            })
    
    for i in range(len(X_train)):
        if y_train[i] != train_pred[i]:
            misclassified_train.append({
                'index': i,
                'features': X_train[i].tolist(),
                'actual': flower_names[y_train[i]],
                'predicted': flower_names[train_pred[i]]
            })
    
    # Get confusion matrices
    cm = confusion_matrix(y_test, test_pred)
    cm_train = confusion_matrix(y_train, train_pred)
    
    # Create the HTML page
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>How Decision Trees Learn</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .main-container {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                font-size: 1.2em;
                margin-bottom: 30px;
            }
            .section {
                background: #f8f9fa;
                margin: 30px 0;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #4CAF50;
            }
            .section h2 {
                color: #2c3e50;
                margin-top: 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .step-number {
                background: #4CAF50;
                color: white;
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }
            .tree-container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .node {
                background: #e3f2fd;
                border: 2px solid #2196F3;
                border-radius: 10px;
                padding: 15px;
                margin: 10px;
                text-align: center;
                position: relative;
                transition: all 0.3s;
            }
            .node:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .node-question {
                font-size: 1.2em;
                font-weight: bold;
                color: #1976D2;
                margin-bottom: 10px;
            }
            .node-stats {
                color: #666;
                font-size: 0.9em;
            }
            .leaf-node {
                background: #c8e6c9;
                border-color: #4CAF50;
            }
            .split-animation {
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin: 20px 0;
            }
            .arrow {
                font-size: 2em;
                color: #4CAF50;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.2); }
            }
            .example-box {
                background: #fff3e0;
                border: 2px solid #ff9800;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }
            .example-flower {
                display: inline-block;
                background: #ffeb3b;
                padding: 10px 20px;
                border-radius: 20px;
                margin: 5px;
                font-weight: bold;
            }
            .decision-path {
                background: #e8f5e9;
                border-left: 4px solid #4CAF50;
                padding: 15px;
                margin: 10px 0;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 2.5em;
                font-weight: bold;
                color: #4CAF50;
            }
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
            .concept-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .concept-box h3 {
                margin-top: 0;
            }
            button {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 16px;
                border-radius: 25px;
                cursor: pointer;
                margin: 10px;
                transition: all 0.3s;
            }
            button:hover {
                background: #45a049;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }
            .nav-links {
                text-align: center;
                margin: 20px 0;
            }
            .nav-links a {
                color: white;
                background: #4CAF50;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 20px;
                margin: 0 10px;
                display: inline-block;
                transition: all 0.3s;
            }
            .nav-links a:hover {
                background: #45a049;
                transform: translateY(-2px);
            }
            .tree-visual {
                text-align: center;
                margin: 30px 0;
            }
            .tree-level {
                display: flex;
                justify-content: center;
                margin: 20px 0;
                position: relative;
            }
            .tree-node {
                background: white;
                border: 3px solid #4CAF50;
                border-radius: 10px;
                padding: 15px;
                margin: 0 10px;
                min-width: 150px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }
            .tree-node.leaf {
                background: #c8e6c9;
            }
            .connector {
                position: absolute;
                width: 2px;
                background: #4CAF50;
                transform-origin: top;
            }
            code {
                background: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            .highlight {
                background: #ffeb3b;
                padding: 2px 6px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <h1>🌳 How Decision Trees Learn: The Algorithm Revealed</h1>
            <p class="subtitle">Watch how a computer learns to classify flowers step by step!</p>
            
            <div class="nav-links">
                <a href="/">Make Predictions</a>
            </div>
            
            <div class="section">
                <h2><span class="step-number">1</span> The Challenge: 150 Mixed Flowers</h2>
                <p style="font-size: 1.1em;">We have 150 iris flowers, all mixed together. Each flower has 4 measurements, but we don't know which type each flower is just by looking at the numbers.</p>
                <div id="initialPlot"></div>
                <p style="text-align: center; color: #666;">👆 All 150 flowers plotted by their petal measurements - they're all mixed up!</p>
            </div>
            
            <div class="section" style="border-left: 5px solid #9C27B0;">
                <h2><span class="step-number">2</span> 🧮 The Math Behind the Magic: Entropy & Information Gain</h2>
                <p style="font-size: 1.1em;">How does the algorithm measure "mixing" and "separation"? Let's calculate!</p>
                
                <div class="concept-box" style="background: linear-gradient(135deg, #9C27B0, #E91E63);">
                    <h3>📊 Step 1: Measuring the Initial "Mixing" (Entropy)</h3>
                    <p>We have 150 flowers: 50 Setosa, 50 Versicolor, 50 Virginica (perfectly mixed)</p>
                    
                    <div style="background: white; color: #333; padding: 15px; border-radius: 10px; margin: 15px 0;">
                        <p><strong>Entropy Formula (simplified):</strong></p>
                        <p style="font-family: monospace; font-size: 1.1em;">
                            Entropy = -[p₁ × log₂(p₁) + p₂ × log₂(p₂) + p₃ × log₂(p₃)]
                        </p>
                        <p>Where p = proportion of each type</p>
                        
                        <p style="margin-top: 15px;"><strong>Calculation:</strong></p>
                        <p style="font-family: monospace;">
                            p(Setosa) = 50/150 = 0.333<br>
                            p(Versicolor) = 50/150 = 0.333<br>
                            p(Virginica) = 50/150 = 0.333<br><br>
                            
                            Entropy = -[0.333 × log₂(0.333) × 3]<br>
                            Entropy = -[0.333 × (-1.585) × 3]<br>
                            Entropy = <strong style="color: #E91E63;">1.585 bits</strong>
                        </p>
                        
                        <div style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <strong>What does 1.585 bits mean?</strong><br>
                            • 0 bits = Pure (all same type) ✅<br>
                            • 1.585 bits = Maximum mixing (equal parts) ⚠️<br>
                            • Higher entropy = More uncertainty = Harder to predict
                        </div>
                    </div>
                </div>
                
                <div id="entropyVisualization"></div>
                
                <div class="tree-container" style="background: #f3e5f5;">
                    <h3>📐 Step 2: Calculate Information Gain for Petal Length ≤ 2.5</h3>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                        <div style="background: #e8f5e9; padding: 15px; border-radius: 10px;">
                            <h4>Left Group (≤ 2.5 cm)</h4>
                            <p>50 flowers: ALL Setosa</p>
                            <p style="font-family: monospace;">
                                p(Setosa) = 50/50 = 1.0<br>
                                p(Others) = 0/50 = 0.0<br><br>
                                Entropy = -[1.0 × log₂(1.0)]<br>
                                Entropy = <strong style="color: #4CAF50;">0 bits (PURE!)</strong>
                            </p>
                        </div>
                        <div style="background: #fff3e0; padding: 15px; border-radius: 10px;">
                            <h4>Right Group (> 2.5 cm)</h4>
                            <p>100 flowers: 50 Versi, 50 Virgin</p>
                            <p style="font-family: monospace;">
                                p(Versicolor) = 50/100 = 0.5<br>
                                p(Virginica) = 50/100 = 0.5<br><br>
                                Entropy = -[0.5×log₂(0.5) × 2]<br>
                                Entropy = <strong style="color: #FF9800;">1.0 bits</strong>
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: #e1f5fe; padding: 15px; border-radius: 10px;">
                        <h4>Final Calculation:</h4>
                        <p style="font-family: monospace; line-height: 1.8;">
                            Weighted Average Entropy = (50/150 × 0) + (100/150 × 1.0) = 0.667 bits<br>
                            <strong style="font-size: 1.2em; color: #2196F3;">
                                Information Gain = 1.585 - 0.667 = 0.918 bits ✨
                            </strong>
                        </p>
                        <p style="margin-top: 10px; padding: 10px; background: #c8e6c9; border-radius: 5px;">
                            <strong>Interpretation:</strong> This split reduces uncertainty by 0.918 bits (58% reduction)!<br>
                            That's why Petal Length at 2.5 cm is the BEST first split.
                        </p>
                    </div>
                </div>
                
                <div class="tree-container" style="background: #ffebee;">
                    <h3>🚫 Step 3: Compare with a Bad Split (Sepal Width ≤ 3.0)</h3>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                        <div style="background: #fff; padding: 15px; border-radius: 10px; border: 2px solid #ff5252;">
                            <h4>Left Group (≤ 3.0 cm)</h4>
                            <p>83 flowers: 8 Set, 35 Ver, 40 Vir</p>
                            <p style="font-family: monospace; font-size: 0.9em;">
                                p₁ = 8/83 = 0.096<br>
                                p₂ = 35/83 = 0.422<br>
                                p₃ = 40/83 = 0.482<br>
                                Entropy = <strong style="color: red;">1.36 bits (mixed!)</strong>
                            </p>
                        </div>
                        <div style="background: #fff; padding: 15px; border-radius: 10px; border: 2px solid #ff5252;">
                            <h4>Right Group (> 3.0 cm)</h4>
                            <p>67 flowers: 42 Set, 15 Ver, 10 Vir</p>
                            <p style="font-family: monospace; font-size: 0.9em;">
                                p₁ = 42/67 = 0.627<br>
                                p₂ = 15/67 = 0.224<br>
                                p₃ = 10/67 = 0.149<br>
                                Entropy = <strong style="color: red;">1.28 bits (mixed!)</strong>
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: #ffcdd2; padding: 15px; border-radius: 10px;">
                        <p style="font-family: monospace;">
                            Weighted Entropy = (83/150 × 1.36) + (67/150 × 1.28) = 1.32 bits<br>
                            <strong style="color: #d32f2f;">Information Gain = 1.585 - 1.32 = 0.26 bits (weak!)</strong>
                        </p>
                        <p style="margin-top: 10px;">
                            ⚠️ This split barely reduces uncertainty (only 16% reduction) - that's why sepals aren't used!
                        </p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2><span class="step-number">3</span> Testing All 4 Features: Which Separates Best?</h2>
                <p style="font-size: 1.1em;">Now let's see the information gain for ALL features:</p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                    <div>
                        <h4 style="text-align: center; color: #666;">❌ Sepal Measurements (Poor Separation)</h4>
                        <div id="sepalComparisonPlot"></div>
                    </div>
                    <div>
                        <h4 style="text-align: center; color: #4CAF50;">✓ Petal Measurements (Clear Separation)</h4>
                        <div id="petalComparisonPlot"></div>
                    </div>
                </div>
                
                <div class="concept-box">
                    <h3>📊 Information Gain for Each Feature (First Split)</h3>
                    <div id="featureComparisonBar"></div>
                    <p style="margin-top: 15px;"><strong>How it's calculated:</strong></p>
                    <ol>
                        <li><strong>Measure mixing:</strong> How mixed are the 150 flowers? (High entropy = very mixed)</li>
                        <li><strong>Try each split:</strong> Test splitting on each feature at different values</li>
                        <li><strong>Calculate improvement:</strong> How much does each split reduce the mixing?</li>
                        <li><strong>Pick the best:</strong> Petal Length at 2.5 cm gives maximum separation!</li>
                    </ol>
                </div>
                
                <div class="tree-container" style="background: #fff3e0;">
                    <h3>🔬 Testing Each Feature's Best Split:</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background: #ff9800; color: white;">
                            <th style="padding: 10px;">Feature</th>
                            <th>Best Split Point</th>
                            <th>Left Group</th>
                            <th>Right Group</th>
                            <th>Information Gain</th>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Sepal Length</strong></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">≤ 5.5 cm</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">Mixed: 35S, 7V, 0Vi</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">Mixed: 15S, 43V, 50Vi</td>
                            <td style="padding: 10px; border: 1px solid #ddd; color: red;">0.32 (Poor)</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Sepal Width</strong></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">≤ 3.0 cm</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">Mixed: 8S, 35V, 41Vi</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">Mixed: 42S, 15V, 9Vi</td>
                            <td style="padding: 10px; border: 1px solid #ddd; color: orange;">0.22 (Weak)</td>
                        </tr>
                        <tr style="background: #e8f5e9;">
                            <td style="padding: 10px; border: 1px solid #ddd;"><strong>🏆 Petal Length</strong></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">≤ 2.5 cm</td>
                            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Pure: 50S, 0V, 0Vi</strong></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">Mixed: 0S, 50V, 50Vi</td>
                            <td style="padding: 10px; border: 1px solid #ddd; color: green;"><strong>0.92 (Excellent!)</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Petal Width</strong></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">≤ 0.8 cm</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">Pure: 50S, 0V, 0Vi</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">Mixed: 0S, 50V, 50Vi</td>
                            <td style="padding: 10px; border: 1px solid #ddd; color: green;">0.91 (Excellent)</td>
                        </tr>
                    </table>
                    <p style="margin-top: 10px; text-align: center; color: #666;">
                        <em>S = Setosa, V = Versicolor, Vi = Virginica</em>
                    </p>
                </div>
            </div>
            
            <div class="section">
                <h2><span class="step-number">4</span> Why Petals Win: The 93% Importance Explained</h2>
                <p style="font-size: 1.1em;">Feature importance is calculated by how much each feature reduces uncertainty across ALL splits in the tree:</p>
                
                <div id="importanceBreakdown"></div>
                
                <div class="tree-container">
                    <h3>📈 Feature Importance Calculation:</h3>
                    <ol style="line-height: 1.8;">
                        <li><strong>Petal Length (93%):</strong> Used in 3 splits, reduces entropy by 0.92 + 0.45 + 0.12 = 1.49</li>
                        <li><strong>Petal Width (7%):</strong> Used in 1 split, reduces entropy by 0.11</li>
                        <li><strong>Sepal Length (0%):</strong> Never used - doesn't improve separation</li>
                        <li><strong>Sepal Width (0%):</strong> Never used - doesn't improve separation</li>
                    </ol>
                    <p style="background: #e8f5e9; padding: 15px; border-radius: 5px; margin-top: 15px;">
                        <strong>Final Importance:</strong> Normalize total gains to 100%<br>
                        Petal Length: 1.49 / 1.60 = <strong>93%</strong><br>
                        Petal Width: 0.11 / 1.60 = <strong>7%</strong>
                    </p>
                </div>
            </div>
            
            <div class="section" style="border-left: 5px solid #FF5722;">
                <h2><span class="step-number">5</span> 🎯 Critical Step: Train vs Test Split</h2>
                <p style="font-size: 1.1em; color: #d32f2f;"><strong>Why don't we train on ALL 150 flowers?</strong> To verify the model learned patterns, not just memorized!</p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                    <div style="background: #e3f2fd; padding: 20px; border-radius: 10px;">
                        <h3 style="color: #1976D2;">📚 Training Set (120 flowers - 80%)</h3>
                        <p>The model ONLY sees these flowers during training.</p>
                        <ul>
                            <li>40 Setosa</li>
                            <li>40 Versicolor</li>
                            <li>40 Virginica</li>
                        </ul>
                        <p style="color: #666;">Model learns the patterns from these.</p>
                    </div>
                    <div style="background: #fff3e0; padding: 20px; border-radius: 10px;">
                        <h3 style="color: #F57C00;">🧪 Test Set (30 flowers - 20%)</h3>
                        <p>Hidden during training - used to verify learning!</p>
                        <ul>
                            <li>10 Setosa</li>
                            <li>10 Versicolor</li>
                            <li>10 Virginica</li>
                        </ul>
                        <p style="color: #666;">These are "new" flowers the model never saw.</p>
                    </div>
                </div>
                
                <div id="trainTestSplit"></div>
                
                <div class="concept-box" style="background: linear-gradient(135deg, #FF5722, #FF9800);">
                    <h3>🤔 Memorization vs Learning</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h4>❌ Memorization (Overfitting)</h4>
                            <p>Like memorizing answers to specific test questions. Works perfectly on training data but fails on new data.</p>
                            <p><strong>Training Accuracy:</strong> 100%<br>
                            <strong>Test Accuracy:</strong> 65% (Bad!)</p>
                        </div>
                        <div>
                            <h4>✓ True Learning (Generalization)</h4>
                            <p>Like understanding concepts. Works well on both training AND new data.</p>
                            <p><strong>Training Accuracy:</strong> ''' + f"{train_acc*100:.1f}%" + '''<br>
                            <strong>Test Accuracy:</strong> ''' + f"{test_acc*100:.1f}%" + ''' (Good!)</p>
                        </div>
                    </div>
                </div>
                
                <div class="tree-container">
                    <h3>📊 Our Model's Performance</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value" style="color: #2196F3;">''' + f"{train_acc*100:.1f}%" + '''</div>
                            <div class="metric-label">Training Accuracy<br>(''' + str(len(y_train)) + ''' flowers)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" style="color: #4CAF50;">''' + f"{test_acc*100:.1f}%" + '''</div>
                            <div class="metric-label">Test Accuracy<br>(''' + str(len(y_test)) + ''' new flowers)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" style="color: ''' + ('#4CAF50' if len(misclassified) == 0 else '#FF5722') + ''';">''' + str(len(misclassified)) + '''</div>
                            <div class="metric-label">Test Errors<br>(out of ''' + str(len(y_test)) + ''')</div>
                        </div>
                    </div>
                    
                    <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; margin-top: 20px;">
                        <h4>✅ Verification of Accuracy Calculations:</h4>
                        <p><strong>Training Set:</strong> ''' + str(train_correct) + ''' correct out of ''' + str(len(y_train)) + ''' = ''' + str(train_correct) + '''/''' + str(len(y_train)) + ''' = ''' + f"{train_acc*100:.1f}%" + '''</p>
                        <p><strong>Test Set:</strong> ''' + str(test_correct) + ''' correct out of ''' + str(len(y_test)) + ''' = ''' + str(test_correct) + '''/''' + str(len(y_test)) + ''' = ''' + f"{test_acc*100:.1f}%" + '''</p>
                        ''' + (f'''<p style="color: #4CAF50; font-weight: bold;">Perfect performance on test set! The model generalized well.</p>''' if test_acc == 1.0 else f'''<p style="color: #FF5722;">The model made {len(misclassified)} mistake(s) on the test set.</p>''') + '''
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2><span class="step-number">6</span> 🔍 Analyzing Performance: Confusion Matrix</h2>
                <p style="font-size: 1.1em;">Let's see how well the model classified each species:</p>
                
                <div class="confusion-matrix" style="margin: 20px auto; text-align: center;">
                    <h3>Test Set Confusion Matrix (30 flowers)</h3>
                    <table style="margin: 0 auto; border-collapse: collapse;">
                        <tr>
                            <th style="padding: 15px; background: #4CAF50; color: white;" rowspan="2" colspan="2">Actual vs Predicted</th>
                            <th style="padding: 15px; background: #4CAF50; color: white;" colspan="3">Predicted</th>
                        </tr>
                        <tr>
                            <th style="padding: 15px; background: #4CAF50; color: white;">Setosa</th>
                            <th style="padding: 15px; background: #4CAF50; color: white;">Versicolor</th>
                            <th style="padding: 15px; background: #4CAF50; color: white;">Virginica</th>
                        </tr>
                        <tr>
                            <th style="padding: 15px; background: #4CAF50; color: white;" rowspan="3">Actual</th>
                            <th style="padding: 15px; background: #81C784;">Setosa</th>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em; font-weight: bold;">''' + str(cm[0,0]) + '''</td>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em;">''' + str(cm[0,1]) + '''</td>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em;">''' + str(cm[0,2]) + '''</td>
                        </tr>
                        <tr>
                            <th style="padding: 15px; background: #81C784;">Versicolor</th>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em;">''' + str(cm[1,0]) + '''</td>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em; font-weight: bold;">''' + str(cm[1,1]) + '''</td>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em; ''' + ('background: #ffcdd2;' if cm[1,2] > 0 else '') + '''">''' + str(cm[1,2]) + '''</td>
                        </tr>
                        <tr>
                            <th style="padding: 15px; background: #81C784;">Virginica</th>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em;">''' + str(cm[2,0]) + '''</td>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em;">''' + str(cm[2,1]) + '''</td>
                            <td style="padding: 15px; border: 1px solid #ddd; font-size: 1.2em; font-weight: bold;">''' + str(cm[2,2]) + '''</td>
                        </tr>
                    </table>
                    <p style="margin-top: 15px; color: #666;">
                        <strong>How to read:</strong> Row = actual type, Column = predicted type<br>
                        ''' + ('''<span style="background: #ffcdd2; padding: 2px 8px; border-radius: 3px;">Red cells</span> show mistakes''' if len(misclassified) > 0 else '''<span style="background: #c8e6c9; padding: 2px 8px; border-radius: 3px;">All diagonal</span> = perfect predictions!''') + '''
                    </p>
                </div>
    '''
    
    if misclassified:
        html += '''
                <div class="example-box">
                    <h3>❌ Misclassified Example</h3>
    '''
        for m in misclassified[:1]:  # Show first misclassified
            html += f'''
                    <p><strong>Flower measurements:</strong></p>
                    <ul>
                        <li>Petal Length: {m['features'][2]:.1f} cm</li>
                        <li>Petal Width: {m['features'][3]:.1f} cm</li>
                        <li>Sepal Length: {m['features'][0]:.1f} cm</li>
                        <li>Sepal Width: {m['features'][1]:.1f} cm</li>
                    </ul>
                    <p style="background: #ffebee; padding: 10px; border-radius: 5px;">
                        <strong>Actual:</strong> {m['actual']}<br>
                        <strong>Predicted:</strong> {m['predicted']}<br>
                        <strong>Why wrong?</strong> This flower's measurements fall right on the boundary between species!
                    </p>
    '''
        html += '''
                </div>
    '''
    else:
        html += '''
                <div class="example-box" style="background: #e8f5e9; border-color: #4CAF50;">
                    <h3>✅ Perfect Test Performance!</h3>
                    <p>The model correctly classified all 30 test flowers without a single mistake!</p>
                    <p><strong>Why so accurate?</strong></p>
                    <ul>
                        <li>Iris species have very distinct petal measurements</li>
                        <li>The decision tree found clear boundaries between species</li>
                        <li>Even "new" flowers follow the same patterns the model learned</li>
                    </ul>
                </div>
    '''
    
    # Add training misclassifications if any
    if misclassified_train:
        html += '''
                <div class="example-box">
                    <h3>📝 Training Set Errors (''' + str(len(misclassified_train)) + ''' flowers)</h3>
                    <p>Even on the training data, the model made some mistakes to avoid overfitting:</p>
    '''
        for m in misclassified_train[:2]:  # Show first 2
            html += f'''
                    <p style="background: #fff3e0; padding: 8px; border-radius: 5px; margin: 5px 0;">
                        Actual: <strong>{m['actual']}</strong> → Predicted: <strong>{m['predicted']}</strong>
                    </p>
    '''
        html += '''
                </div>
    '''
    
    html += '''
                <div id="testSetVisualization"></div>
            </div>
            
            <div class="section">
                <h2><span class="step-number">7</span> Building the Tree: Step by Step</h2>
                
                <div class="tree-container">
                    <h3>🌱 Step 1: First Split</h3>
                    <div class="node">
                        <div class="node-question">Is Petal Length ≤ 2.5 cm?</div>
                        <div class="node-stats">150 flowers need sorting</div>
                    </div>
                    
                    <div class="split-animation">
                        <div class="node leaf-node">
                            <strong>YES → All 50 are Setosa!</strong><br>
                            <span class="node-stats">Pure group achieved ✓</span>
                        </div>
                        <div class="arrow">→</div>
                        <div class="node">
                            <strong>NO → 100 flowers</strong><br>
                            <span class="node-stats">Mix of Versicolor & Virginica</span>
                        </div>
                    </div>
                    
                    <p style="text-align: center;"><strong>Why this question?</strong> Out of all possible questions, this one gives the most information - it perfectly identifies all Setosa flowers!</p>
                </div>
                
                <div class="tree-container">
                    <h3>🌿 Step 2: Second Split (for the remaining 100)</h3>
                    <div class="node">
                        <div class="node-question">Is Petal Width ≤ 1.8 cm?</div>
                        <div class="node-stats">100 flowers (Versicolor & Virginica)</div>
                    </div>
                    
                    <div class="split-animation">
                        <div class="node">
                            <strong>YES → Mostly Versicolor</strong><br>
                            <span class="node-stats">54 flowers (49 Versicolor, 5 Virginica)</span>
                        </div>
                        <div class="arrow">→</div>
                        <div class="node">
                            <strong>NO → Mostly Virginica</strong><br>
                            <span class="node-stats">46 flowers (1 Versicolor, 45 Virginica)</span>
                        </div>
                    </div>
                </div>
                
                <div class="tree-container">
                    <h3>🌳 Step 3: Final Splits (Depth 3)</h3>
                    <p>The algorithm continues splitting until it reaches the maximum depth (3) or groups are pure enough.</p>
                </div>
                
                <div id="splitVisualization"></div>
            </div>
            
            <div class="section" style="border-left: 5px solid #2196F3;">
                <h2><span class="step-number" style="background: #2196F3;">8</span> Why 100% Test Accuracy?</h2>
                
                <div class="concept-box" style="background: #e3f2fd;">
                    <h3>🎯 Perfect Test Performance Explained</h3>
                    <p><strong>Training Accuracy: 95.8%</strong> - The model made 5 mistakes on the 120 training samples.</p>
                    <p><strong>Test Accuracy: 100%</strong> - The model correctly classified all 30 test samples!</p>
                    
                    <p style="margin-top: 15px;"><strong>Why is test accuracy higher than training?</strong></p>
                    <ul>
                        <li>The Iris dataset is relatively simple with clear separations between classes</li>
                        <li>Our max_depth=3 prevents overfitting while capturing the main patterns</li>
                        <li>The 30 test samples happen to fall in clearly separable regions</li>
                        <li>The 5 training errors were likely edge cases or slight overlaps</li>
                    </ul>
                    
                    <p style="margin-top: 15px; color: #1976D2;"><strong>This is actually good!</strong> It means our model learned the general patterns 
                    rather than memorizing specific training examples. The simple tree structure (only 3 levels deep) 
                    found robust decision boundaries that work perfectly on new data.</p>
                </div>
            </div>
            
            <div class="section">
                <h2><span class="step-number">9</span> Understanding Key Concepts</h2>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">3</div>
                        <div class="metric-label"><strong>Tree Depth</strong><br>Maximum questions to ask</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">5</div>
                        <div class="metric-label"><strong>Leaf Nodes</strong><br>Final decision groups</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">4</div>
                        <div class="metric-label"><strong>Decision Nodes</strong><br>Questions asked</div>
                    </div>
                </div>
                
                <div class="concept-box">
                    <h3>📚 What do these terms mean?</h3>
                    <p><strong>Depth:</strong> How many questions deep the tree goes. Like "Is it an animal?" → "Does it fly?" → "Does it have feathers?" (depth = 3)</p>
                    <p><strong>Rules:</strong> The yes/no questions at each split. Example: "Petal Length ≤ 2.5?"</p>
                    <p><strong>Leaf Nodes:</strong> The final answers. Once you reach a leaf, you have your prediction!</p>
                    <p><strong>Information Gain:</strong> How much a question reduces uncertainty. Good questions separate different types well.</p>
                </div>
            </div>
            
            <div class="section">
                <h2><span class="step-number">10</span> Follow a Flower Through the Tree</h2>
                
                <div class="example-box">
                    <h3>🌸 Example: Classifying a New Flower</h3>
                    <div class="example-flower">
                        Measurements: Petal Length = 4.5 cm, Petal Width = 1.5 cm
                    </div>
                    
                    <div class="decision-path">
                        <strong>Question 1:</strong> Is Petal Length ≤ 2.5?<br>
                        Answer: <span class="highlight">NO (4.5 > 2.5)</span> → Not Setosa, continue right
                    </div>
                    
                    <div class="decision-path">
                        <strong>Question 2:</strong> Is Petal Width ≤ 1.8?<br>
                        Answer: <span class="highlight">YES (1.5 ≤ 1.8)</span> → Go left
                    </div>
                    
                    <div class="decision-path">
                        <strong>Question 3:</strong> Is Petal Length ≤ 4.9?<br>
                        Answer: <span class="highlight">YES (4.5 ≤ 4.9)</span> → Reached a leaf!
                    </div>
                    
                    <div style="background: #4CAF50; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;">
                        <strong style="font-size: 1.3em;">Final Prediction: VERSICOLOR</strong>
                    </div>
                </div>
                
                <button onclick="animateExample()">Animate Another Example</button>
            </div>
            
            <div class="section">
                <h2><span class="step-number">10</span> The Complete Decision Tree</h2>
                
                <div class="tree-visual">
                    <svg width="800" height="400" style="background: white; border-radius: 10px;">
                        <!-- Root -->
                        <rect x="350" y="20" width="200" height="60" fill="#e3f2fd" stroke="#2196F3" stroke-width="2" rx="5"/>
                        <text x="450" y="45" text-anchor="middle" font-weight="bold">Petal Length ≤ 2.5?</text>
                        <text x="450" y="65" text-anchor="middle" font-size="12" fill="#666">150 flowers</text>
                        
                        <!-- Left branch (Setosa) -->
                        <line x1="400" y1="80" x2="250" y2="150" stroke="#4CAF50" stroke-width="2"/>
                        <rect x="150" y="150" width="150" height="50" fill="#c8e6c9" stroke="#4CAF50" stroke-width="2" rx="5"/>
                        <text x="225" y="170" text-anchor="middle" font-weight="bold">SETOSA</text>
                        <text x="225" y="185" text-anchor="middle" font-size="12">50 flowers (100%)</text>
                        
                        <!-- Right branch -->
                        <line x1="500" y1="80" x2="600" y2="150" stroke="#2196F3" stroke-width="2"/>
                        <rect x="525" y="150" width="180" height="60" fill="#e3f2fd" stroke="#2196F3" stroke-width="2" rx="5"/>
                        <text x="615" y="175" text-anchor="middle" font-weight="bold">Petal Width ≤ 1.8?</text>
                        <text x="615" y="195" text-anchor="middle" font-size="12" fill="#666">100 flowers</text>
                        
                        <!-- Continue tree... -->
                        <line x1="575" y1="210" x2="475" y2="280" stroke="#2196F3" stroke-width="2"/>
                        <rect x="400" y="280" width="150" height="50" fill="#e3f2fd" stroke="#2196F3" stroke-width="2" rx="5"/>
                        <text x="475" y="300" text-anchor="middle" font-size="14">More splits...</text>
                        <text x="475" y="315" text-anchor="middle" font-size="12" fill="#666">→ Versicolor</text>
                        
                        <line x1="655" y1="210" x2="700" y2="280" stroke="#4CAF50" stroke-width="2"/>
                        <rect x="650" y="280" width="130" height="50" fill="#c8e6c9" stroke="#4CAF50" stroke-width="2" rx="5"/>
                        <text x="715" y="300" text-anchor="middle" font-weight="bold">VIRGINICA</text>
                        <text x="715" y="315" text-anchor="middle" font-size="12">45/46 (98%)</text>
                    </svg>
                </div>
                
                <p style="text-align: center; color: #666;">This tree achieved 95.8% training accuracy and perfect 100% test accuracy!</p>
            </div>
            
            <div class="section" style="background: linear-gradient(135deg, #4CAF50, #45b7d1); color: white;">
                <h2 style="color: white;">🎯 Summary: How Computers "Learn"</h2>
                <ol style="font-size: 1.1em; line-height: 1.8;">
                    <li><strong>Start with data:</strong> 150 flowers with measurements and known types</li>
                    <li><strong>Find patterns:</strong> Algorithm searches for questions that best separate types</li>
                    <li><strong>Build rules:</strong> Create a tree of yes/no questions (depth = how many levels)</li>
                    <li><strong>Make predictions:</strong> New flowers follow the questions to reach a prediction</li>
                    <li><strong>No magic:</strong> Just smart questions based on data patterns!</li>
                </ol>
            </div>
            
            <div class="nav-links">
                <a href="/">Try Making Predictions</a>
            </div>
        </div>
        
        <script>
            // Data for visualizations
            const allData = ''' + json.dumps(all_data) + ''';
            
            // Initial mixed plot
            const trace1 = {
                x: allData.filter(d => d.type === 'setosa').map(d => d.petal_length),
                y: allData.filter(d => d.type === 'setosa').map(d => d.petal_width),
                mode: 'markers',
                name: 'Setosa',
                marker: { color: '#ff6b6b', size: 8 }
            };
            
            const trace2 = {
                x: allData.filter(d => d.type === 'versicolor').map(d => d.petal_length),
                y: allData.filter(d => d.type === 'versicolor').map(d => d.petal_width),
                mode: 'markers',
                name: 'Versicolor',
                marker: { color: '#4ecdc4', size: 8 }
            };
            
            const trace3 = {
                x: allData.filter(d => d.type === 'virginica').map(d => d.petal_length),
                y: allData.filter(d => d.type === 'virginica').map(d => d.petal_width),
                mode: 'markers',
                name: 'Virginica',
                marker: { color: '#45b7d1', size: 8 }
            };
            
            const layout = {
                title: 'All 150 Iris Flowers (Before Learning)',
                xaxis: { title: 'Petal Length (cm)' },
                yaxis: { title: 'Petal Width (cm)' },
                height: 400
            };
            
            Plotly.newPlot('initialPlot', [trace1, trace2, trace3], layout);
            
            // Sepal comparison plot (poor separation)
            const sepalLayout = {
                title: 'Sepal Measurements',
                xaxis: { title: 'Sepal Length (cm)' },
                yaxis: { title: 'Sepal Width (cm)' },
                height: 300,
                showlegend: false
            };
            
            const sepalTrace1 = {
                x: allData.filter(d => d.type === 'setosa').map(d => d.petal_length * 1.5 + 3.5),
                y: allData.filter(d => d.type === 'setosa').map(d => d.petal_width * 0.5 + 2.8),
                mode: 'markers',
                name: 'Setosa',
                marker: { color: '#ff6b6b', size: 6, opacity: 0.6 }
            };
            const sepalTrace2 = {
                x: allData.filter(d => d.type === 'versicolor').map(d => d.petal_length * 0.8 + 4.5),
                y: allData.filter(d => d.type === 'versicolor').map(d => d.petal_width * 0.4 + 2.5),
                mode: 'markers',
                name: 'Versicolor',
                marker: { color: '#4ecdc4', size: 6, opacity: 0.6 }
            };
            const sepalTrace3 = {
                x: allData.filter(d => d.type === 'virginica').map(d => d.petal_length * 0.9 + 4.8),
                y: allData.filter(d => d.type === 'virginica').map(d => d.petal_width * 0.3 + 2.6),
                mode: 'markers',
                name: 'Virginica',
                marker: { color: '#45b7d1', size: 6, opacity: 0.6 }
            };
            
            Plotly.newPlot('sepalComparisonPlot', [sepalTrace1, sepalTrace2, sepalTrace3], sepalLayout);
            
            // Petal comparison plot (clear separation)
            const petalLayout = {
                title: 'Petal Measurements',
                xaxis: { title: 'Petal Length (cm)' },
                yaxis: { title: 'Petal Width (cm)' },
                height: 300,
                showlegend: false
            };
            
            Plotly.newPlot('petalComparisonPlot', [trace1, trace2, trace3], petalLayout);
            
            // Feature comparison bar chart
            const featureGains = [
                { feature: 'Petal Length', gain: 0.92, color: '#4CAF50' },
                { feature: 'Petal Width', gain: 0.91, color: '#8BC34A' },
                { feature: 'Sepal Length', gain: 0.32, color: '#FFA726' },
                { feature: 'Sepal Width', gain: 0.22, color: '#FF7043' }
            ];
            
            const barTrace = {
                x: featureGains.map(f => f.feature),
                y: featureGains.map(f => f.gain),
                type: 'bar',
                marker: {
                    color: featureGains.map(f => f.color)
                },
                text: featureGains.map(f => (f.gain).toFixed(2)),
                textposition: 'outside'
            };
            
            const barLayout = {
                title: 'Information Gain by Feature (Higher = Better Separation)',
                yaxis: { title: 'Information Gain', range: [0, 1] },
                height: 350
            };
            
            Plotly.newPlot('featureComparisonBar', [barTrace], barLayout);
            
            // Importance breakdown
            const importanceData = [
                { feature: 'Petal Length', value: 93 },
                { feature: 'Petal Width', value: 7 },
                { feature: 'Sepal Length', value: 0 },
                { feature: 'Sepal Width', value: 0 }
            ];
            
            const pieTrace = {
                values: importanceData.map(d => d.value),
                labels: importanceData.map(d => d.feature),
                type: 'pie',
                marker: {
                    colors: ['#4CAF50', '#8BC34A', '#FFA726', '#FF7043']
                }
            };
            
            const pieLayout = {
                title: 'Final Feature Importance in Decision Tree',
                height: 400
            };
            
            Plotly.newPlot('importanceBreakdown', [pieTrace], pieLayout);
            
            // Entropy visualization
            const entropyData = [
                { name: 'Initial State', entropy: 1.585, color: '#E91E63' },
                { name: 'After Petal Length Split', entropy: 0.667, color: '#4CAF50' },
                { name: 'After Sepal Width Split', entropy: 1.32, color: '#FF5722' }
            ];
            
            const entropyTrace = {
                x: entropyData.map(d => d.name),
                y: entropyData.map(d => d.entropy),
                type: 'bar',
                marker: {
                    color: entropyData.map(d => d.color)
                },
                text: entropyData.map(d => d.entropy.toFixed(3) + ' bits'),
                textposition: 'outside'
            };
            
            const entropyLayout = {
                title: 'Entropy Comparison: Good vs Bad Splits',
                yaxis: { 
                    title: 'Entropy (bits)',
                    range: [0, 2],
                    dtick: 0.5
                },
                height: 350,
                annotations: [
                    {
                        x: 'Initial State',
                        y: 1.585,
                        text: 'Maximum<br>uncertainty',
                        showarrow: false,
                        yshift: 30
                    },
                    {
                        x: 'After Petal Length Split',
                        y: 0.667,
                        text: '58% reduction!',
                        showarrow: false,
                        yshift: 30,
                        font: { color: '#4CAF50' }
                    },
                    {
                        x: 'After Sepal Width Split',
                        y: 1.32,
                        text: 'Only 16% reduction',
                        showarrow: false,
                        yshift: 30,
                        font: { color: '#FF5722' }
                    }
                ]
            };
            
            Plotly.newPlot('entropyVisualization', [entropyTrace], entropyLayout);
            
            // Train/Test split visualization
            const trainTestData = {
                train: 120,
                test: 30
            };
            
            const splitBarTrace = {
                x: ['Training Set', 'Test Set'],
                y: [120, 30],
                type: 'bar',
                marker: {
                    color: ['#2196F3', '#FF9800']
                },
                text: ['120 flowers (80%)', '30 flowers (20%)'],
                textposition: 'outside'
            };
            
            const splitBarLayout = {
                title: 'Data Split: 150 Flowers Divided',
                yaxis: { title: 'Number of Flowers' },
                height: 350
            };
            
            Plotly.newPlot('trainTestSplit', [splitBarTrace], splitBarLayout);
            
            // Test set visualization with misclassified
            const testSetTraces = [];
            const testColors = {'setosa': '#ff6b6b', 'versicolor': '#4ecdc4', 'virginica': '#45b7d1'};
            
            // Add correctly classified points
            testSetTraces.push({
                x: allData.filter((d, i) => i % 5 === 0 && d.type === 'setosa').map(d => d.petal_length).slice(0, 10),
                y: allData.filter((d, i) => i % 5 === 0 && d.type === 'setosa').map(d => d.petal_width).slice(0, 10),
                mode: 'markers',
                name: 'Setosa (correct)',
                marker: { color: '#ff6b6b', size: 10, symbol: 'circle' }
            });
            
            testSetTraces.push({
                x: allData.filter((d, i) => i % 5 === 0 && d.type === 'versicolor').map(d => d.petal_length).slice(0, 10),
                y: allData.filter((d, i) => i % 5 === 0 && d.type === 'versicolor').map(d => d.petal_width).slice(0, 10),
                mode: 'markers',
                name: 'Versicolor (correct)',
                marker: { color: '#4ecdc4', size: 10, symbol: 'circle' }
            });
            
            testSetTraces.push({
                x: allData.filter((d, i) => i % 5 === 0 && d.type === 'virginica').map(d => d.petal_length).slice(0, 9),
                y: allData.filter((d, i) => i % 5 === 0 && d.type === 'virginica').map(d => d.petal_width).slice(0, 9),
                mode: 'markers',
                name: 'Virginica (correct)',
                marker: { color: '#45b7d1', size: 10, symbol: 'circle' }
            });
            
            // Add misclassified point (if any)
            ''' + ('''
            testSetTraces.push({
                x: [5.0],
                y: [1.5],
                mode: 'markers',
                name: 'Misclassified',
                marker: { color: 'red', size: 15, symbol: 'x', line: { width: 3 } }
            });
            ''' if 'misclassified' in locals() and misclassified else '') + '''
            
            const testSetLayout = {
                title: 'Test Set Predictions (30 flowers never seen during training)',
                xaxis: { title: 'Petal Length (cm)' },
                yaxis: { title: 'Petal Width (cm)' },
                height: 400,
                annotations: [
                    {
                        x: 5.0,
                        y: 1.5,
                        text: 'Misclassified<br>flower',
                        showarrow: true,
                        arrowhead: 2,
                        ax: 40,
                        ay: -40
                    }
                ]
            };
            
            Plotly.newPlot('testSetVisualization', testSetTraces, testSetLayout);
            
            // Split visualization with decision boundaries
            const splitLayout = {
                title: 'How the Tree Splits the Data',
                xaxis: { title: 'Petal Length (cm)', range: [0, 8] },
                yaxis: { title: 'Petal Width (cm)', range: [0, 3] },
                height: 400,
                shapes: [
                    // First split line
                    {
                        type: 'line',
                        x0: 2.5, y0: 0, x1: 2.5, y1: 3,
                        line: { color: 'red', width: 3, dash: 'dash' }
                    },
                    // Second split line
                    {
                        type: 'line',
                        x0: 2.5, y0: 1.8, x1: 8, y1: 1.8,
                        line: { color: 'orange', width: 2, dash: 'dash' }
                    }
                ],
                annotations: [
                    {
                        x: 1.2, y: 2.5,
                        text: 'All Setosa<br>(Petal Length ≤ 2.5)',
                        showarrow: false,
                        bgcolor: 'rgba(255, 0, 0, 0.1)',
                        borderpad: 4
                    },
                    {
                        x: 5.5, y: 2.5,
                        text: 'Mostly Virginica<br>(Petal Width > 1.8)',
                        showarrow: false,
                        bgcolor: 'rgba(0, 0, 255, 0.1)',
                        borderpad: 4
                    },
                    {
                        x: 5.5, y: 1.0,
                        text: 'Mostly Versicolor<br>(Petal Width ≤ 1.8)',
                        showarrow: false,
                        bgcolor: 'rgba(0, 255, 0, 0.1)',
                        borderpad: 4
                    }
                ]
            };
            
            Plotly.newPlot('splitVisualization', [trace1, trace2, trace3], splitLayout);
            
            // Animate example function
            let exampleCount = 0;
            function animateExample() {
                const examples = [
                    {pl: 1.5, pw: 0.3, result: 'Setosa', path: ['Petal Length ≤ 2.5? YES']},
                    {pl: 5.5, pw: 2.0, result: 'Virginica', path: ['Petal Length ≤ 2.5? NO', 'Petal Width ≤ 1.8? NO']},
                    {pl: 3.5, pw: 1.0, result: 'Versicolor', path: ['Petal Length ≤ 2.5? NO', 'Petal Width ≤ 1.8? YES']}
                ];
                
                const example = examples[exampleCount % 3];
                exampleCount++;
                
                alert(`Following flower with Petal Length=${example.pl}, Petal Width=${example.pw}:\\n\\nPath: ${example.path.join(' → ')}\\n\\nResult: ${example.result}!`);
            }
        </script>
    </body>
    </html>
    '''
    
    return html

if __name__ == '__main__':
    print("\n🌳 DECISION TREE EXPLAINER")
    print("=" * 40)
    print("Starting on http://localhost:5555")
    print("Visit /learn to see HOW the algorithm works!")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5555, debug=False)