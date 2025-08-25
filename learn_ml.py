#!/usr/bin/env python3
"""
SIMPLE ML DEMO - Learn machine learning in 50 lines!
Run: python learn_ml.py
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

print("\n=== SIMPLE MACHINE LEARNING DEMO ===\n")

# 1. LOAD DATA
iris = load_iris()
X, y = iris.data, iris.target
flower_names = iris.target_names

print("1. Here are 5 flowers from our data:")
print("   Sepal-L  Sepal-W  Petal-L  Petal-W  --> Type")
for i in range(5):
    print(f"   {X[i,0]:.1f}     {X[i,1]:.1f}     {X[i,2]:.1f}     {X[i,3]:.1f}     --> {flower_names[y[i]]}")

# 2. TRAIN MODEL (3 lines!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

print(f"\n2. Trained model on {len(X_train)} flowers!")

# 3. TEST ON 1 FLOWER
test_flower = X_test[0]
prediction = model.predict([test_flower])[0]
actual = y_test[0]

print(f"\n3. Testing on 1 flower:")
print(f"   Measurements: {test_flower}")
print(f"   Model predicted: {flower_names[prediction]}")
print(f"   Actually was: {flower_names[actual]}")
print(f"   Result: {'✓ CORRECT!' if prediction == actual else '✗ WRONG'}")

# Save model for Flask app
with open('models/simple_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n4. Model saved for web app!")

# 4. LET USER PREDICT
print("\n5. YOUR TURN - Predict a flower type!")
print("   Enter measurements (or 'quit'):\n")

while True:
    try:
        user_input = input("   Sepal Length (4-8): ")
        if user_input.lower() == 'quit':
            break
        sl = float(user_input)
        sw = float(input("   Sepal Width (2-4.5): "))
        pl = float(input("   Petal Length (1-7): "))
        pw = float(input("   Petal Width (0.1-2.5): "))
        
        prediction = model.predict([[sl, sw, pl, pw]])[0]
        print(f"\n   --> This is a {flower_names[prediction].upper()}!\n")
    except:
        print("   Invalid input, try again\n")
        
print("\nDone! Run 'python app.py' to start the web version.")