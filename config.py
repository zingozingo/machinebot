"""
Simple configuration file
"""

# Flask settings
PORT = 5555
HOST = '0.0.0.0'
DEBUG = False

# Model settings
MODEL_PATH = 'models/simple_model.pkl'

# Flower types
FLOWER_NAMES = ['setosa', 'versicolor', 'virginica']

# Feature names
FEATURES = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']