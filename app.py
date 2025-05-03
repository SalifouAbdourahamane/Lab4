from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

app = Flask(__name__)

# Global variables
models = {}
datasets = {}
current_model_id = 0
current_dataset_id = 0

# Add this right after your global variables initialization
def initialize_default_dataset():
    global current_dataset_id, datasets
    try:
        df = pd.read_csv("iris_extended_encoded.csv")
        le = LabelEncoder()
        features = df.iloc[:, 1:].values
        labels = le.fit_transform(df.iloc[:, 0].values)
        
        datasets[0] = {
            'features': features,
            'labels': labels,
            'label_encoder': le
        }
        current_dataset_id = 1  # Next dataset will be ID 1
    except Exception as e:
        print(f"Couldn't load default dataset: {str(e)}")

# Call this before starting the app
initialize_default_dataset()

@app.route('/iris/datasets', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file.save('temp.csv')
    df = pd.read_csv('temp.csv')
    
    global current_dataset_id
    dataset_id = current_dataset_id
    current_dataset_id += 1
    
    # Store dataset
    le = LabelEncoder()
    features = df.iloc[:, 1:].values
    labels = le.fit_transform(df.iloc[:, 0].values)
    
    datasets[dataset_id] = {
        'features': features,
        'labels': labels,
        'label_encoder': le
    }
    
    return jsonify({'dataset_id': dataset_id}), 201

@app.route('/iris/model/new', methods=['POST'])
def create_model():
    dataset_id = int(request.form.get('dataset', 0))
    
    if dataset_id not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_dim=20),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # Train model
    dataset = datasets[dataset_id]
    history = model.fit(dataset['features'], dataset['labels'], epochs=10, verbose=0)
    
    # Store model
    global current_model_id
    model_id = current_model_id
    current_model_id += 1
    
    models[model_id] = {
        'model': model,
        'history': history.history
    }
    
    return jsonify({
        'model_id': model_id,
        'history': history.history
    }), 201

@app.route('/iris/model/<int:model_id>/train', methods=['PUT'])
def train_model(model_id):
    dataset_id = int(request.args.get('dataset', 0))
    
    if model_id not in models:
        return jsonify({'error': 'Model not found'}), 404
    if dataset_id not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    model = models[model_id]['model']
    dataset = datasets[dataset_id]
    
    history = model.fit(dataset['features'], dataset['labels'], epochs=10, verbose=0)
    models[model_id]['history'] = history.history
    
    return jsonify(history.history), 200
@app.route('/iris/model/<int:model_id>/predict', methods=['POST'])
def predict(model_id):
    if model_id not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    prediction = models[model_id]['model'].predict(features)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    
    # Get class name
    le = datasets[0]['label_encoder']  # Using first dataset's encoder
    class_name = le.inverse_transform([predicted_class])[0]
    
    return jsonify({
        'prediction': predicted_class,
        'species': str(class_name),
        'probabilities': prediction[0].tolist()
    })

@app.route('/iris/model/<int:model_id>/test', methods=['GET'])
def test_model(model_id):
    dataset_id = int(request.args.get('dataset', 0))
    
    if model_id not in models:
        return jsonify({'error': 'Model not found'}), 404
    if dataset_id not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    model = models[model_id]['model']
    dataset = datasets[dataset_id]
    
    # Evaluate model
    loss, accuracy = model.evaluate(dataset['features'], dataset['labels'], verbose=0)
    
    # Make predictions
    y_pred = model.predict(dataset['features'])
    predicted = np.argmax(y_pred, axis=1)
    
    return jsonify({
        'accuracy': float(accuracy),
        'loss': float(loss),
        'actual': dataset['labels'].tolist(),
        'predicted': predicted.tolist()
    })

if __name__ == '__main__':
    app.run(port=4000, debug=True)