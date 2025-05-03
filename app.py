import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask, jsonify, request
import sys
import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import json

app = Flask(__name__)

# AWS Configuration
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_REGION = 'eu-north-1'  
PRIMARY_TABLE_NAME = 'IrisExtended'
RETRAIN_TABLE_NAME = 'IrisExtendedRetrain'

# Initialize DynamoDB resources
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
primary_table = dynamodb.Table(PRIMARY_TABLE_NAME)
retrain_table = dynamodb.Table(RETRAIN_TABLE_NAME)

# Global storage
models = []
datasets = []
current_model_id = 0
current_dataset_id = 0

def load_dataset(filepath='iris_extended_encoded.csv'):
    global current_dataset_id
    print(f"Loading dataset from {filepath}", file=sys.stdout, flush=True)
    try:
        df = pd.read_csv(filepath, header=0)
        features = df.iloc[:, 1:21].values
        labels = df.iloc[:, 0].values
        feature_columns = df.columns[1:21].tolist()
        feature_data = df.iloc[:, 1:21].to_dict('records')
        
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        dataset_id = current_dataset_id
        current_dataset_id += 1
        
        dataset = {
            'features': features,
            'labels': encoded_labels,
            'label_encoder': le,
            'name': os.path.basename(filepath),
            'id': dataset_id,
            'feature_columns': feature_columns,
            'feature_data': feature_data,
            'original_labels': labels
        }
        
        datasets.append(dataset)
        print(f"Dataset loaded with ID: {dataset_id}", file=sys.stdout, flush=True)
        return dataset_id
    except Exception as e:
        print(f"Error loading dataset: {str(e)}", file=sys.stderr, flush=True)
        return None

def build_model(input_dim=20, output_classes=3):
    print("Building model", file=sys.stdout, flush=True)
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    print("Model built", file=sys.stdout, flush=True)
    return model

def train_model(model, dataset_id, epochs=10):
    dataset = next((d for d in datasets if d['id'] == dataset_id), None)
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    print(f"Training model on dataset {dataset_id}", file=sys.stdout, flush=True)
    model.fit(dataset['features'], dataset['labels'], epochs=epochs, verbose=0)
    print("Training completed", file=sys.stdout, flush=True)
    return model

def post_score(table, feature_string, class_string, actual_string, prob_string):
    current_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    
    try:
        response = table.put_item(
           Item={
                'partition_key': current_time,
                'sort_key': "abc",
                'Features': feature_string,
                'Class': class_string,
                'Actual': actual_string,
                'Probability': prob_string
            }
        )
        return response
    except ClientError as e:
        print(f"Error posting to DynamoDB: {e.response['Error']['Message']}")
        return None

@app.route('/iris/datasets', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    temp_path = f"temp_{current_dataset_id}.csv"
    file.save(temp_path)
    
    dataset_id = load_dataset(temp_path)
    os.remove(temp_path)
    
    if dataset_id is None:
        return jsonify({'error': 'Failed to load dataset'}), 400
    
    return jsonify({
        'dataset_id': dataset_id,
        'message': 'Dataset uploaded successfully'
    }), 201

@app.route('/iris/model', methods=['POST'])
def create_model():
    global current_model_id
    
    dataset_id = int(request.form.get('dataset', 0))
    model_id = current_model_id
    current_model_id += 1
    
    model = build_model()
    
    try:
        model = train_model(model, dataset_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    models.append({
        'id': model_id,
        'model': model,
        'dataset_id': dataset_id,
        'test_results': None
    })
    
    return jsonify({
        'model_id': model_id,
        'dataset_id': dataset_id,
        'message': 'Model created and trained'
    }), 201

@app.route('/iris/model/<int:model_id>/train', methods=['PUT'])
def retrain_model(model_id):
    dataset_id = int(request.args.get('dataset', 0))
    
    model_data = next((m for m in models if m['id'] == model_id), None)
    if not model_data:
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        model_data['model'] = train_model(model_data['model'], dataset_id)
        model_data['dataset_id'] = dataset_id
        return jsonify({
            'model_id': model_id,
            'dataset_id': dataset_id,
            'message': 'Model retrained successfully'
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/iris/model/<int:model_id>/test', methods=['GET'])
def test_model(model_id):
    dataset_id = int(request.args.get('dataset', 0))
    
    model_data = next((m for m in models if m['id'] == model_id), None)
    if not model_data:
        return jsonify({'error': 'Model not found'}), 404
    
    dataset = next((d for d in datasets if d['id'] == dataset_id), None)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    X_test = dataset['features']
    y_test = dataset['labels']
    
    loss, accuracy = model_data['model'].evaluate(X_test, y_test, verbose=0)
    y_pred = model_data['model'].predict(X_test)
    predicted = np.argmax(y_pred, axis=1)
    
    le = dataset['label_encoder']
    actual_labels = le.inverse_transform(y_test)
    predicted_labels = le.inverse_transform(predicted)
    original_labels = dataset['original_labels']
    
    # Log each test record to DynamoDB
    for i in range(len(X_test)):
        feature_dict = dict(zip(dataset['feature_columns'], X_test[i]))
        feature_string = str(feature_dict)
        class_string = predicted_labels[i]
        actual_string = original_labels[i]  # Use original label before encoding
        prob_string = str(float(np.max(y_pred[i])))
        
        # Post to primary table
        post_score(primary_table, feature_string, class_string, actual_string, prob_string)
    
    test_results = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'actual': y_test.tolist(),
        'actual_labels': actual_labels.tolist(),
        'predicted': predicted.tolist(),
        'predicted_labels': predicted_labels.tolist(),
        'dataset_id': dataset_id,
        'model_id': model_id
    }
    
    model_data['test_results'] = test_results
    return jsonify(test_results), 200

@app.route('/iris/model/<int:model_id>/predict', methods=['GET'])
def predict(model_id):
    fields = list(map(float, request.args.get('fields').split(',')))
    if len(fields) != 20:
        return jsonify({'error': 'Exactly 20 features required'}), 400
    
    model_data = next((m for m in models if m['id'] == model_id), None)
    if not model_data:
        return jsonify({'error': 'Model not found'}), 404
    
    features = np.array(fields).reshape(1, -1)
    prediction = model_data['model'].predict(features)
    predicted_class = int(np.argmax(prediction))
    
    if datasets:
        le = datasets[0]['label_encoder']
        class_name = le.inverse_transform([predicted_class])[0]
    else:
        class_name = str(predicted_class)
    
    return jsonify({
        'prediction': predicted_class,
        'species': class_name,
        'probabilities': prediction[0].tolist()
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    print(f"Starting server on port {port}", file=sys.stdout, flush=True)
    app.run(host='0.0.0.0', port=port)
