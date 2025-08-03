from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/api/prices')
def get_prices():
    df = pd.read_csv('../data/processed/merged_data.csv')
    return jsonify(df.to_dict('records'))

@app.route('/api/events')
def get_events():
    events = pd.read_csv('../data/raw/events.csv')
    return jsonify(events.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)