import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("app/keys/cement-dashboard-2025-78acd-firebase-adminsdk-fbsvc-5bbdf22113.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def push_prediction(sensor_id: str, data: dict, prediction: dict, alerts: list):
    doc_ref = db.collection("cement_predictions").document(sensor_id)
    doc_ref.set({
        "sensor_data": data,
        "prediction": prediction,
        "alerts": alerts,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
