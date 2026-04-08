import pickle
import os

# Load model
model_path = os.path.join("models", "crop_model.pkl")
model = pickle.load(open(model_path, "rb"))

def crop_tool(input_data: str):
    try:
        values = list(map(float, input_data.split(",")))
        prediction = model.predict([values])
        return f"🌾 Recommended crop: {prediction[0]}"
    except Exception as e:
        return f"Error: {str(e)}"