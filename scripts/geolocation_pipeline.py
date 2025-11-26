import os
import re
import json
import math
import time
import base64
import openai
import google.generativeai as genai
from PIL import Image

# Configuration
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset")
MAP_FILE = os.path.join(DATASET_DIR, "location.json")
RESULTS_FILE = "experiment_results.json"


OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Model Mapping
MODEL_MAPPING = {
    "Gemini 2.5 Pro": "gemini-2.5-pro", 
    "GPT-5": "gpt-5"              
}

PROMPT_TEXT = """
Analyze the attached image to determine its geographic location. Follow these steps in your reasoning:
1. Initial Observation: Describe the overall scene. Is it urban or rural? What is the climate like?
2. Identify Key Clues: Look for specific, identifiable features (Language, Architecture, Flora/Fauna, Vehicles, Landscape).
3. Synthesize and Hypothesize: Based on the clues, form a hypothesis about the country, region, and city.
4. Final Conclusion: State your final conclusion for the location, providing the most precise coordinates you can determine.
"""

def load_ground_truth(file_path):
    """Reads the location.json file and returns a dictionary of ground truth coordinates."""
    if not os.path.exists(file_path):
        print(f"Warning: Map file not found at {file_path}")
        return {}

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}")
            return {}

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on Earth."""
    R = 6371.0  # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def extract_coordinates_from_response(text):
    """Extracts coordinates from the model's text response using Regex."""
    # Look for decimal pairs like "12.34, 56.78"
    decimal_pattern = r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)"
    matches = re.findall(decimal_pattern, text)
    
    if matches:
        # Return the last match as it's likely the "Final Conclusion"
        lat, lon = matches[-1]
        return float(lat), float(lon)
    
    return None, None

def encode_image(image_path):
    """Encodes an image to base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_openai_api(model_name, image_path, prompt):
    """Calls the OpenAI API with the specified model and image."""
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("Error: OpenAI API Key not set.")
        return "Error: API Key missing."

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    base64_image = encode_image(image_path)
    
    real_model = MODEL_MAPPING.get(model_name, "gpt-4-turbo")

    try:
        response = client.chat.completions.create(
            model=real_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return f"Error: {e}"

def call_gemini_api(model_name, image_path, prompt):
    """Calls the Google Gemini API with the specified model and image."""
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("Error: Gemini API Key not set.")
        return "Error: API Key missing."

    genai.configure(api_key=GEMINI_API_KEY)
    real_model = MODEL_MAPPING.get(model_name, "gemini-1.5-pro")
    
    try:
        model = genai.GenerativeModel(real_model)
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Error: {e}"

def process_image(model_name, image_path, prompt):
    """Dispatches the request to the appropriate API handler."""
    print(f"[{model_name}] Processing {os.path.basename(image_path)}...")
    
    if "GPT" in model_name:
        return call_openai_api(model_name, image_path, prompt)
    elif "Gemini" in model_name:
        return call_gemini_api(model_name, image_path, prompt)
    else:
        return "Error: Unknown model type."

def run_experiment():
    print("Starting Geolocation Experiment...")
    
    # 1. Load Ground Truth
    ground_truth = load_ground_truth(MAP_FILE)
    print(f"Loaded {len(ground_truth)} ground truth locations from {MAP_FILE}.")
    
    results = []
    
    # 2. Iterate through dataset
    for img_id, location_data in ground_truth.items():
        true_lat = location_data.get("lat")
        true_lon = location_data.get("lon")
        
        if true_lat is None or true_lon is None:
            continue

        # Find image file (handle jpg/jpeg)
        img_filename = f"{img_id}.jpeg"
        if not os.path.exists(os.path.join(DATASET_DIR, img_filename)):
             img_filename = f"{img_id}.jpg"
        
        img_path = os.path.join(DATASET_DIR, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Image {img_filename} not found, skipping.")
            continue
            
        # 3. Query Models
        models = ["Gemini 2.5 Pro", "GPT-5"]
        
        for model in models:
            response_text = process_image(model, img_path, PROMPT_TEXT)
            
            # 4. Parse Response
            pred_lat, pred_lon = extract_coordinates_from_response(response_text)
            
            error_km = None
            if pred_lat is not None and pred_lon is not None:
                # 5. Calculate Error
                error_km = haversine(true_lat, true_lon, pred_lat, pred_lon)
            
            # 6. Log Result
            result_entry = {
                "image_id": img_id,
                "model": model,
                "true_location": {"lat": true_lat, "lon": true_lon},
                "predicted_location": {"lat": pred_lat, "lon": pred_lon},
                "error_km": error_km,
                "response_text": response_text.strip() if response_text else ""
            }
            results.append(result_entry)
            print(f"  -> {model}: Error = {error_km:.2f} km" if error_km is not None else f"  -> {model}: Could not extract coordinates.")
            
            # Rate limiting / politeness
            time.sleep(1)

    # 7. Save Results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Experiment complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiment()
