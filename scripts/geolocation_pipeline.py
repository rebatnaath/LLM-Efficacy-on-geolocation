import os
import re
import json
import math
import time
import base64
import openai  
import google.generativeai as genai 

# Configuration
DATASET_DIR = os.path.abspath("../dataset")
MAP_FILE = os.path.join(DATASET_DIR, "map.md")
RESULTS_FILE = "experiment_results.json"

PROMPT_TEXT = """
Analyze the attached image to determine its geographic location. Follow these steps in your reasoning:
1. Initial Observation: Describe the overall scene. Is it urban or rural? What is the climate like?
2. Identify Key Clues: Look for specific, identifiable features (Language, Architecture, Flora/Fauna, Vehicles, Landscape).
3. Synthesize and Hypothesize: Based on the clues, form a hypothesis about the country, region, and city.
4. Final Conclusion: State your final conclusion for the location, providing the most precise coordinates you can determine.
"""

def dms_to_decimal(dms_str):
    """Converts DMS string (e.g., 52°20'09.6"N) to decimal degrees."""
    dms_str = dms_str.strip()
    regex = r"(\d+)°(\d+)'([\d.]+)\"([NSEW])"
    match = re.match(regex, dms_str)
    if not match:
        return None
    
    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def parse_coordinate_string(coord_str):
    """Parses a coordinate string which might be decimal or DMS."""
    coord_str = coord_str.strip()
    # Try simple decimal first
    try:
        return float(coord_str)
    except ValueError:
        pass
    
    # Try DMS
    return dms_to_decimal(coord_str)

def parse_ground_truth(file_path):
    """Reads the map.md file and returns a dictionary of ground truth coordinates."""
    ground_truth = {}
    if not os.path.exists(file_path):
        print(f"Warning: Map file not found at {file_path}")
        return ground_truth

    with open(file_path, 'r') as f:
        for line in f:
            if "->" in line:
                parts = line.split("->")
                img_id = parts[0].strip()
                coords_part = parts[1].strip()
                
                # Split lat, lon
                # Handle cases like "lat, lon" or "lat lon"
                if "," in coords_part:
                    lat_str, lon_str = coords_part.split(",", 1)
                else:
                    # Fallback split by space if no comma
                    tokens = coords_part.split()
                    if len(tokens) >= 2:
                        lat_str, lon_str = tokens[0], tokens[1]
                    else:
                        continue

                lat = parse_coordinate_string(lat_str)
                lon = parse_coordinate_string(lon_str)
                
                if lat is not None and lon is not None:
                    ground_truth[img_id] = (lat, lon)
    return ground_truth

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
    # Pattern for "Latitude: 12.34, Longitude: 56.78" or similar
    # This is a simplified pattern; models can be unpredictable.
    # Look for decimal pairs
    decimal_pattern = r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)"
    matches = re.findall(decimal_pattern, text)
    
    if matches:
        # Return the last match as it's likely the "Final Conclusion"
        lat, lon = matches[-1]
        return float(lat), float(lon)
    
    # Add patterns for DMS if needed, but models usually output decimal if asked or standard format
    return None, None

def mock_api_call(model_name, image_path, prompt):
    """
    Mock API call for demonstration. 
    Replace this with actual API calls to OpenAI or Google.
    """
    print(f"[{model_name}] Processing {os.path.basename(image_path)}...")
    time.sleep(1) # Simulate latency
    
    # Return a fake response for testing
    return f"Based on the architecture, this is likely the UK. Final Coordinates: 53.605, -1.958"

def run_experiment():
    print("Starting Geolocation Experiment...")
    
    # 1. Load Ground Truth
    ground_truth = parse_ground_truth(MAP_FILE)
    print(f"Loaded {len(ground_truth)} ground truth locations.")
    
    results = []
    
    # 2. Iterate through dataset
    # Assuming images are named 1.jpeg, 2.jpeg etc matching IDs in map.md
    for img_id, (true_lat, true_lon) in ground_truth.items():
        # Find image file (handle jpg/jpeg)
        img_filename = f"{img_id}.jpeg"
        if not os.path.exists(os.path.join(DATASET_DIR, img_filename)):
             img_filename = f"{img_id}.jpg"
        
        img_path = os.path.join(DATASET_DIR, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Image {img_filename} not found, skipping.")
            continue
            
        # 3. Query Models (Mocked here)
        models = ["Gemini 2.5 Pro", "GPT-5"]
        
        for model in models:
            response_text = mock_api_call(model, img_path, PROMPT_TEXT)
            
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
                "response_text": response_text
            }
            results.append(result_entry)
            print(f"  -> {model}: Error = {error_km:.2f} km" if error_km is not None else f"  -> {model}: Could not extract coordinates.")

    # 7. Save Results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Experiment complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiment()
