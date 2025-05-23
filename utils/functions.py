import aiohttp
from typing import List, Dict, Any

async def fetch_openrouter_models() -> List[Dict[str, Any]]:
    """
    Asynchronously fetch available models from the OpenRouter API.
        
    Returns:
        A list of dictionaries containing model information.
        Empty list if the request fails.
    """
    # Define the endpoint
    models_endpoint = "https://openrouter.ai/api/v1/models"
    
    try:
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make the async request
        async with aiohttp.ClientSession() as session:
            async with session.get(models_endpoint, headers=headers) as response:
                # Check if request was successful
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    error_text = await response.text()
                    print(f"Error fetching models: HTTP {response.status} - {error_text}")
                    return []
            
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return []
    

async def list_openrouter_models_summary() -> List[Dict[str, Any]]:
    """
    Asynchronously returns a simplified list of available OpenRouter models with key information.
           
    Returns:
        A list of dictionaries with model ID, name, context length, and pricing.
    """
    models = await fetch_openrouter_models()
    
    summary = []
    for model in models:
        summary.append({
            "id": model.get("id"),
            "name": model.get("name"),
            "context_length": model.get("context_length"),
            "modality": model.get("architecture", {}).get("modality"),
            "pricing": {
                "prompt": model.get("pricing", {}).get("prompt"),
                "completion": model.get("pricing", {}).get("completion")
            }
        })
        
    return summary