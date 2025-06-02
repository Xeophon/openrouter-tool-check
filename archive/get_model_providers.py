#!/usr/bin/env python3
"""
Script to get the list of providers that support a specific model on OpenRouter.

This demonstrates the exact API calls and response formats for extracting provider information.
"""

import os
import requests
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

def get_model_endpoints(model_id: str) -> Dict[str, Any]:
    """
    Get the list of endpoints (providers) for a specific model.
    
    Args:
        model_id: The model ID in format "author/model-name"
        
    Returns:
        Dict containing model information and available endpoints
    """
    # Split the model ID to get author and slug
    parts = model_id.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid model ID format: {model_id}. Expected 'author/slug'")
    
    author, slug = parts
    
    # Construct the endpoint URL
    url = f"{BASE_URL}/models/{author}/{slug}/endpoints"
    
    # Make the request
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Fetching endpoints for model: {model_id}")
    print(f"URL: {url}")
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        print(f"Response: {response.text}")
        return {}
    
    return response.json()

def extract_provider_names(endpoints_data: Dict[str, Any]) -> List[str]:
    """
    Extract provider names from the endpoints response.
    
    Args:
        endpoints_data: The response from the endpoints API
        
    Returns:
        List of provider names
    """
    providers = []
    
    if "data" in endpoints_data and "endpoints" in endpoints_data["data"]:
        for endpoint in endpoints_data["data"]["endpoints"]:
            if "provider_name" in endpoint:
                providers.append(endpoint["provider_name"])
    
    return providers

def format_provider_for_api(provider_name: str) -> str:
    """
    Format a provider name for use in API requests.
    
    Based on the documentation, providers can be specified in the request body.
    """
    # Provider names are typically lowercase
    return provider_name.lower()

def main():
    """Main function to demonstrate getting providers for a model."""
    
    # The model we're interested in
    model_id = "qwen/qwen3-30b-a3b"
    
    # Get endpoints data
    endpoints_data = get_model_endpoints(model_id)
    
    if not endpoints_data:
        print("Failed to get endpoints data")
        return
    
    # Print the full response for reference
    print("\nFull API Response:")
    print("-" * 80)
    import json
    print(json.dumps(endpoints_data, indent=2))
    print("-" * 80)
    
    # Extract provider names
    providers = extract_provider_names(endpoints_data)
    
    print(f"\nProviders supporting {model_id}:")
    for provider in providers:
        print(f"  - {provider}")
    
    # Show how to use providers in API requests
    print("\nProvider Usage in API Requests:")
    print("-" * 80)
    
    for provider in providers:
        formatted_provider = format_provider_for_api(provider)
        example_request = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "provider": {
                "order": [formatted_provider],  # Prioritize this provider
                "only": [formatted_provider]     # Or use only this provider
            }
        }
        
        print(f"\nExample request for provider '{provider}':")
        print(json.dumps(example_request, indent=2))
    
    # Alternative: List all models and filter
    print("\n\nAlternative Method: List all models")
    print("-" * 80)
    
    models_url = f"{BASE_URL}/models"
    response = requests.get(models_url, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
    
    if response.status_code == 200:
        models_data = response.json()
        
        # Find our specific model
        for model in models_data.get("data", []):
            if model.get("id") == model_id:
                print(f"\nModel found in /models endpoint:")
                print(f"  ID: {model.get('id')}")
                print(f"  Name: {model.get('name')}")
                print(f"  Top Provider Moderated: {model.get('top_provider', {}).get('is_moderated')}")
                print(f"  Supported Parameters: {model.get('supported_parameters', [])}")
                break

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenRouter API key")
    else:
        main()