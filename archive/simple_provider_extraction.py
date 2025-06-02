#!/usr/bin/env python3
"""
Simple example showing the EXACT way to extract provider names for a model.
"""

import requests
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_provider_names_for_model(model_id: str) -> list[str]:
    """
    Get the exact provider names that can be used in API requests for a model.
    
    Args:
        model_id: Model identifier like "qwen/qwen3-30b-a3b"
        
    Returns:
        List of provider names that can be used in the 'provider' field
    """
    # Split model ID
    author, slug = model_id.split("/")
    
    # Make API request
    url = f"https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return []
    
    # Extract provider names from response
    data = response.json()
    provider_names = []
    
    for endpoint in data["data"]["endpoints"]:
        # The 'provider_name' field is what you use in API requests
        provider_names.append(endpoint["provider_name"])
    
    return provider_names

# Example usage
if __name__ == "__main__":
    model = "qwen/qwen3-30b-a3b"
    providers = get_provider_names_for_model(model)
    
    print(f"Providers for {model}:")
    for provider in providers:
        print(f"  - {provider}")
    
    print("\nTo test a specific provider, use:")
    print(f'''
    request_body = {{
        "model": "{model}",
        "messages": [{{"role": "user", "content": "Hello"}}],
        "provider": {{
            "only": ["{providers[0] if providers else 'provider_name'}"]
        }}
    }}
    ''')