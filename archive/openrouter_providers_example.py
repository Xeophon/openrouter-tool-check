#!/usr/bin/env python3
"""
OpenRouter API Example: Getting Provider Information for Models

This script demonstrates how to:
1. List all available models
2. Get specific provider/endpoint information for a model (e.g., qwen/qwen3-30b-a3b)
3. Make authenticated requests to OpenRouter API
"""

import requests
import json
from typing import Dict, List, Optional

class OpenRouterAPI:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: str):
        """
        Initialize the OpenRouter API client
        
        Args:
            api_key: Your OpenRouter API key
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def list_models(self) -> Optional[Dict]:
        """
        List all available models
        
        Returns:
            Dictionary containing model information or None if error
        """
        url = f"{self.base_url}/models"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return None
    
    def get_model_endpoints(self, author: str, slug: str) -> Optional[Dict]:
        """
        Get provider/endpoint information for a specific model
        
        Args:
            author: Model author (e.g., "qwen")
            slug: Model slug (e.g., "qwen3-30b-a3b")
            
        Returns:
            Dictionary containing endpoint information or None if error
        """
        url = f"{self.base_url}/models/{author}/{slug}/endpoints"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting model endpoints: {e}")
            return None
    
    def find_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Find information about a specific model from the models list
        
        Args:
            model_id: Model ID to search for (e.g., "qwen/qwen3-30b-a3b")
            
        Returns:
            Model information dictionary or None if not found
        """
        models_data = self.list_models()
        if not models_data or 'data' not in models_data:
            return None
        
        for model in models_data['data']:
            if model.get('id') == model_id:
                return model
        
        return None
    
    def test_chat_completion(self, model: str, message: str) -> Optional[Dict]:
        """
        Test a chat completion request with a specific model
        
        Args:
            model: Model ID (e.g., "qwen/qwen3-30b-a3b:free")
            message: Message to send to the model
            
        Returns:
            Response dictionary or None if error
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error with chat completion: {e}")
            return None


def main():
    """Main function demonstrating OpenRouter API usage"""
    
    # Replace with your actual API key
    API_KEY = "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"
    
    # Initialize client
    client = OpenRouterAPI(API_KEY)
    
    print("=== OpenRouter API Demo ===\n")
    
    # Example 1: List all available models
    print("1. Listing all available models...")
    models = client.list_models()
    if models and 'data' in models:
        print(f"Found {len(models['data'])} models")
        # Show first few models
        for i, model in enumerate(models['data'][:5]):
            print(f"  - {model.get('id', 'Unknown ID')}: {model.get('name', 'Unknown Name')}")
        print("  ...\n")
    
    # Example 2: Find specific model information
    model_id = "qwen/qwen3-30b-a3b"
    print(f"2. Finding information for model: {model_id}")
    model_info = client.find_model_info(model_id)
    if model_info:
        print(f"  Name: {model_info.get('name', 'Unknown')}")
        print(f"  Description: {model_info.get('description', 'No description')[:100]}...")
        print(f"  Context Length: {model_info.get('context_length', 'Unknown')}")
        print(f"  Pricing: ${model_info.get('pricing', {}).get('prompt', 'Unknown')}/1K prompt tokens")
        print()
    
    # Example 3: Get provider endpoints for the model
    print(f"3. Getting provider/endpoint information for: {model_id}")
    # Split the model ID to get author and slug
    if '/' in model_id:
        author, slug = model_id.split('/', 1)
        endpoints = client.get_model_endpoints(author, slug)
        
        if endpoints:
            print("  Available endpoints:")
            if 'endpoints' in endpoints:
                for endpoint in endpoints['endpoints']:
                    print(f"    - Provider: {endpoint.get('provider', 'Unknown')}")
                    print(f"      Name: {endpoint.get('name', 'Unknown')}")
                    print(f"      Context: {endpoint.get('context_length', 'Unknown')} tokens")
                    print(f"      Pricing: ${endpoint.get('pricing', {}).get('prompt', 'Unknown')}/1K prompt")
                    print()
            else:
                print(json.dumps(endpoints, indent=2))
    
    # Example 4: Test chat completion (optional)
    print("4. Testing chat completion (optional)")
    print("   To test, uncomment the code below and ensure you have a valid API key\n")
    
    # Uncomment to test:
    # response = client.test_chat_completion(
    #     model="qwen/qwen3-30b-a3b:free",
    #     message="Hello, what's the weather like today?"
    # )
    # if response:
    #     print(f"Response: {response.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")


if __name__ == "__main__":
    main()