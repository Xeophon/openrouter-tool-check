#!/usr/bin/env python3
"""
Script to test each provider individually for a specific model on OpenRouter.

This demonstrates how to:
1. Get the list of providers for a model
2. Test each provider separately
3. Handle provider-specific routing in OpenRouter
"""

import os
import requests
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

def get_model_providers(model_id: str) -> List[Dict[str, Any]]:
    """
    Get detailed provider information for a model.
    
    Returns a list of provider dictionaries with all available information.
    """
    parts = model_id.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid model ID format: {model_id}")
    
    author, slug = parts
    url = f"{BASE_URL}/models/{author}/{slug}/endpoints"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error getting providers: HTTP {response.status_code}")
        return []
    
    data = response.json()
    providers = []
    
    if "data" in data and "endpoints" in data["data"]:
        for endpoint in data["data"]["endpoints"]:
            provider_info = {
                "name": endpoint.get("name", ""),
                "provider_name": endpoint.get("provider_name", ""),
                "context_length": endpoint.get("context_length", 0),
                "pricing": endpoint.get("pricing", {}),
                "supported_parameters": endpoint.get("supported_parameters", [])
            }
            providers.append(provider_info)
    
    return providers

def test_provider(model_id: str, provider_name: str, test_message: str = "Hello, can you hear me?") -> Dict[str, Any]:
    """
    Test a specific provider for a model.
    
    Args:
        model_id: The model ID (e.g., "qwen/qwen3-30b-a3b")
        provider_name: The provider name to test
        test_message: The test message to send
        
    Returns:
        Dict with test results including response, latency, and any errors
    """
    url = f"{BASE_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/openrouter-test",  # Optional but recommended
        "X-Title": "Provider Test Script"  # Optional but recommended
    }
    
    # Request body with provider routing
    request_body = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": test_message
            }
        ],
        "provider": {
            "only": [provider_name.lower()],  # Force this specific provider
            "require_parameters": True  # Ensure provider supports all parameters
        },
        "max_tokens": 100,  # Limit response length for testing
        "temperature": 0.7
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=request_body)
        end_time = time.time()
        latency = end_time - start_time
        
        result = {
            "provider": provider_name,
            "status_code": response.status_code,
            "latency": latency,
            "success": response.status_code == 200
        }
        
        if response.status_code == 200:
            data = response.json()
            result["response"] = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            result["model_used"] = data.get("model", "")
            result["usage"] = data.get("usage", {})
            
            # Check if a specific provider was used (from response headers if available)
            if "x-provider" in response.headers:
                result["actual_provider"] = response.headers["x-provider"]
        else:
            result["error"] = response.text
            
    except Exception as e:
        result = {
            "provider": provider_name,
            "status_code": None,
            "latency": time.time() - start_time,
            "success": False,
            "error": str(e)
        }
    
    return result

def test_all_providers(model_id: str):
    """Test all available providers for a model."""
    
    print(f"Getting providers for model: {model_id}")
    print("=" * 80)
    
    providers = get_model_providers(model_id)
    
    if not providers:
        print("No providers found!")
        return
    
    print(f"\nFound {len(providers)} provider(s):")
    for i, provider in enumerate(providers, 1):
        print(f"\n{i}. Provider: {provider['provider_name']}")
        print(f"   Name: {provider['name']}")
        print(f"   Context Length: {provider['context_length']}")
        print(f"   Pricing: {provider['pricing']}")
    
    print("\n" + "=" * 80)
    print("Testing each provider individually...")
    print("=" * 80)
    
    results = []
    
    for provider in providers:
        provider_name = provider['provider_name']
        print(f"\nTesting provider: {provider_name}")
        print("-" * 40)
        
        result = test_provider(model_id, provider_name)
        results.append(result)
        
        if result['success']:
            print(f"✓ Success! (Latency: {result['latency']:.2f}s)")
            print(f"  Response: {result['response'][:100]}...")
            if 'usage' in result:
                print(f"  Tokens used: {result['usage']}")
        else:
            print(f"✗ Failed!")
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        # Small delay between tests to avoid rate limiting
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal providers tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if successful > 0:
        avg_latency = sum(r['latency'] for r in results if r['success']) / successful
        print(f"Average latency (successful requests): {avg_latency:.2f}s")

def main():
    """Main function."""
    
    # You can test different models by changing this
    models_to_test = [
        "qwen/qwen3-30b-a3b",
        # "qwen/qwen3-30b-a3b:free",  # Free variant if available
    ]
    
    for model_id in models_to_test:
        print(f"\n{'#' * 80}")
        print(f"# Testing Model: {model_id}")
        print(f"{'#' * 80}")
        
        test_all_providers(model_id)
        
        # Delay between different models
        if len(models_to_test) > 1:
            print("\nWaiting before testing next model...")
            time.sleep(2)

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenRouter API key")
    else:
        main()