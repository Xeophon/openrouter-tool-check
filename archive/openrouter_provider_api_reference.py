#!/usr/bin/env python3
"""
OpenRouter Provider API Reference

This file contains the exact API endpoints, request formats, and response structures
for getting provider information for models on OpenRouter.
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================

"""
1. List All Models Endpoint
   GET https://openrouter.ai/api/v1/models
   
   Returns all available models with basic information.
   Note: This endpoint shows supported_parameters as a union of all providers.

2. List Model Endpoints (Providers) - THIS IS THE KEY ENDPOINT
   GET https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints
   
   Example: GET https://openrouter.ai/api/v1/models/qwen/qwen3-30b-a3b/endpoints
   
   Returns detailed provider information for a specific model.
"""

# ============================================================================
# RESPONSE FORMAT FOR MODEL ENDPOINTS
# ============================================================================

EXAMPLE_ENDPOINTS_RESPONSE = {
    "data": {
        "id": "qwen/qwen3-30b-a3b",
        "name": "Qwen3 30B A3B",
        "created": 1741818122,  # Unix timestamp
        "description": "Model description here",
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "GPT"
        },
        "endpoints": [
            {
                "name": "DeepInfra (FP8)",  # Display name
                "provider_name": "deepinfra",  # Provider identifier for API use
                "context_length": 131072,
                "pricing": {
                    "request": "0",
                    "prompt": "0.00000008",  # Per token
                    "completion": "0.00000029"  # Per token
                },
                "supported_parameters": [
                    "temperature",
                    "top_p",
                    "top_k",
                    "frequency_penalty",
                    "presence_penalty",
                    "repetition_penalty",
                    "min_p",
                    "top_a",
                    "max_tokens",
                    "response_format",
                    "stop",
                    "seed",
                    "tools",
                    "tool_choice"
                ]
            },
            {
                "name": "Chutes",
                "provider_name": "chutes",  # Free provider
                "context_length": 32768,
                "pricing": {
                    "request": "0",
                    "prompt": "0",  # Free
                    "completion": "0"  # Free
                },
                "supported_parameters": [
                    "temperature",
                    "max_tokens",
                    "stop"
                ]
            }
        ]
    }
}

# ============================================================================
# PROVIDER ROUTING IN API REQUESTS
# ============================================================================

"""
When making requests to OpenRouter, you can control provider selection using
the 'provider' object in your request body.
"""

# Example 1: Force a specific provider
FORCE_SPECIFIC_PROVIDER_REQUEST = {
    "model": "qwen/qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "provider": {
        "only": ["deepinfra"]  # Use ONLY this provider
    }
}

# Example 2: Set provider priority order
PROVIDER_PRIORITY_REQUEST = {
    "model": "qwen/qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "provider": {
        "order": ["deepinfra", "chutes"]  # Try deepinfra first, then chutes
    }
}

# Example 3: Exclude specific providers
EXCLUDE_PROVIDERS_REQUEST = {
    "model": "qwen/qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "provider": {
        "exclude": ["chutes"]  # Don't use this provider
    }
}

# Example 4: Require parameter support
REQUIRE_PARAMETERS_REQUEST = {
    "model": "qwen/qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [...],  # Some tool definitions
    "provider": {
        "require_parameters": True  # Only use providers that support 'tools'
    }
}

# ============================================================================
# PROVIDER NAME FORMATS
# ============================================================================

"""
Provider names in OpenRouter follow these patterns:

1. Basic provider name (most common):
   - "deepinfra"
   - "chutes"
   - "together"
   - "groq"

2. Provider with variant (for specific endpoints):
   - "deepinfra/fp8"  # Specific floating-point precision
   - "together/turbo"  # Performance variant
   
3. In the endpoints response:
   - 'provider_name' field contains the base provider name
   - 'name' field contains the display name with details

4. When using in API requests:
   - Use the lowercase provider name from 'provider_name'
   - Don't include variant suffixes unless specifically needed
"""

# ============================================================================
# COMPLETE WORKING EXAMPLE
# ============================================================================

def get_and_test_providers_example():
    """Complete example of getting providers and testing each one."""
    
    import requests
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Step 1: Get providers for the model
    model_id = "qwen/qwen3-30b-a3b"
    author, slug = model_id.split("/")
    
    endpoints_url = f"https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(endpoints_url, headers=headers)
    data = response.json()
    
    # Step 2: Extract provider names
    providers = []
    for endpoint in data["data"]["endpoints"]:
        providers.append({
            "display_name": endpoint["name"],
            "api_name": endpoint["provider_name"],
            "pricing": endpoint["pricing"],
            "context_length": endpoint["context_length"]
        })
    
    # Step 3: Test each provider
    chat_url = "https://openrouter.ai/api/v1/chat/completions"
    
    for provider in providers:
        request_body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say hello"}],
            "provider": {
                "only": [provider["api_name"]]  # Force this provider
            },
            "max_tokens": 50
        }
        
        response = requests.post(chat_url, headers=headers, json=request_body)
        print(f"Provider {provider['display_name']}: {response.status_code}")

# ============================================================================
# KEY POINTS SUMMARY
# ============================================================================

"""
SUMMARY OF KEY INFORMATION:

1. Endpoint for getting providers:
   GET /api/v1/models/{author}/{slug}/endpoints
   
2. Provider information is in the 'endpoints' array:
   - 'provider_name': The identifier to use in API requests
   - 'name': Display name with additional info
   - 'pricing': Cost per token for prompt/completion
   - 'supported_parameters': What features this provider supports
   
3. To force a specific provider in requests:
   Add "provider": {"only": ["provider_name"]} to your request body
   
4. Provider names are typically lowercase and simple:
   Examples: "deepinfra", "chutes", "together", "groq"
   
5. Free providers typically have:
   - Zero pricing ("0" for prompt and completion)
   - May have limited features or rate limits
   - Often have "free" in the model variant (e.g., "qwen/qwen3-30b-a3b:free")
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nSee the code above for exact API formats and examples.")