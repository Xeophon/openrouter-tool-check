# OpenRouter API Documentation Summary

## Overview
OpenRouter provides a unified API to access 300+ AI models through a single endpoint with automatic provider routing and fallback handling.

## Key API Endpoints

### 1. List Available Models
- **Endpoint**: `GET https://openrouter.ai/api/v1/models`
- **Purpose**: Returns all available models
- **Authentication**: Bearer token required

### 2. Get Model Provider/Endpoints
- **Endpoint**: `GET https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints`
- **Purpose**: Returns available providers and endpoints for a specific model
- **Example**: `GET https://openrouter.ai/api/v1/models/qwen/qwen3-30b-a3b/endpoints`
- **Authentication**: Bearer token required

### 3. Chat Completions
- **Endpoint**: `POST https://openrouter.ai/api/v1/chat/completions`
- **Purpose**: Make chat completion requests
- **Authentication**: Bearer token required

## Authentication

### API Key
- Obtain from OpenRouter account settings
- Use in Authorization header: `Authorization: Bearer YOUR_API_KEY`

### Optional Headers
- `HTTP-Referer`: Your application URL
- `X-Title`: Your application name

## Model Information for qwen/qwen3-30b-a3b

### Model Variants
- **Standard**: `qwen/qwen3-30b-a3b`
- **Free tier**: `qwen/qwen3-30b-a3b:free`

### Model Specifications
- 30.5 billion parameters (3.3 billion activated)
- 48 layers, 128 experts (8 activated per task)
- Supports up to 131K token contexts

## Example: Getting Provider Information

```python
import requests

# Set your API key
api_key = "YOUR_OPENROUTER_API_KEY"
headers = {"Authorization": f"Bearer {api_key}"}

# Get providers for qwen/qwen3-30b-a3b
url = "https://openrouter.ai/api/v1/models/qwen/qwen3-30b-a3b/endpoints"
response = requests.get(url, headers=headers)
providers = response.json()

# The response will include:
# - Available providers (e.g., different hosting services)
# - Endpoint details for each provider
# - Context length limits
# - Pricing information
# - Supported parameters
```

## Response Format

The endpoints API returns:
```json
{
  "id": "qwen/qwen3-30b-a3b",
  "name": "Qwen3 30B A3B",
  "description": "...",
  "endpoints": [
    {
      "provider": "provider_name",
      "name": "endpoint_name",
      "context_length": 131072,
      "pricing": {
        "prompt": 0.001,
        "completion": 0.002
      },
      "supported_parameters": ["temperature", "max_tokens", ...]
    }
  ]
}
```

## Important Notes

1. **Provider Routing**: OpenRouter automatically routes to the best available provider
2. **Free Tier**: Models with `:free` suffix have usage limitations
3. **Compatibility**: API is compatible with OpenAI SDK
4. **Rate Limits**: Check `/api/v1/auth/key` for rate limit information
5. **Parameter Support**: The `supported_parameters` field shows a union of all parameters across providers