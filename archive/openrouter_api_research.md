# OpenRouter API Documentation Research

## 1. Provider Routing Specification

### Basic Provider Routing
OpenRouter allows you to specify which providers to use through the `provider` object in your API request:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-api-key"
)

response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "provider": {
            "order": ["Azure", "OpenAI"],     # Prioritize Azure, then OpenAI
            "allow_fallbacks": True,          # Allow fallback to other providers
            "require_parameters": True,       # Only use providers supporting all parameters
            "sort": "price"                   # Sort by: price, throughput, or latency
        }
    }
)
```

### Provider Object Fields
- **`order`**: List of provider slugs to try in order
- **`allow_fallbacks`**: Whether to allow backup providers (default: true)
- **`require_parameters`**: Only use providers supporting all request parameters
- **`data_collection`**: Control providers that may store data ("allow" or "deny")
- **`only`**: List of allowed providers (exclusive)
- **`ignore`**: List of providers to skip
- **`quantizations`**: Filter providers by quantization levels
- **`sort`**: Sort providers by "price", "throughput", or "latency"
- **`max_price`**: Maximum pricing limits for the request

### Provider Routing Shortcuts
You can append suffixes to model names for quick routing preferences:
- `:nitro` - Prioritizes throughput
- `:floor` - Prioritizes lowest price

Example:
```python
response = client.chat.completions.create(
    model="openai/gpt-4:nitro",  # Prioritize throughput
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Direct HTTP Request Example
```python
import requests

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer your-openrouter-api-key",
    "HTTP-Referer": "https://your-site.com",  # Optional
    "X-Title": "Your App Name"                # Optional
}

data = {
    "model": "meta-llama/llama-3-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "provider": {
        "order": ["DeepInfra/turbo"],  # Specific provider endpoint
        "only": ["DeepInfra"],         # Only allow this provider
        "ignore": ["Together"]         # Ignore this provider
    }
}

response = requests.post(url, headers=headers, json=data)
```

## 2. Tool Calling (Function Calling) Format

### Overview
OpenRouter standardizes tool calling across all providers, following OpenAI's format. The LLM suggests tool calls, which you execute and return results for final response generation.

### Tool Definition Format
Tools are defined using OpenAI's function calling parameter format:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
}]
```

### Making a Tool Call Request
```python
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto",  # Can be: "none", "auto", "required", or specific tool
    extra_body={
        "provider": {
            "order": ["OpenAI", "Anthropic"]
        }
    }
)
```

### Tool Choice Options
- **`"none"`**: Model will not call any tool
- **`"auto"`**: Model can choose to call tools or generate a message
- **`"required"`**: Model must call one or more tools
- **Specific tool**: `{"type": "function", "function": {"name": "get_weather"}}`

### Handling Tool Call Responses
When the model wants to use a tool, it returns:
- `finish_reason`: "tool_calls"
- `tool_calls`: Array of requested tool calls

```python
if response.choices[0].finish_reason == "tool_calls":
    tool_calls = response.choices[0].message.tool_calls
    
    # Execute each tool call
    tool_results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Execute your function here
        result = execute_function(function_name, function_args)
        
        tool_results.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": str(result)
        })
    
    # Send results back to the model
    messages.append(response.choices[0].message)
    messages.extend(tool_results)
    
    final_response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=messages
    )
```

## 3. Provider-Specific Requirements

### Tool Calling Support
- OpenRouter automatically routes requests with tools only to providers that support tool use
- Different providers may have varying levels of tool calling support
- The API normalizes the interface across all providers

### Parameter Support
- Use `require_parameters: true` to ensure providers support all your parameters
- Some providers may ignore unknown parameters by default
- Check individual model documentation for specific parameter support

### Finish Reasons
OpenRouter normalizes finish reasons across providers:
- `tool_calls`: Model wants to use tools
- `stop`: Normal completion
- `length`: Hit token limit
- `content_filter`: Content was filtered
- `error`: An error occurred

## 4. Complete Example: Tool Calling with Provider Routing

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-api-key"
)

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "search_books",
        "description": "Search for books by title or author",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}]

# Initial request with tool availability and provider routing
messages = [{"role": "user", "content": "Find me books about Python programming"}]

response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    extra_body={
        "provider": {
            "order": ["OpenAI", "Anthropic", "Google"],
            "require_parameters": True,
            "allow_fallbacks": True,
            "data_collection": "deny"  # Don't use providers that train on data
        }
    }
)

# Handle tool calls if requested
if response.choices[0].finish_reason == "tool_calls":
    tool_calls = response.choices[0].message.tool_calls
    messages.append(response.choices[0].message)
    
    for tool_call in tool_calls:
        if tool_call.function.name == "search_books":
            args = json.loads(tool_call.function.arguments)
            # Simulate book search
            results = f"Found 5 books about '{args['query']}': ..."
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "search_books",
                "content": results
            })
    
    # Get final response
    final_response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=messages,
        extra_body={
            "provider": {
                "order": ["OpenAI", "Anthropic", "Google"]
            }
        }
    )
    
    print(final_response.choices[0].message.content)
```

## 5. Key Limitations and Notes

1. **Provider Availability**: Not all providers support all features. Tool calling will only route to compatible providers.

2. **Parameter Compatibility**: Different providers may have different parameter limits (e.g., max_tokens).

3. **Pricing**: Provider routing affects pricing. Use the `max_price` field to control costs.

4. **Fallback Behavior**: OpenRouter automatically falls back to alternative providers on errors unless `allow_fallbacks` is set to false.

5. **API Compatibility**: OpenRouter maintains compatibility with OpenAI's API format, making it easy to switch between providers.

6. **Authentication**: Always protect your API keys and never commit them to public repositories.

## Additional Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Provider Routing Guide](https://openrouter.ai/docs/features/provider-routing)
- [Tool Calling Guide](https://openrouter.ai/docs/features/tool-calling)
- [GitHub Examples](https://github.com/OpenRouterTeam/openrouter-examples-python)