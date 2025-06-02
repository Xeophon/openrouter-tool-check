#!/usr/bin/env python3
"""
Example demonstrating OpenRouter API usage with provider routing and tool calling.

This script shows:
1. How to specify providers when making API calls
2. How to use tool calling (function calling) with OpenRouter
3. Provider-specific routing options and limitations
"""

import os
import json
from openai import OpenAI
from typing import Dict, Any, List

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
)


def get_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
    """Mock weather function for demonstration."""
    # In a real application, this would call a weather API
    return {
        "location": location,
        "temperature": 72 if unit == "fahrenheit" else 22,
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": "65%"
    }


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Mock web search function for demonstration."""
    # In a real application, this would call a search API
    results = []
    for i in range(num_results):
        results.append({
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a sample search result for the query '{query}'..."
        })
    return results


# Define available tools in OpenAI format
TOOLS = [
    {
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
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                        "default": "fahrenheit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def execute_function(function_name: str, function_args: Dict[str, Any]) -> Any:
    """Execute the requested function with given arguments."""
    if function_name == "get_weather":
        return get_weather(**function_args)
    elif function_name == "search_web":
        return search_web(**function_args)
    else:
        raise ValueError(f"Unknown function: {function_name}")


def demo_basic_provider_routing():
    """Demonstrate basic provider routing."""
    print("\n=== Basic Provider Routing Demo ===\n")
    
    # Example 1: Prioritize specific providers
    response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Hello! Tell me a joke."}],
        extra_body={
            "provider": {
                "order": ["Azure", "OpenAI"],  # Try Azure first, then OpenAI
                "allow_fallbacks": True        # Allow other providers if these fail
            }
        }
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Provider used: {response.provider if hasattr(response, 'provider') else 'Unknown'}")


def demo_provider_with_constraints():
    """Demonstrate provider routing with constraints."""
    print("\n=== Provider Routing with Constraints Demo ===\n")
    
    # Example 2: Route based on specific requirements
    response = client.chat.completions.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[{"role": "user", "content": "Explain quantum computing briefly."}],
        max_tokens=150,
        extra_body={
            "provider": {
                "order": ["DeepInfra", "Together", "Fireworks"],
                "require_parameters": True,    # Only use providers supporting max_tokens
                "data_collection": "deny",     # Don't use providers that train on data
                "sort": "price"                # Sort remaining providers by price
            }
        }
    )
    
    print(f"Response: {response.choices[0].message.content}")


def demo_model_shortcuts():
    """Demonstrate model routing shortcuts."""
    print("\n=== Model Shortcuts Demo ===\n")
    
    # Use :nitro suffix for throughput optimization
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo:nitro",  # Prioritize throughput
        messages=[{"role": "user", "content": "List 3 benefits of exercise."}]
    )
    
    print("Throughput-optimized response:")
    print(response.choices[0].message.content)
    
    # Use :floor suffix for cost optimization
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo:floor",  # Prioritize lowest cost
        messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    
    print("\nCost-optimized response:")
    print(response.choices[0].message.content)


def demo_tool_calling():
    """Demonstrate tool calling with provider routing."""
    print("\n=== Tool Calling Demo ===\n")
    
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco and New York? Also search for the best restaurants in San Francisco."}
    ]
    
    # Initial request with tools
    response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",  # Let model decide when to use tools
        extra_body={
            "provider": {
                "order": ["OpenAI", "Anthropic"],  # Only these providers support tools well
                "require_parameters": True
            }
        }
    )
    
    # Check if model wants to use tools
    if response.choices[0].finish_reason == "tool_calls":
        print("Model requested tool calls:")
        
        # Add assistant's message with tool calls to conversation
        messages.append(response.choices[0].message)
        
        # Process each tool call
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\nCalling function: {function_name}")
            print(f"Arguments: {function_args}")
            
            # Execute the function
            try:
                result = execute_function(function_name, function_args)
                print(f"Result: {result}")
                
                # Add function result to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result)
                })
            except Exception as e:
                print(f"Error executing function: {e}")
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: {str(e)}"
                })
        
        # Get final response with tool results
        print("\nGetting final response with tool results...")
        final_response = client.chat.completions.create(
            model="openai/gpt-4",
            messages=messages,
            extra_body={
                "provider": {
                    "order": ["OpenAI", "Anthropic"]
                }
            }
        )
        
        print(f"\nFinal response:\n{final_response.choices[0].message.content}")
    else:
        print(f"Response (no tools used): {response.choices[0].message.content}")


def demo_streaming_with_provider():
    """Demonstrate streaming with provider routing."""
    print("\n=== Streaming Demo ===\n")
    
    stream = client.chat.completions.create(
        model="anthropic/claude-3-sonnet",
        messages=[{"role": "user", "content": "Write a haiku about programming."}],
        stream=True,
        extra_body={
            "provider": {
                "order": ["Anthropic", "AWS"],
                "allow_fallbacks": False  # Only use specified providers
            }
        }
    )
    
    print("Streaming response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def demo_provider_specific_features():
    """Demonstrate provider-specific features and limitations."""
    print("\n=== Provider-Specific Features Demo ===\n")
    
    # Example with specific provider endpoint
    response = client.chat.completions.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[{"role": "user", "content": "What is machine learning?"}],
        extra_body={
            "provider": {
                "only": ["DeepInfra"],  # Only use this specific provider
                "max_price": {
                    "prompt": 0.001,     # Max $0.001 per 1K prompt tokens
                    "completion": 0.002  # Max $0.002 per 1K completion tokens
                }
            }
        }
    )
    
    print(f"Response: {response.choices[0].message.content[:200]}...")


if __name__ == "__main__":
    print("OpenRouter API Demo - Provider Routing and Tool Calling")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\nWarning: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it to run the demos:")
        print("export OPENROUTER_API_KEY='your-api-key-here'")
        print("\nUsing placeholder key for demonstration purposes only.\n")
    
    try:
        # Run all demos
        demo_basic_provider_routing()
        demo_provider_with_constraints()
        demo_model_shortcuts()
        demo_tool_calling()
        demo_streaming_with_provider()
        demo_provider_specific_features()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("\nMake sure you have a valid OPENROUTER_API_KEY set.")