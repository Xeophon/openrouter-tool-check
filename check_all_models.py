#!/usr/bin/env python3
"""
Check tool support for all models and providers on OpenRouter.
Generates a comprehensive report for multiple models.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx

# Load environment variables
load_dotenv()


class OpenRouterToolSupportChecker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        
    async def get_model_providers(self, model_id: str) -> List[Dict[str, str]]:
        """Fetch available providers for a specific model."""
        async with httpx.AsyncClient() as client:
            try:
                # Split model ID to get author and slug
                parts = model_id.split("/")
                if len(parts) != 2:
                    print(f"Invalid model ID format: {model_id}")
                    return []
                
                author, slug = parts
                
                # Get provider information from the endpoints API
                response = await client.get(
                    f"{self.base_url}/models/{author}/{slug}/endpoints",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code != 200:
                    print(f"Failed to fetch endpoints for {model_id}: {response.status_code}")
                    return []
                
                data = response.json()
                providers = []
                
                # Extract provider information from endpoints
                if "data" in data and "endpoints" in data["data"]:
                    for endpoint in data["data"]["endpoints"]:
                        provider_info = {
                            "provider_name": endpoint.get("provider_name", ""),
                            "display_name": endpoint.get("name", ""),
                            "context_length": endpoint.get("context_length", 0),
                            "has_pricing": "pricing" in endpoint
                        }
                        if provider_info["provider_name"]:
                            providers.append(provider_info)
                
                return providers
                
            except Exception as e:
                print(f"Error fetching providers for {model_id}: {e}")
                return []
    
    @staticmethod
    def get_weather_tool() -> Dict[str, Any]:
        """Return the weather tool definition in OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
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
                            "description": "The unit for temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    
    async def test_provider_tool_support(self, model_id: str, provider_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test if a specific provider supports tool calls for a model."""
        provider_name = provider_info["provider_name"]
        display_name = provider_info.get("display_name", provider_name)
        
        result = {
            "model_id": model_id,
            "provider_name": provider_name,
            "display_name": display_name,
            "supports_tools": False,
            "tool_call_made": False,
            "status": "unknown",  # "success", "no_tool_call", "error", "unclear"
            "error": None,
            "response_content": None,
            "tool_calls": None,
            "finish_reason": None,
            "model_used": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Create the completion with provider routing
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in San Francisco? Please use the get_weather function."
                    }
                ],
                tools=[self.get_weather_tool()],
                max_tokens=1000,
                # Specify the provider using extra_body
                extra_body={
                    "provider": {
                        "only": [provider_name]
                    }
                }
            )
            
            # Extract debugging information
            message = response.choices[0].message
            result["finish_reason"] = response.choices[0].finish_reason
            result["model_used"] = response.model if hasattr(response, 'model') else None
            
            # Check if the model made tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                result["supports_tools"] = True
                result["tool_call_made"] = True
                result["status"] = "success"
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            else:
                # No tool calls made - need to analyze why
                result["response_content"] = message.content
                
                if not message.content or message.content.strip() == "":
                    # Empty response - unclear if tools are supported
                    result["status"] = "unclear"
                    result["supports_tools"] = None  # Unknown
                elif any(phrase in str(message.content).lower() for phrase in [
                    "i can't", "i cannot", "don't have access", "unable to", "no access to tools",
                    "function calling", "tool use", "weather function"
                ]):
                    # Model explicitly says it can't use tools
                    result["status"] = "no_tool_call"
                    result["supports_tools"] = False
                else:
                    # Model responded but didn't use tools when asked
                    result["status"] = "no_tool_call"
                    result["supports_tools"] = False
                
        except Exception as e:
            error_str = str(e)
            result["error"] = error_str
            result["status"] = "error"
            
            # Analyze error type
            if any(keyword in error_str.lower() for keyword in ["tool", "function", "not supported", "invalid"]):
                result["supports_tools"] = False
            else:
                # Other errors - unclear if tools are supported
                result["supports_tools"] = None
            
        return result
    
    async def check_model(self, model_id: str) -> Dict[str, Any]:
        """Check all providers for a specific model."""
        print(f"\n{'='*60}")
        print(f"Checking model: {model_id}")
        print(f"{'='*60}")
        
        # Get providers for this model
        providers = await self.get_model_providers(model_id)
        
        if not providers:
            print(f"No providers found for {model_id}")
            return {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "providers_tested": 0,
                "providers": []
            }
        
        print(f"Found {len(providers)} providers")
        
        # Test each provider multiple times
        results = []
        for i, provider in enumerate(providers, 1):
            display_name = provider.get("display_name", provider["provider_name"])
            provider_name = provider["provider_name"]
            
            print(f"\n[{i}/{len(providers)}] Testing: {display_name}")
            
            # Run 3 tests for this provider
            test_runs = []
            for run in range(3):
                print(f"  Run {run + 1}/3...", end="", flush=True)
                
                test_result = await self.test_provider_tool_support(model_id, provider)
                test_runs.append(test_result)
                
                # Quick status indicator
                if test_result["status"] == "success":
                    print(" ✓", end="", flush=True)
                elif test_result["status"] == "unclear":
                    print(" ?", end="", flush=True)
                elif test_result["status"] == "error":
                    print(" ⚠", end="", flush=True)
                else:
                    print(" ✗", end="", flush=True)
                
                # Small delay between runs
                if run < 2:
                    await asyncio.sleep(0.3)
            
            print()  # New line after run indicators
            
            # Summarize results
            success_count = sum(1 for r in test_runs if r["status"] == "success")
            error_count = sum(1 for r in test_runs if r["status"] == "error")
            unclear_count = sum(1 for r in test_runs if r["status"] == "unclear")
            
            # Create aggregated result
            aggregated_result = {
                "model_id": model_id,
                "provider_name": provider_name,
                "display_name": display_name,
                "test_runs": test_runs,
                "summary": {
                    "total_runs": 3,
                    "success_count": success_count,
                    "error_count": error_count,
                    "unclear_count": unclear_count,
                    "no_tool_call_count": 3 - success_count - error_count - unclear_count
                },
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(aggregated_result)
            
            # Print summary
            print(f"  Summary: {success_count}/3 successful")
            if error_count > 0:
                print(f"  Errors: {error_count}/3")
            if unclear_count > 0:
                print(f"  Unclear: {unclear_count}/3")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        return {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "providers_tested": len(providers),
            "providers": results
        }


async def main():
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY in your .env file")
        return
    
    # Load models from JSON file
    try:
        with open("models.json", "r") as f:
            models = json.load(f)
    except FileNotFoundError:
        print("models.json not found!")
        return
    except json.JSONDecodeError:
        print("Invalid JSON in models.json!")
        return
    
    print("OpenRouter Tool Support Checker")
    print(f"Testing {len(models)} models")
    print("=" * 60)
    
    checker = OpenRouterToolSupportChecker(api_key)
    
    # Check all models
    all_results = {
        "generated_at": datetime.now().isoformat(),
        "total_models": len(models),
        "models": []
    }
    
    for model_id in models:
        model_result = await checker.check_model(model_id)
        all_results["models"].append(model_result)
        
        # Save intermediate results
        output_file = f"data/tool_support_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Save final results
    final_output = "data/tool_support_results_latest.json"
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {final_output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_providers = 0
    total_fully_supporting = 0  # 3/3 success
    total_partially_supporting = 0  # 1-2/3 success
    total_not_supporting = 0  # 0/3 success
    
    for model in all_results["models"]:
        providers = model["providers"]
        model_fully_supporting = sum(1 for p in providers if p["summary"]["success_count"] == 3)
        model_partially_supporting = sum(1 for p in providers if 0 < p["summary"]["success_count"] < 3)
        model_not_supporting = sum(1 for p in providers if p["summary"]["success_count"] == 0)
        
        total_providers += len(providers)
        total_fully_supporting += model_fully_supporting
        total_partially_supporting += model_partially_supporting
        total_not_supporting += model_not_supporting
        
        print(f"\n{model['model_id']}:")
        print(f"  Providers tested: {len(providers)}")
        print(f"  Full support (3/3): {model_fully_supporting}")
        if model_partially_supporting > 0:
            print(f"  Partial support (1-2/3): {model_partially_supporting}")
        if model_not_supporting > 0:
            print(f"  No support (0/3): {model_not_supporting}")
    
    print(f"\n\nTotal providers tested: {total_providers}")
    print(f"Full support (3/3): {total_fully_supporting}")
    print(f"Partial support (1-2/3): {total_partially_supporting}")
    print(f"No support (0/3): {total_not_supporting}")
    
    if total_providers > 0:
        print(f"\nPercentages:")
        print(f"  Full support: {(total_fully_supporting/total_providers*100):.1f}%")
        print(f"  Partial support: {(total_partially_supporting/total_providers*100):.1f}%")
        print(f"  No support: {(total_not_supporting/total_providers*100):.1f}%")


if __name__ == "__main__":
    asyncio.run(main())