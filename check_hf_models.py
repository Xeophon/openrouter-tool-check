#!/usr/bin/env python3
"""
Check tool support for all models and providers on Hugging Face Inference API.
Generates a comprehensive report for multiple models.
Optimized version with concurrent requests.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from huggingface_hub import model_info, InferenceClient

# Load environment variables
load_dotenv()


class HuggingFaceToolSupportChecker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(5)

    async def get_model_providers(self, model_id: str) -> List[Dict[str, str]]:
        """Fetch available providers for a specific model."""
        try:
            # Get model info with inference provider mapping
            info = await asyncio.to_thread(
                model_info,
                model_id,
                expand="inferenceProviderMapping",
                token=self.api_key,
            )

            providers = []

            # Extract provider information from inference provider mapping
            if (
                hasattr(info, "inference_provider_mapping")
                and info.inference_provider_mapping
            ):
                for (
                    provider_name,
                    provider_info,
                ) in info.inference_provider_mapping.items():
                    if isinstance(provider_info, dict):
                        provider_data = {
                            "provider_name": provider_name,
                            "display_name": provider_info.get(
                                "display_name", provider_name
                            ),
                            "vendor": provider_info.get("vendor", provider_name),
                            "region": provider_info.get("region", ""),
                            "base_url": provider_info.get("base_url", ""),
                            "endpoint_url": provider_info.get("endpoint_url", ""),
                        }
                        providers.append(provider_data)
                    else:
                        # Simple provider format
                        providers.append(
                            {
                                "provider_name": provider_name,
                                "display_name": provider_name,
                                "vendor": provider_name,
                                "region": "",
                                "base_url": "",
                                "endpoint_url": "",
                            }
                        )

            return providers
        except Exception as e:
            print(f"Error getting providers for {model_id}: {e}")
            return []

    async def test_provider(
        self, model_id: str, provider: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test if a specific provider supports tool calling."""
        provider_name = provider["provider_name"]
        display_name = provider.get("display_name", provider_name)

        result = {
            "model_id": model_id,
            "provider_name": provider_name,
            "display_name": display_name,
            "vendor": provider.get("vendor", ""),
            "region": provider.get("region", ""),
            "supports_tools": False,
            "tool_call_made": False,
            "status": "unknown",  # "success", "error", "unclear", "no_tool_call"
            "error": None,
            "response_content": None,
            "tool_calls": None,
            "finish_reason": None,
            "model_used": None,
            "timestamp": datetime.now().isoformat(),
        }

        async with self.semaphore:  # Limit concurrent requests
            try:
                # Create InferenceClient with provider parameter if not serverless
                if provider_name == "serverless":
                    # Use default HuggingFace serverless API
                    client = InferenceClient(
                        token=self.api_key,
                    )
                else:
                    # Use specific provider
                    vendor = provider.get("vendor", provider_name).lower()
                    # Map common vendor names to InferenceClient provider names
                    provider_mapping = {
                        "aws": "aws",
                        "amazon": "aws",
                        "sagemaker": "aws",
                        "azure": "azure",
                        "microsoft": "azure",
                        "together": "together",
                        "replicate": "replicate",
                        "cohere": "cohere",
                        "nebius": "nebius",
                    }
                    mapped_provider = provider_mapping.get(vendor, vendor)
                    
                    try:
                        client = InferenceClient(
                            provider=mapped_provider,
                            token=self.api_key,
                        )
                    except Exception:
                        # Fallback to serverless if provider is not supported
                        client = InferenceClient(
                            token=self.api_key,
                        )

                # Create the completion with tools using synchronous call in async context
                response = await asyncio.to_thread(
                    client.chat_completion,
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": "What's the weather like in San Francisco?",
                        }
                    ],
                    tools=[
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
                                            "description": "The city and state, e.g. San Francisco, CA",
                                        },
                                        "unit": {
                                            "type": "string",
                                            "enum": ["celsius", "fahrenheit"],
                                            "description": "The unit of temperature",
                                        },
                                    },
                                    "required": ["location"],
                                },
                            },
                        }
                    ],
                    max_tokens=1000,
                    )

                # Extract debugging information
                message = response.choices[0].message
                result["finish_reason"] = response.choices[0].finish_reason
                result["model_used"] = (
                    response.model if hasattr(response, "model") else None
                )

                # Check if the model made a tool call
                if hasattr(message, "tool_calls") and message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                    result["tool_call_made"] = True
                    result["status"] = "success"
                    result["supports_tools"] = True
                elif message.content:
                    result["response_content"] = message.content
                    result["tool_call_made"] = False
                    result["status"] = "no_tool_call"
                    result["supports_tools"] = False
                else:
                    result["status"] = "unclear"
                    result["supports_tools"] = None

            except Exception as e:
                error_str = str(e)
                result["error"] = error_str
                result["status"] = "error"

                # Analyze error type
                if "tool" in error_str.lower() or "function" in error_str.lower():
                    result["supports_tools"] = False
                elif "404" in error_str:
                    result["supports_tools"] = False
                else:
                    # Other errors - unclear if tools are supported
                    result["supports_tools"] = None

            return result

    async def test_provider_structured_output(
        self, model_id: str, provider: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test if a specific provider supports structured output."""
        provider_name = provider["provider_name"]
        display_name = provider.get("display_name", provider_name)

        result = {
            "model_id": model_id,
            "provider_name": provider_name,
            "display_name": display_name,
            "vendor": provider.get("vendor", ""),
            "region": provider.get("region", ""),
            "supports_structured_output": False,
            "status": "unknown",  # "success", "error", "unclear"
            "error": None,
            "response_content": None,
            "finish_reason": None,
            "model_used": None,
            "timestamp": datetime.now().isoformat(),
        }

        async with self.semaphore:  # Limit concurrent requests
            try:
                # Create InferenceClient with provider parameter if not serverless
                if provider_name == "serverless":
                    # Use default HuggingFace serverless API
                    client = InferenceClient(
                        token=self.api_key,
                    )
                else:
                    # Use specific provider
                    vendor = provider.get("vendor", provider_name).lower()
                    # Map common vendor names to InferenceClient provider names
                    provider_mapping = {
                        "aws": "aws",
                        "amazon": "aws",
                        "sagemaker": "aws",
                        "azure": "azure",
                        "microsoft": "azure",
                        "together": "together",
                        "replicate": "replicate",
                        "cohere": "cohere",
                        "nebius": "nebius",
                    }
                    mapped_provider = provider_mapping.get(vendor, vendor)
                    
                    try:
                        client = InferenceClient(
                            provider=mapped_provider,
                            token=self.api_key,
                        )
                    except Exception:
                        # Fallback to serverless if provider is not supported
                        client = InferenceClient(
                            token=self.api_key,
                        )

                # Create the completion with structured output format using synchronous call in async context
                response = await asyncio.to_thread(
                    client.chat_completion,
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": "What's the weather like in London?",
                        }
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "weather",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City or location name",
                                    },
                                    "temperature": {
                                        "type": "number",
                                        "description": "Temperature in Celsius",
                                    },
                                    "conditions": {
                                        "type": "string",
                                        "description": "Weather conditions description",
                                    },
                                },
                                "required": ["location", "temperature", "conditions"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    max_tokens=1000,
                    )

                # Extract debugging information
                message = response.choices[0].message
                result["finish_reason"] = response.choices[0].finish_reason
                result["model_used"] = (
                    response.model if hasattr(response, "model") else None
                )

                # Check if the model returned valid JSON according to our schema
                if message.content:
                    result["response_content"] = message.content
                    try:
                        json_response = json.loads(message.content)
                        if all(
                            key in json_response
                            for key in ["location", "temperature", "conditions"]
                        ):
                            result["supports_structured_output"] = True
                            result["status"] = "success"
                        else:
                            result["supports_structured_output"] = False
                            result["status"] = "invalid_schema"
                    except json.JSONDecodeError:
                        result["supports_structured_output"] = False
                        result["status"] = "invalid_json"
                else:
                    result["status"] = "unclear"
                    result["supports_structured_output"] = None

            except Exception as e:
                error_str = str(e)
                result["error"] = error_str
                result["status"] = "error"

                # Analyze error type
                if any(
                    keyword in error_str.lower()
                    for keyword in [
                        "response_format",
                        "json_schema",
                        "not supported",
                        "invalid",
                    ]
                ):
                    result["supports_structured_output"] = False
                else:
                    # Other errors - unclear if structured output is supported
                    result["supports_structured_output"] = None

            return result

    async def check_model_structured_output(self, model_id: str) -> Dict[str, Any]:
        """Check all providers for structured output support for a specific model."""
        print(f"\n{'=' * 60}")
        print(f"Checking structured output support for model: {model_id}")
        print(f"{'=' * 60}")

        # Get providers for this model
        providers = await self.get_model_providers(model_id)

        if not providers:
            print(f"No providers found for {model_id}")
            return {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "providers_tested": 0,
                "providers": [],
            }

        print(f"Found {len(providers)} providers")

        # Create all test tasks
        tasks = []
        for provider in providers:
            # Create 3 test tasks for each provider
            for run in range(3):
                task = self.test_provider_structured_output(model_id, provider)
                tasks.append((provider, run, task))

        print(f"Running {len(tasks)} structured output tests concurrently...")

        # Execute all tests concurrently
        results = await asyncio.gather(*[task for _, _, task in tasks])

        # Group results by provider
        provider_results = {}
        for i, (provider, run, _) in enumerate(tasks):
            provider_name = provider["provider_name"]
            if provider_name not in provider_results:
                provider_results[provider_name] = {
                    "provider": provider,
                    "test_runs": [],
                }
            provider_results[provider_name]["test_runs"].append(results[i])

        # Process and format results
        final_results = []
        for i, (provider_name, data) in enumerate(provider_results.items(), 1):
            provider = data["provider"]
            test_runs = data["test_runs"]
            display_name = provider.get("display_name", provider_name)

            # Calculate summary
            success_count = sum(1 for r in test_runs if r["status"] == "success")
            error_count = sum(1 for r in test_runs if r["status"] == "error")
            unclear_count = sum(1 for r in test_runs if r["status"] == "unclear")
            fail_count = 3 - success_count - error_count - unclear_count

            # Display results
            print(f"\n[{i}/{len(providers)}] {display_name}:")
            print(f"  Summary: {success_count}/3 successful")
            if error_count > 0:
                print(f"  Errors: {error_count}/3")
            if unclear_count > 0:
                print(f"  Unclear: {unclear_count}/3")

            final_results.append(
                {
                    "model_id": model_id,
                    "provider_name": provider_name,
                    "display_name": display_name,
                    "vendor": provider.get("vendor", ""),
                    "region": provider.get("region", ""),
                    "test_runs": test_runs,
                    "summary": {
                        "total_runs": 3,
                        "success_count": success_count,
                        "error_count": error_count,
                        "unclear_count": unclear_count,
                        "fail_count": fail_count,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "providers_tested": len(providers),
            "providers": final_results,
        }

    async def check_model(self, model_id: str) -> Dict[str, Any]:
        """Check all providers for a specific model."""
        print(f"\n{'=' * 60}")
        print(f"Checking model: {model_id}")
        print(f"{'=' * 60}")

        # Get providers for this model
        providers = await self.get_model_providers(model_id)

        if not providers:
            print(f"No providers found for {model_id}")
            return {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "providers_tested": 0,
                "providers": [],
            }

        print(f"Found {len(providers)} providers")

        # Create all test tasks
        tasks = []
        for provider in providers:
            # Create 3 test tasks for each provider
            for run in range(3):
                task = self.test_provider(model_id, provider)
                tasks.append((provider, run, task))

        print(f"Running {len(tasks)} tests concurrently...")

        # Execute all tests concurrently
        results = await asyncio.gather(*[task for _, _, task in tasks])

        # Group results by provider
        provider_results = {}
        for i, (provider, run, _) in enumerate(tasks):
            provider_name = provider["provider_name"]
            if provider_name not in provider_results:
                provider_results[provider_name] = {
                    "provider": provider,
                    "test_runs": [],
                }
            provider_results[provider_name]["test_runs"].append(results[i])

        # Process and format results
        final_results = []
        for i, (provider_name, data) in enumerate(provider_results.items(), 1):
            provider = data["provider"]
            test_runs = data["test_runs"]
            display_name = provider.get("display_name", provider_name)

            # Calculate summary
            success_count = sum(1 for r in test_runs if r["status"] == "success")
            error_count = sum(1 for r in test_runs if r["status"] == "error")
            unclear_count = sum(1 for r in test_runs if r["status"] == "unclear")
            no_tool_call_count = sum(
                1 for r in test_runs if r["status"] == "no_tool_call"
            )

            # Display results
            print(f"\n[{i}/{len(providers)}] {display_name}:")
            print(f"  Summary: {success_count}/3 successful")
            if error_count > 0:
                print(f"  Errors: {error_count}/3")
            if unclear_count > 0:
                print(f"  Unclear: {unclear_count}/3")

            final_results.append(
                {
                    "model_id": model_id,
                    "provider_name": provider_name,
                    "display_name": display_name,
                    "vendor": provider.get("vendor", ""),
                    "region": provider.get("region", ""),
                    "test_runs": test_runs,
                    "summary": {
                        "total_runs": 3,
                        "success_count": success_count,
                        "error_count": error_count,
                        "unclear_count": unclear_count,
                        "no_tool_call_count": no_tool_call_count,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "providers_tested": len(providers),
            "providers": final_results,
        }


async def main():
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        # Try alternate environment variable names
        api_key = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not api_key:
            print("Error: HF_TOKEN environment variable not set")
            print(
                "Please set one of: HF_TOKEN, HUGGINGFACE_TOKEN, or HUGGING_FACE_HUB_TOKEN"
            )
            return

    # Load models from unified models file
    with open("models.json", "r") as f:
        models_data = json.load(f)
    
    # Extract HuggingFace models
    models = []
    for model_name, platforms in models_data.items():
        if "huggingface" in platforms and platforms["huggingface"]:
            models.extend(platforms["huggingface"])

    print("Hugging Face Inference API Tool Support Checker")
    print(f"Testing {len(models)} models")
    print("=" * 60)

    checker = HuggingFaceToolSupportChecker(api_key)

    # Check all models
    all_results = {
        "generated_at": datetime.now().isoformat(),
        "total_models": len(models),
        "models": [],
    }

    # Process all models concurrently in batches
    batch_size = 2  # Process 2 models at a time to avoid overwhelming the API
    for i in range(0, len(models), batch_size):
        batch = models[i : i + batch_size]
        print(
            f"\nProcessing batch {i // batch_size + 1}/{(len(models) + batch_size - 1) // batch_size}"
        )

        # Create tasks for this batch
        batch_tasks = []
        for model_id in batch:
            # Run both tool support and structured output tests concurrently
            tool_task = checker.check_model(model_id)
            structured_task = checker.check_model_structured_output(model_id)
            batch_tasks.append((model_id, tool_task, structured_task))

        # Execute batch
        for model_id, tool_task, structured_task in batch_tasks:
            tool_result = await tool_task
            structured_result = await structured_task

            # Combine results
            tool_result["structured_output"] = structured_result["providers"]
            all_results["models"].append(tool_result)

    # Save final results
    final_output = "data_hf.json"
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {final_output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_providers = 0
    total_fully_supporting_tools = 0  # 3/3 success
    total_partially_supporting_tools = 0  # 1-2/3 success
    total_not_supporting_tools = 0  # 0/3 success

    # Structured output counters
    total_fully_supporting_structured_output = 0
    total_partially_supporting_structured_output = 0
    total_not_supporting_structured_output = 0

    for model_data in all_results["models"]:
        model_id = model_data["model_id"]
        providers_data = model_data["providers"]

        for provider_data in providers_data:
            total_providers += 1
            summary = provider_data["summary"]

            # Count tool support
            if summary["success_count"] == 3:
                total_fully_supporting_tools += 1
            elif summary["success_count"] > 0:
                total_partially_supporting_tools += 1
            else:
                total_not_supporting_tools += 1

        # Count structured output support
        if "structured_output" in model_data:
            for provider_data in model_data["structured_output"]:
                summary = provider_data["summary"]
                if summary["success_count"] == 3:
                    total_fully_supporting_structured_output += 1
                elif summary["success_count"] > 0:
                    total_partially_supporting_structured_output += 1
                else:
                    total_not_supporting_structured_output += 1

    print(f"\nTotal provider endpoints tested: {total_providers}")
    print("\nTool support:")
    print(f"  Fully supporting (3/3): {total_fully_supporting_tools}")
    print(f"  Partially supporting (1-2/3): {total_partially_supporting_tools}")
    print(f"  Not supporting (0/3): {total_not_supporting_tools}")

    if (
        total_fully_supporting_structured_output > 0
        or total_partially_supporting_structured_output > 0
    ):
        print("\nStructured output support:")
        print(f"  Fully supporting (3/3): {total_fully_supporting_structured_output}")
        print(
            f"  Partially supporting (1-2/3): {total_partially_supporting_structured_output}"
        )
        print(f"  Not supporting (0/3): {total_not_supporting_structured_output}")


if __name__ == "__main__":
    asyncio.run(main())
