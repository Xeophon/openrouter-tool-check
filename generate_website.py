#!/usr/bin/env python3
"""
Generate a static website from the tool support test results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict


def load_latest_results():
    """Load the latest test results from JSON."""
    results_file = "data.json"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, "r") as f:
        return json.load(f)


def group_models(models):
    """Group models by base name (e.g., group 'model' and 'model:free' together)."""
    grouped = defaultdict(list)
    
    for model in models:
        model_id = model["model_id"]
        # Extract base model name (before :free or other suffixes)
        base_name = model_id.split(":")[0]
        grouped[base_name].append(model)
    
    return grouped


def get_all_providers(results):
    """Get a sorted list of all unique providers across all models."""
    providers = set()
    
    for model in results["models"]:
        for provider in model["providers"]:
            providers.add(provider["provider_name"])
    
    return sorted(list(providers))


def has_structured_output_data(results):
    """Check if the results contain structured output test data."""
    if not results or not "models" in results:
        return False
    
    # Check if any model has structured_output data
    for model in results["models"]:
        if "structured_output" in model:
            return True
    
    return False


def get_cell_status(model_data, provider_name, data_type="tool_support"):
    """Get the status for a specific model-provider combination.
    
    Args:
        model_data: The model data dictionary
        provider_name: The provider name to check
        data_type: Either "tool_support" (default) or "structured_output"
    """
    providers_list = model_data["providers"]
    
    # Use structured_output data if requested and available
    if data_type == "structured_output" and "structured_output" in model_data:
        providers_list = model_data["structured_output"]
    
    for provider in providers_list:
        if provider["provider_name"] == provider_name:
            summary = provider["summary"]
            success_count = summary["success_count"]
            
            # Determine status and details
            if success_count == 3:
                return "success", f"{success_count}/3", None
            elif success_count == 0:
                # Collect error reasons
                reasons = []
                for run in provider["test_runs"]:
                    if run["status"] == "error" and run["error"]:
                        error = run["error"][:100]
                        if error not in reasons:
                            reasons.append(error)
                    elif run["status"] == "unclear":
                        reasons.append("Empty response")
                    elif run["status"] == "no_tool_call" or run["status"] == "invalid_json" or run["status"] == "invalid_schema":
                        if run["response_content"]:
                            reasons.append(f"No proper response: {run['response_content'][:50]}...")
                return "failure", f"{success_count}/3", reasons
            else:
                # Partial success - collect both successes and failures
                reasons = []
                for run in provider["test_runs"]:
                    if run["status"] != "success":
                        if run["status"] == "error" and run["error"]:
                            reasons.append(f"Error: {run['error'][:50]}...")
                        elif run["status"] == "unclear":
                            reasons.append("Empty response")
                        elif run["status"] == "no_tool_call" or run["status"] == "invalid_json" or run["status"] == "invalid_schema":
                            reasons.append("Invalid response format")
                return "partial", f"{success_count}/3", reasons
    
    return "none", "-", None


def format_reasons_for_tooltip(reasons):
    """Format reasons for tooltip, escaping HTML-sensitive characters."""
    if not reasons:
        return ""
    # Escape single quotes, double quotes, and ampersands for HTML attribute
    return " | ".join(reasons).replace("&", "&amp;").replace("'", "&apos;").replace("\"", "&quot;")


def generate_html(results):
    """Generate the HTML content for the website."""
    generated_at = datetime.fromisoformat(results["generated_at"]).strftime("%Y-%m-%d %H:%M UTC")
    
    grouped_models = group_models(results["models"])
    all_providers = get_all_providers(results)
    
    # Check if structured output data is available
    has_structured_data = has_structured_output_data(results)

    # CSS styles - add tab styles if we have structured output data
    style_sheet = f"""<style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.4; /* Further reduced line height */
            color: #333;
            background-color: #f4f7f9;
            margin: 0;
            padding: 5px; /* Further reduced padding */
        }}
        .container {{
            max-width: 99%; /* Maximize width */
            margin: 0 auto;
            padding: 10px; /* Further reduced padding */
            background-color: #fff;
            border-radius: 4px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        }}
        h1 {{
            font-size: 22px; /* Further reduced font size */
            margin-bottom: 4px; /* Further reduced margin */
            color: #2c3e50;
            text-align: center;
        }}
        .subtitle {{
            color: #555;
            font-size: 12px; /* Further reduced font size */
            margin-bottom: 10px; /* Further reduced margin */
            text-align: center;
        }}
        .legend {{
            margin-bottom: 10px; /* Further reduced margin */
            padding: 8px; /* Further reduced padding */
            background-color: #f8f9fa;
            border-radius: 3px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px 12px; /* Further reduced gap */
            justify-content: center;
            border: 1px solid #dee2e6;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 11px; /* Further reduced font size */
        }}
        .legend-color {{
            width: 12px; /* Smaller swatch */
            height: 12px; /* Smaller swatch */
            margin-right: 5px;
            border-radius: 2px;
        }}
        .legend-color.success-swatch {{ background-color: #d4edda; border: 1px solid #155724; }}
        .legend-color.partial-swatch {{ background-color: #fff3cd; border: 1px solid #856404; }}
        .legend-color.failure-swatch {{ background-color: #f8d7da; border: 1px solid #721c24; }}
        .legend-color.not-available-swatch {{ background-color: #e9ecef; border: 1px solid #adb5bd; }}

        .table-container {{
            overflow-x: auto;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px; /* Further reduced font size */
            min-width: 800px; /* Adjusted min-width */
        }}
        th, td {{
            padding: 4px 6px; /* Further reduced padding */
            text-align: left;
            border: 1px solid #e9ecef;
            vertical-align: top;
        }}
        th {{
            background-color: #f1f3f5;
            color: #343a40;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th.model-header {{
            min-width: 180px; /* Further reduced min-width */
            text-align: left;
            position: sticky;
            left: 0;
            z-index: 11;
            background-color: #e9ecef;
        }}
        th.provider-header {{
            writing-mode: vertical-rl;
            text-orientation: mixed;
            white-space: nowrap;
            text-align: center;
            min-width: 30px; /* Further reduced min-width */
            max-width: 35px; /* Further reduced max-width */
            height: 120px; /* Further reduced height */
            vertical-align: bottom;
            padding-bottom: 4px;
        }}
        tbody tr:hover {{
            background-color: #f8f9fa;
        }}
        .model-name-cell {{
            font-weight: 500;
            color: #2c3e50;
            background-color: #f8f9fa;
            position: sticky;
            left: 0;
            z-index: 5;
        }}
        .provider-cell {{
            text-align: center;
            min-width: 50px; /* Further reduced min-width */
        }}
        .variant-info {{
            padding: 1px 0; /* Further reduced padding */
            /* Consider removing border if labels are gone and only one item usually appears */
            /* border-bottom: 1px dashed #e0e0e0; */ 
        }}
        .variant-info:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}
        .variant-info:first-child {{
            padding-top: 0;
        }}
        /* .variant-label is no longer used for text, but class might be on div */
        
        .cell {{
            display: inline-block;
            padding: 1px 4px; /* Further reduced padding */
            border-radius: 2px;
            font-weight: bold;
            font-size: 10px; /* Further reduced font size */
            cursor: default; /* No longer help cursor as tooltip is gone */
            border: 1px solid transparent;
        }}
        .cell.success {{
            background-color: #d4edda;
            color: #155724;
            border-color: #c3e6cb;
        }}
        .cell.partial {{
            background-color: #fff3cd;
            color: #856404;
            border-color: #ffeeba;
        }}
        .cell.failure {{
            background-color: #f8d7da;
            color: #721c24;
            border-color: #f5c6cb;
        }}
        .cell.none {{
            color: #6c757d;
            font-weight: normal;
        }}
        .footer {{
            text-align: center;
            margin-top: 15px; /* Further reduced margin */
            font-size: 10px; /* Further reduced font size to 10px */
            color: #6c757d;
        }}
        .footer a {{
            color: #007bff;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Tab styles for switching between Tool Support and Structured Output */
        .tabs {{
            display: flex;
            margin-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        .tab {{
            padding: 8px 12px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            font-size: 14px;
            background-color: #f8f9fa;
            margin-right: 2px;
        }}
        .tab.active {{
            background-color: #fff;
            border-color: #dee2e6;
            border-bottom-color: white;
            margin-bottom: -1px;
            font-weight: 600;
            color: #2c3e50;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .tab-heading {{
            font-size: 16px;
            font-weight: 600;
            text-align: center;
            margin: 10px 0;
            color: #2c3e50;
        }}
    </style>"""

    # HTML Structure
    html_start = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenRouter Tool Support Matrix</title>
    {style_sheet}
</head>
<body>
    <div class="container">
        <h1>OpenRouter Support Matrix</h1>
        <p class="subtitle">Last updated: {generated_at}</p>
        
        <div class="legend">
            <div class="legend-item"><span class="legend-color success-swatch"></span>Full support (3/3)</div>
            <div class="legend-item"><span class="legend-color partial-swatch"></span>Partial support (1-2/3)</div>
            <div class="legend-item"><span class="legend-color failure-swatch"></span>No support (0/3)</div>
            <div class="legend-item"><span class="legend-color not-available-swatch"></span>Not available</div>
        </div>
"""

    # If we have structured output data, create tabs to toggle between reports
    if has_structured_data:
        html_start += """
        <div class="tabs">
            <div class="tab active" id="tab-tool-support">Tool Support</div>
            <div class="tab" id="tab-structured-output">Structured Output</div>
        </div>
        
        <div class="tab-content active" id="content-tool-support">
            <div class="tab-heading">Tool Support Results</div>
"""
    
    # Provider Headers - reused for both tables
    provider_headers = ""
    for provider_name in all_providers:
        provider_headers += f"<th class='provider-header'>{provider_name}</th>"

    # Start tool support table
    html_start += """
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th class="model-header">Model</th>
""" + provider_headers + """</tr>
                </thead>
                <tbody>
"""

    # Table Rows for Tool Support
    tool_support_rows_html = ""
    for base_name, model_variants in sorted(grouped_models.items()):
        tool_support_rows_html += f"<tr><td class='model-name-cell'>{base_name}</td>"
        
        for provider_name in all_providers:
            cell_content_parts = []
            found_any_for_provider = False

            def get_variant_sort_key(model_data_item):
                model_id_val = model_data_item["model_id"]
                if ":" not in model_id_val: return (0, model_id_val) # Standard first
                suffix = model_id_val.split(":")[-1].lower()
                if suffix == "free": return (1, suffix)
                return (2, suffix)
            
            sorted_variants = sorted(model_variants, key=get_variant_sort_key)

            for model_data in sorted_variants:
                model_id = model_data["model_id"]
                variant_suffix = 'standard' # Default, not displayed
                if ':' in model_id:
                    suffix_part = model_id.split(':')[-1]
                    if base_name != model_id: 
                         variant_suffix = suffix_part # Still useful for class if needed
                
                status, text, reasons = get_cell_status(model_data, provider_name, "tool_support")
                
                if status != "none":
                    found_any_for_provider = True
                    cell_content_parts.append(
                        f"<div class='variant-info variant-{variant_suffix.lower()}'>"
                        f"<span class='cell {status}'>{text}</span>"
                        f"</div>"
                    )
            
            if found_any_for_provider:
                tool_support_rows_html += f"<td class='provider-cell'>{''.join(cell_content_parts)}</td>"
            else:
                tool_support_rows_html += f"<td class='provider-cell'><span class='cell none'>-</span></td>"
        
        tool_support_rows_html += "</tr>"
    
    # Close tool support table
    tool_support_table_end = """
                </tbody>
            </table>
        </div>
"""
    
    # If we have structured output data, create a structured output table
    structured_output_html = ""
    if has_structured_data:
        structured_output_html = """
        </div> <!-- End tool support tab content -->
        
        <div class="tab-content" id="content-structured-output">
            <div class="tab-heading">Structured Output Results</div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th class="model-header">Model</th>
""" + provider_headers + """</tr>
                    </thead>
                    <tbody>
"""

        # Table Rows for Structured Output
        for base_name, model_variants in sorted(grouped_models.items()):
            structured_output_html += f"<tr><td class='model-name-cell'>{base_name}</td>"
            
            for provider_name in all_providers:
                cell_content_parts = []
                found_any_for_provider = False
                
                sorted_variants = sorted(model_variants, key=get_variant_sort_key)

                for model_data in sorted_variants:
                    model_id = model_data["model_id"]
                    variant_suffix = 'standard' # Default, not displayed
                    if ':' in model_id:
                        suffix_part = model_id.split(':')[-1]
                        if base_name != model_id: 
                             variant_suffix = suffix_part # Still useful for class if needed
                    
                    status, text, reasons = get_cell_status(model_data, provider_name, "structured_output")
                    
                    if status != "none":
                        found_any_for_provider = True
                        cell_content_parts.append(
                            f"<div class='variant-info variant-{variant_suffix.lower()}'>"
                            f"<span class='cell {status}'>{text}</span>"
                            f"</div>"
                        )
                
                if found_any_for_provider:
                    structured_output_html += f"<td class='provider-cell'>{''.join(cell_content_parts)}</td>"
                else:
                    structured_output_html += f"<td class='provider-cell'><span class='cell none'>-</span></td>"
            
            structured_output_html += "</tr>"
        
        # Close structured output table
        structured_output_html += """
                    </tbody>
                </table>
            </div>
        </div> <!-- End structured output tab content -->
"""
    
    # Footer content
    footer_html = f"""
        <br>
        Sometimes, the model (or the provider) does not properly call tools or return structured output. That's why every call is made three times.<br>
        The code to generate this website is available on <a href="https://github.com/Xeophon/openrouter-tool-check" target="_blank">GitHub</a>.
        
        <div class="footer">
            <p>Generated automatically by <a href="https://github.com/Xeophon/openrouter-tool-check" target="_blank">OpenRouter Tool Support Tracker</a>.</p>
            <p>Updates every 12 hours &bull; Data source: <a href="https://openrouter.ai/docs/api-reference" target="_blank">OpenRouter API</a></p>
        </div>
    </div>

    <!-- JavaScript for tab switching -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => {{
                tab.addEventListener('click', function() {{
                    // Remove active class from all tabs and content
                    tabs.forEach(t => t.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Show corresponding content
                    const contentId = this.id.replace('tab', 'content');
                    document.getElementById(contentId).classList.add('active');
                }});
            }});
        }});
    </script>
</body>
</html>"""

    # Combine all parts
    if has_structured_data:
        return html_start + tool_support_rows_html + tool_support_table_end + structured_output_html + footer_html
    else:
        return html_start + tool_support_rows_html + tool_support_table_end + footer_html


def main():
    """Main function to generate the website."""
    print("Loading test results...")
    results = load_latest_results()
    
    if not results:
        print("No results found. Run check_all_models.py first.")
        return
    
    print("Generating HTML...")
    html = generate_html(results)
    
    # Create output directory
    os.makedirs("docs", exist_ok=True)
    
    # Write HTML file
    output_file = "docs/index.html"
    with open(output_file, "w") as f:
        f.write(html)
    
    print(f"Website generated: {output_file}")


if __name__ == "__main__":
    main()
