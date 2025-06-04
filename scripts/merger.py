import json
import re

def merge_campedui_files(components_file_path, prompts_file_path, output_file_path):
    """
    Merges CampedUI components documentation with prompts to create a comprehensive structure.
    
    Args:
        components_file_path: Path to the components JSON file (campedui_components_playwright_codeclick.json)
        prompts_file_path: Path to the prompts JSON file (PROMPTS_CAMPEDUI_CODE.json)
        output_file_path: Path for the merged output file
    """
    
    # Load the files
    with open(components_file_path, 'r', encoding='utf-8') as f:
        components_data = json.load(f)
    
    with open(prompts_file_path, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # Create a mapping of prompts by component and section
    prompts_map = {}
    for prompt_item in prompts_data:
        component = prompt_item['component']
        section = prompt_item['section']
        key = f"{component}::{section}"
        
        if key not in prompts_map:
            prompts_map[key] = []
        prompts_map[key].append({
            'prompt': prompt_item['prompt'],
            'section_detail': section
        })
    
    # Merged result
    merged_data = []
    
    def extract_code_from_snippets(snippets):
        """Extract and clean code from pre_snippets or code_snippets"""
        if not snippets:
            return ""
        return "\n\n".join(snippets)
    
    def normalize_section_name(section_name):
        """Normalize section names to match between files"""
        return section_name.strip().replace("→", "→").replace("->", "→")
    
    # Process each component
    for component in components_data:
        if component.get('error'):
            continue
            
        component_name = component['name']
        
        # Process each section
        for section in component['sections']:
            section_name = section['section']
            
            # Get code from pre_snippets
            code = extract_code_from_snippets(section.get('pre_snippets', []))
            
            # Check for matching prompts
            section_key = f"{component_name}::{section_name}"
            if section_key in prompts_map:
                for prompt_data in prompts_map[section_key]:
                    merged_data.append({
                        "component": component_name,
                        "section": section_name,
                        "prompt": prompt_data['prompt'],
                        "code": code
                    })
            
            # Process subsections
            for subsection in section.get('subsections', []):
                subsection_name = subsection['subsection']
                full_section = f"{section_name} → {subsection_name}"
                subsection_code = extract_code_from_snippets(subsection.get('code_snippets', []))
                
                # Check for matching prompts with subsection
                subsection_key = f"{component_name}::{full_section}"
                if subsection_key in prompts_map:
                    for prompt_data in prompts_map[subsection_key]:
                        merged_data.append({
                            "component": component_name,
                            "section": full_section,
                            "prompt": prompt_data['prompt'],
                            "code": subsection_code if subsection_code else code
                        })
    
    # Add any prompts that didn't match (in case there are extras)
    processed_keys = set()
    for item in merged_data:
        key = f"{item['component']}::{item['section']}"
        processed_keys.add(key)
    
    for prompt_item in prompts_data:
        key = f"{prompt_item['component']}::{prompt_item['section']}"
        if key not in processed_keys:
            merged_data.append({
                "component": prompt_item['component'],
                "section": prompt_item['section'],
                "prompt": prompt_item['prompt'],
                "code": "// Code snippet not found in documentation"
            })
    
    # Sort by component name and section for better organization
    merged_data.sort(key=lambda x: (x['component'], x['section']))
    
    # Save the merged file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully merged {len(merged_data)} items")
    print(f"Output saved to: {output_file_path}")
    
    return merged_data

def analyze_files(components_file_path, prompts_file_path):
    """
    Analyze both files to show structure and help with debugging
    """
    print("=== ANALYZING FILES ===\n")
    
    # Analyze components file
    with open(components_file_path, 'r', encoding='utf-8') as f:
        components_data = json.load(f)
    
    print(f"Components file has {len(components_data)} components:")
    for comp in components_data[:3]:  # Show first 3 as example
        print(f"  - {comp['name']}: {len(comp['sections'])} sections")
        for section in comp['sections'][:2]:  # Show first 2 sections
            print(f"    - {section['section']}: {len(section.get('subsections', []))} subsections")
    
    # Analyze prompts file
    with open(prompts_file_path, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    print(f"\nPrompts file has {len(prompts_data)} prompts:")
    components_in_prompts = {}
    for prompt in prompts_data:
        comp = prompt['component']
        if comp not in components_in_prompts:
            components_in_prompts[comp] = []
        components_in_prompts[comp].append(prompt['section'])
    
    for comp, sections in list(components_in_prompts.items())[:3]:
        print(f"  - {comp}: {sections}")

# Example usage
if __name__ == "__main__":
    # File paths - update these to match your actual file locations
    components_file = "data/campedui_components_playwright_codeclick.json"
    prompts_file = "data/PROMPTS_CAMPEDUI_CODE.json"
    output_file = "data/merged_campedui_components.json"
    
    try:
        # Analyze files first (optional)
        print("Analyzing input files...")
        analyze_files(components_file, prompts_file)
        
        print("\n" + "="*50)
        print("Starting merge process...")
        
        # Perform the merge
        merged_result = merge_campedui_files(components_file, prompts_file, output_file)
        
        print(f"\nMerge completed successfully!")
        print(f"Created {len(merged_result)} merged entries")
        
        # Show a sample of the merged data
        print("\nSample merged entry:")
        if merged_result:
            sample = merged_result[0]
            print(json.dumps(sample, indent=2))
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please make sure both input files exist in the current directory")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")