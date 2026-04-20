# =================================================================
# START bs4_router_path_extractor.py (Requires: pip install beautifulsoup4)
# =================================================================
import os
import json
import sys
from bs4 import BeautifulSoup, Tag
import re

# Define the directories to scan relative to where the script is run
TEMPLATE_DIRS = ['templates', 'templates/partials']

def normalize_path(path):
    """
    Cleans up a raw path string extracted from an attribute.
    Removes query strings, hash fragments, and leading/trailing slashes.
    """
    if not path or path.strip() in ['#', '']:
        return '' # Return empty string for root/current page, or ignore fragment links

    # Handle common Twig syntax like {{ path('route_name') }} which we can't parse here
    if '{{' in path and '}}' in path:
        return None

    # Remove protocol, domain, and port if present
    path = re.sub(r'https?://[^/]+', '', path, 1)

    # Remove everything after a question mark or hash fragment
    path = path.split('?')[0].split('#')[0]

    # Remove leading and trailing slashes for router compatibility
    return path.strip('/')

def extract_form_details(form_tag):
    """
    Extracts the action and all named inputs from a BeautifulSoup <form> tag.
    """
    raw_action = form_tag.get('action', '')
    clean_action = normalize_path(raw_action)
    
    # If action is None (e.g., Twig function) or empty (posts to current page), use the current page path placeholder
    if clean_action is None:
        return None, None
    if clean_action == '':
        clean_action = '/' # Represent the root or current path as '/' for POST

    input_names = set()

    # Find all elements that can submit data with a 'name' attribute
    # We look inside the form tag, not the whole file
    for element in form_tag.find_all(['input', 'textarea', 'select']):
        # Ignore submit buttons and reset buttons unless they have a specific name for logic
        input_type = element.get('type', '').lower()
        if input_type in ['submit', 'reset']:
             # We include the name if present, as it can be part of the POST payload
            if element.get('name'):
                 input_names.add(element['name'])
            continue

        name = element.get('name')
        if name:
            input_names.add(name)
            
    return clean_action, input_names

def extract_routes_from_file(filepath):
    """Reads a file and extracts all POST forms/variables and GET links."""
    post_routes = {}
    get_links = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            
            # --- 1. Extract POST Forms and Variables ---
            for form in soup.find_all('form', method=re.compile('post', re.I)):
                action_path, variables = extract_form_details(form)
                
                if action_path is not None and variables:
                    # Merge variables if the same action path is used in multiple forms
                    if action_path not in post_routes:
                        post_routes[action_path] = set()
                    post_routes[action_path].update(variables)

            # --- 2. Extract GET Links (<a> tags) ---
            for link in soup.find_all('a', href=True):
                raw_href = link['href']
                clean_path = normalize_path(raw_href)
                
                # Check for valid, non-twig paths
                if clean_path is not None and clean_path not in ['', '/']:
                    get_links.add(clean_path)

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading or parsing file {filepath}: {e}", file=sys.stderr)

    # Convert sets of variables back to sorted lists for the final output
    final_post_routes = {
        path: sorted(list(vars_set)) 
        for path, vars_set in post_routes.items()
    }

    return final_post_routes, get_links

def scan_directories(dirs):
    """Recursively scans the provided directories and aggregates results."""
    all_post_routes = {}
    all_get_links = set()

    for base_dir in dirs:
        print(f"Scanning directory: {base_dir}/", file=sys.stderr)
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory not found: {base_dir}", file=sys.stderr)
            continue

        for root, _, files in os.walk(base_dir):
            for file in files:
                # Process common template file extensions
                if file.endswith(('.html', '.htm', '.twig', '.php')):
                    filepath = os.path.join(root, file)
                    print(f"  Processing {filepath}", file=sys.stderr)
                    
                    post_routes, get_links = extract_routes_from_file(filepath)
                    
                    # Aggregate GET links
                    all_get_links.update(get_links)
                    
                    # Aggregate POST routes and merge variables
                    for path, vars_list in post_routes.items():
                        if path not in all_post_routes:
                            all_post_routes[path] = set()
                        all_post_routes[path].update(vars_list)
    
    # Finalize post routes: convert sets to sorted lists
    final_post_routes = {
        path: sorted(list(vars_set)) 
        for path, vars_set in all_post_routes.items()
    }
    
    return sorted(list(all_get_links)), final_post_routes

if __name__ == "__main__":
    get_links, post_routes = scan_directories(TEMPLATE_DIRS)
    
    # --- Print Summary to Stderr ---
    print("\n--- Detected UI Routes Summary ---", file=sys.stderr)
    
    # Print POST Routes (Structured for clarity)
    if post_routes:
        print("\n[POST Routes and Expected Variables (JSON Output)]", file=sys.stderr)
        # Print the structured data as a single JSON object to stdout for parsing
        print(json.dumps(post_routes, indent=4))
        print(f"\nFound {len(post_routes)} unique POST routes.", file=sys.stderr)
    else:
        print("No POST form actions found.", file=sys.stderr)

    # Print GET/Link Routes (Simple List)
    if get_links:
        print("\n[GET Routes / Simple Links]", file=sys.stderr)
        for path in get_links:
            print(path, file=sys.stderr)
        print(f"\nFound {len(get_links)} unique GET/Link routes.", file=sys.stderr)
    else:
        print("No relative GET/Link paths found.", file=sys.stderr)
    
    print("----------------------------------", file=sys.stderr)
