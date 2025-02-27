import re
import pandas as pd
from rapidfuzz import process, fuzz
from typing import List, Dict, Tuple, Optional
import logging
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import ast

# Configure the OpenAI client with the API key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(
    filename='debug.log',
    filemode='w',  # 'w' to overwrite file each run, 'a' to append
    format='%(asctime)s - %(message)s',
    level=logging.DEBUG
)

def debug_log(message: str):
    """Helper function to both log to file and print to console"""
    print(message)  # Still show in console
    logging.debug(message)  # Also write to file
    
def generation_filter_condition(x, lw_generation, lw_model):
    gen_x = get_ipad_generation(x)
    space_digit_x = bool(re.search(r"\s\d\s", normalize_generation_suffixes(x.lower())))
    space_digit_lw = bool(re.search(r"\s\d\s", lw_model.lower()))
    
    # First check generations match
    pass_filter = (gen_x == lw_generation) and (space_digit_x == space_digit_lw)
    
    # For iPad Pro models, also check screen sizes match
    if pass_filter and ("pro" in x.lower() or "pro" in lw_model.lower()):
        x_size = get_ipad_pro_screen_size(x)
        lw_size = get_ipad_pro_screen_size(lw_model)
        
        # If both have screen sizes, they must match
        if x_size is not None and lw_size is not None:
            pass_filter = (x_size == lw_size)
    
    # Log details for debugging
    if "12-inch" in x.lower() or "12 9inch" in x.lower() or " 12.9 " in x.lower():
        debug_log(f"[GEN_FILTER] x='{x}' | [LW_MODEL] lw_model='{lw_model}' | gen_x={gen_x}, lw_generation={lw_generation}, space_digit_x={space_digit_x}, space_digit_lw={space_digit_lw}, pass_filter={pass_filter}")
    
    return pass_filter

def normalize_generation_suffixes(name: str) -> str:
    """
    Convert generation suffixes like:
      - '1st' -> '1'
      - '2nd' -> '2'
      - '3rd' -> '3'
      - '4th' -> '4'
    and so on...
    """
    # This regex captures a digit followed by one of the common suffixes
    # (st|nd|rd|th), and replaces it with just the digit.
    return re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", name, flags=re.IGNORECASE)

def remove_brackets(text: str) -> str:
    """Remove bracketed text like "[Grade A]", "[Like New]" from the string."""
    return re.sub(r"\[.*?\]", "", str(text)).strip()

def normalize_name(name: str) -> str:
    """Normalize product names for better matching."""
    # Remove bracketed text
    name = remove_brackets(name)
    # Replace 'Wi-Fi', 'WIFI', etc. with 'Wifi'
    name = re.sub(r"\b(WIFI|Wi-Fi)\b", "Wifi", str(name), flags=re.IGNORECASE)
    # Remove extra spaces
    name = " ".join(str(name).split()).strip()
    return name

def get_grade_from_name(name: str) -> str:
    """Extract grade from product name."""
    grade_match = re.search(r"\[(Grade [AB]|Like New|Brand New|Open Box)\]", str(name))
    if grade_match:
        grade = grade_match.group(1)
        if grade == "Grade A": return "GA"
        elif grade == "Grade B": return "GB"
        elif grade == "Like New": return "LN"
        elif grade == "Brand New": return "BN"
        elif grade == "Open Box": return "OB"
    return None

def get_capacity(name: str) -> str:
    """
    Extract capacity like '128GB' or '1TB' from the string.
    Returns the capacity with its original unit (GB or TB).
    """
    # Try to match either TB or GB
    match = re.search(r"(\d+)(TB|GB)", str(name))
    return match.group(0) if match else None

def is_cellular_model(name: str) -> bool:
    """Check if the iPad model has cellular capability."""
    name = name.lower()
    return any(term in name for term in ['cellular', ' 4g', '5g'])

def is_wifi_only(name: str) -> bool:
    """Check if the iPad model is explicitly WiFi only."""
    name = name.lower()
    # If it has cellular capability (including 4G), it's not WiFi-only
    if is_cellular_model(name):
        return False
    # Must have WiFi in the name to be WiFi-only
    return 'wifi' in name

def get_ipad_model_type(name: str) -> str:
    """Get the iPad model type (air, pro, mini, or regular)"""
    name = name.lower()
    if "air" in name:
        return "air"
    elif "pro" in name:
        return "pro"
    elif "mini" in name:
        return "mini"
    else:
        return "regular"

def get_ipad_generation(name: str) -> Optional[int]:
    """
    Extract iPad generation number from name.
    Handles regular iPad, iPad Air, iPad Pro, and iPad Mini models.
    For iPad Pro, if no generation is specified, assumes 1st generation.
    """
    name = name.lower()
    model_type = get_ipad_model_type(name)
    
    # Different patterns for different iPad types
    if model_type == "air":
        patterns = [
            r"air\s+(\d+)\b",                     # Air 4
            r"air\s+(\d+)(?:th|rd|nd|st)",        # Air 4th
        ]
    elif model_type == "pro":
        patterns = [
            r"pro\s+(\d+)(?:th|rd|nd|st)\s+gen",  # Pro 4th gen
            r"pro\s+(\d+)(?:th|rd|nd|st)",        # Pro 5th
            r"pro.*?(\d+)(?:th|rd|nd|st)",        # Any position after Pro
            r"pro.*?\s(\d)\s",                    # Pro followed by space-padded number
            r"\s(\d)\s.*?pro",                    # Space-padded number followed by Pro
            r"\s(\d)\s.*?wifi",                   # Space-padded number followed by wifi
            r"pro\s+(\d+)\b(?![-.\s]+\d+\s*-inch)"
        ]
        gen_found = False
        for pattern in patterns:
            gen_match = re.search(pattern, name, flags=re.IGNORECASE)
            if gen_match:
                gen_num = int(gen_match.group(1))
                if gen_num == 11:
                    gen_num = 1
                gen_found = True
                return gen_num
        
        # If it's a Pro model and no generation specified, it's 1st gen
        if model_type == "pro" and not gen_found:
            return 1
            
    elif model_type == "mini":
        patterns = [
            r"mini\s+(\d+)\b",                    # Mini 6
            r"mini\s+(\d+)(?:th|rd|nd|st)",       # Mini 6th
        ]
    else:  # regular iPad
        patterns = [
            r"\b(\d+)(?:th|rd|nd|st)\b",          # 10th, 5th (standalone)
            r"(\d+)(?:th|rd|nd|st)",              # (5th)
            r"ipad\s+(\d+)\b",                    # iPad 6
            r"ipad\s+(\d+)(?:th|rd|nd|st)",       # iPad 10th
        ]
    
    # For non-Pro iPads, search through patterns
    for pattern in patterns:
        gen_match = re.search(pattern, name, flags=re.IGNORECASE)
        if gen_match:
            gen_num = int(gen_match.group(1))
            return gen_num
    
    return None

def get_ipad_year(name: str) -> Optional[int]:
    """Extract iPad year if specified (e.g., 2018, 2020)."""
    match = re.search(r"\(?20\d{2}\)?", name)
    return int(match.group().strip('()')) if match else None

def get_ipad_pro_screen_size(name: str) -> Optional[float]:
    """Extract iPad Pro screen size (e.g., 11, 12.9) from name."""
    name = name.lower()
    if "pro" not in name:
        return None
        
    # First normalize spaces in decimal numbers
    # Convert "12 9-inch" to "12.9-inch" and "10 5-inch" to "10.5-inch"
    name = re.sub(r'\b(\d+)\s+(\d+)((?:-inch|["″])?)\b', r'\1.\2\3', name)
    
    # Look for common screen sizes
    patterns = [
        r"(\d+\.?\d?)[\-\s]?inch",  # Matches "11-inch", "12.9-inch", "11 inch", "12.9inch"
        r"(\d+\.?\d?)[\"″]",        # Matches 11", 12.9″ 
        r"(?<=\s)(\d+\.\d+)(?=\s)"  # Matches " 12.9 " (number with a dot surrounded by spaces)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        # Log the conversion for debugging
        if match:
            try:
                size = float(match.group(1))
                # Handle special cases where the size might be malformed
                if "12 9" in name or "12.912" in name:
                    size = 12.9
                elif "10 5" in name or "10.510" in name:
                    size = 10.5
                return size
            except ValueError:
                continue
    return None

def apply_custom_rules(lw_model: str, pr_name: str, rules_enabled: bool = True) -> str:
    """
    Apply custom matching rules to normalize product names.
    Can be enabled/disabled using rules_enabled parameter.
    """
    if not rules_enabled:
        return pr_name

    # Convert to lowercase for case-insensitive matching
    lw_model = lw_model.lower()
    pr_name = pr_name.lower()
    
    # Rule 1: Handle 4G/Cellular matching
    if "cellular" in pr_name and "4g" in lw_model:
        pr_name = pr_name.replace("cellular", "4g")
    
    if "cellular" not in pr_name and "4g" not in lw_model:
        pr_name = pr_name.replace("cellular", "").replace(" 4g", "")
        lw_model = lw_model.replace("cellular", "").replace(" 4g", "")
    
    # Rule 2: Remove common words that don't affect matching
    words_to_remove = ['wifi', 'gen', 'generation']
    for word in words_to_remove:
        pr_name = pr_name.replace(word, '')
        lw_model = lw_model.replace(word, '')
    
    # Rule 3: Standardize spaces
    pr_name = ' '.join(pr_name.split())
    
    return pr_name

def normalize_with_cellular_at_end(name: str) -> str:
    """Normalize name but ensure cellular/4G terms are at the end."""
    name = name.lower()
    # Remove cellular/4G from middle and add to end
    for term in [' cellular', ' 4g', ' 5g']:
        if term in name:
            name = name.replace(term, '') + term
    return name

def should_use_flexible_matching(model_type: str, model_name: str) -> bool:
    """
    Determine if we should use flexible generation matching for this model.
    Easy to extend for other iPad types in the future.
    """
    if model_type == "pro" in model_name.lower():
        return True
    return False

def reposition_generation_in_model(name: str) -> str:
    """
    For iPad Pro models (11-inch and 12.9-inch) with a space-padded generation number,
    reposition the number to appear right after 'Pro'.
    Examples: 
        Input:  "Apple iPad Pro 11-inch 2 Cellular 128GB"
        Output: "Apple iPad Pro 2 11-inch Cellular 128GB"
        Input:  "Apple iPad Pro 12.9-inch 2 Cellular 256GB"
        Output: "Apple iPad Pro 2 12.9-inch Cellular 256GB"
    """
    # Look for the pattern: <anything>pro<space>11-inch/12.9-inch<space><digit><space><anything>
    pattern = r"(.*?pro\s+)(?:11|12(?:[.\s]+)9|10(?:[.\s]+)5|9(?:[.\s]+)7)-inch(\s+)(\d+)(\s+.*)"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        # Reorder the parts: prefix + number + screen size + suffix
        prefix, space1, number, suffix = match.groups()
        # Extract the screen size from the original string since it wasn't captured in a group
        screen_size = re.search(r"(?:11|12\.9)-inch", name, re.IGNORECASE).group()
        return f"{prefix}{number} {screen_size}{suffix}"
    return name

def match_products(lw_model: str, pr_products: pd.DataFrame, threshold: int = 85, rules_enabled: bool = True) -> Tuple[List[str], float, Dict[str, int]]:
    """
    Match a LW product with PR products using fuzzy matching.
    Returns matched products, match percentage, and grade counts.
    """
    # First validate that required columns exist
    required_columns = ['Name', 'Status', 'Qty']
    missing_columns = [col for col in required_columns if col not in pr_products.columns]
    if missing_columns:
        debug_log(f"Error: Missing required columns in DataFrame: {missing_columns}")
        return [], 0, {"BN": 0, "GA": 0, "GB": 0, "LN": 0, "OB": 0}
    
    # Handle NaN values in the Name column
    pr_products = pr_products.dropna(subset=['Name'])
    
    normalized_lw = normalize_name(lw_model)
    lw_capacity = get_capacity(normalized_lw)
    lw_has_cellular = is_cellular_model(lw_model)
    lw_year = get_ipad_year(lw_model)
    
    # Get iPad generation, model type and screen size from LW model
    debug_log(f"LW model: {lw_model} | normalized: {normalized_lw} | capacity: {lw_capacity} | has Cellular: {lw_has_cellular} | year: {lw_year}")
    lw_generation = get_ipad_generation(lw_model)
    lw_model_type = get_ipad_model_type(lw_model)
    lw_screen_size = get_ipad_pro_screen_size(lw_model)
    
    # Base filtering by capacity, model type, and ensure it's an iPad
    base_filtered = pr_products[
        (pr_products['Name'].str.contains('iPad', case=False, na=False)) &
        (pr_products['Name'].apply(lambda x: get_capacity(x) == lw_capacity)) &
        (pr_products['Name'].apply(lambda x: get_ipad_model_type(x) == lw_model_type))
    ]
    
    if base_filtered.empty:
        return [], 0, {"BN": 0, "GA": 0, "GB": 0, "LN": 0, "OB": 0}
    
    # For iPad Pro models, filter by screen size
    if lw_model_type == "pro" and lw_screen_size is not None:
        debug_log(f"Filtering iPad Pro by screen size: {lw_screen_size}-inch")
        base_filtered = base_filtered[
            base_filtered['Name'].apply(lambda x: get_ipad_pro_screen_size(x) == lw_screen_size)
        ]
        
        if base_filtered.empty:
            debug_log(f"No matches found after screen size filter for: {lw_model}")
            return [], 0, {"BN": 0, "GA": 0, "GB": 0, "LN": 0, "OB": 0}
    
    # Initialize pr_products_filtered with base_filtered
    pr_products_filtered = base_filtered.copy()
    
    # Further filter by generation or year
    if lw_model_type == "pro" and lw_year is not None:
        pr_products_filtered = base_filtered[
            base_filtered['Name'].apply(lambda x: get_ipad_year(x) == lw_year)
        ]
    elif lw_generation is not None:
        # For Pro models, be extra strict about generation matching
        if lw_model_type == "pro":
            # First try with original format
            pr_products_filtered = base_filtered[
                base_filtered['Name'].apply(lambda x: generation_filter_condition(x, lw_generation, lw_model))
            ]
            
            # If no matches and it's an 11-inch or 12.9-inch Pro model, try with repositioned generation
            if pr_products_filtered.empty and any(size in lw_model.lower() for size in ["11-inch","12.9-inch","12.9","10.5-inch","10.5","9.7-inch","9.7"]):
                # Reposition the generation number in the model name
                alt_lw_model = reposition_generation_in_model(lw_model)
                # Get the generation from the repositioned model
                alt_generation = get_ipad_generation(alt_lw_model)
                if alt_generation is not None:
                    # Try matching with the alternative generation
                    pr_products_filtered = base_filtered[
                        base_filtered['Name'].apply(lambda x: get_ipad_generation(x) == alt_generation)
                    ]
        else:
            pr_products_filtered = base_filtered[
                base_filtered['Name'].apply(lambda x: get_ipad_generation(x) == lw_generation)
            ]
    
    # Filter by cellular capability
    before_filter = len(pr_products_filtered)
    if lw_has_cellular:
        pr_products_filtered = pr_products_filtered[
            pr_products_filtered['Name'].apply(is_cellular_model)
        ]
    else:
        pr_products_filtered = pr_products_filtered[
            ~pr_products_filtered['Name'].apply(is_cellular_model)
        ]
    
    if pr_products_filtered.empty:
        return [], 0, {"BN": 0, "GA": 0, "GB": 0, "LN": 0, "OB": 0}
    
    # Get normalized PR names without grades
    pr_names = pr_products_filtered['Name'].apply(normalize_name).tolist()
    
    # First attempt: normal matching
    matches = []
    best_ratio = 0
    
    for idx, pr_name in enumerate(pr_names):
        pr_gen = get_ipad_generation(pr_products_filtered.iloc[idx]['Name'])
        
        # Apply custom rules if enabled
        normalized_pr = apply_custom_rules(normalized_lw, pr_name, rules_enabled)
        normalized_lw_rules = apply_custom_rules(normalized_lw, normalized_lw, rules_enabled)
        
        ratio = fuzz.ratio(normalized_lw_rules, normalized_pr)
        
        if ratio >= threshold:
            matches.append(pr_products_filtered.iloc[idx])
            best_ratio = max(best_ratio, ratio)
    
    # Second attempt: if no matches and LW has cellular/4G in middle, try with cellular at end
    if not matches and lw_has_cellular and any(term in lw_model.lower() for term in [' cellular ', ' 4g ', ' 5g ']):
        normalized_lw = normalize_with_cellular_at_end(lw_model)
        
        for idx, pr_name in enumerate(pr_names):
            pr_gen = get_ipad_generation(pr_products_filtered.iloc[idx]['Name'])
            
            # Normalize PR name with cellular at end too
            normalized_pr = normalize_with_cellular_at_end(pr_name)
            
            # Apply custom rules if enabled
            normalized_pr = apply_custom_rules(normalized_lw, normalized_pr, rules_enabled)
            normalized_lw_rules = apply_custom_rules(normalized_lw, normalized_lw, rules_enabled)
            
            ratio = fuzz.ratio(normalized_lw_rules, normalized_pr)
            
            if ratio >= threshold:
                matches.append(pr_products_filtered.iloc[idx])
                best_ratio = max(best_ratio, ratio)
    
    # Count quantities by grade
    grade_counts = {"BN": 0, "GA": 0, "GB": 0, "LN": 0, "OB": 0}
    matched_products = []
    
    for match in matches:
        grade = get_grade_from_name(match['Name'])
        if grade and match['Status'] == 1:  # Only count in-stock items
            grade_counts[grade] += match['Qty']
            matched_products.append(match['Name'])
    
    # Format grade counts string
    grading_count = f"BN:{grade_counts.get('BN', 0)},GA:{grade_counts.get('GA', 0)},GB:{grade_counts.get('GB', 0)},OB:{grade_counts.get('OB', 0)},LN:{grade_counts.get('LN', 0)}"
    
    # Calculate total
    total_qty = sum(grade_counts.values())
    
    return matched_products, best_ratio, grade_counts

def process_ipads(lw_df=None, pr_df=None, rules_enabled: bool = True):
    """
    Process iPad models from uploaded files and find matches.
    
    Parameters:
        lw_df: DataFrame from LIKEWISE file upload (optional, will read from file if not provided)
        pr_df: DataFrame from PRODUCTS file upload (optional, will read from file if not provided)
        rules_enabled: Whether to apply custom matching rules
    """
    try:
        # Use provided DataFrames or read from files as fallback
        if lw_df is None:
            lw_df = pd.read_excel('LIKEWISE.xlsx', sheet_name='iPad')
        if pr_df is None:
            pr_df = pd.read_excel('PRODUCTS.xlsx')
        
        # Validate required columns in LIKEWISE DataFrame
        if 'Model' not in lw_df.columns:
            raise KeyError("Required column 'Model' not found in LIKEWISE file")
            
        # Validate required columns in PRODUCTS DataFrame
        required_columns = ['Name', 'Status', 'Qty']
        missing_columns = [col for col in required_columns if col not in pr_df.columns]
        if missing_columns:
            raise KeyError(f"Required columns {missing_columns} not found in PRODUCTS file")
        
        # Filter and show iPad products for debugging
        ipad_products = pr_df[pr_df['Name'].str.contains('iPad', case=False, na=False)]
        
        # Initialize new columns
        lw_df['Match Type'] = ''
        lw_df['Matched Products'] = ''
        lw_df['Match Percentage'] = 0.0
        
        # Process each LW product
        for idx, row in lw_df.iterrows():
            if pd.isna(row['Model']):  # Skip empty rows
                continue
            matched_products, match_percentage, grade_counts = match_products(row['Model'], pr_df, rules_enabled=rules_enabled)
            
            total_qty = sum(grade_counts.values())
            # Only include non-zero quantities in the grading count string
            non_zero_grades = [f"{grade}:{count}" for grade, count in grade_counts.items() if count > -1]
            grading_count = ",".join(non_zero_grades) if non_zero_grades else "BN:0"  # Default to BN:0 if no grades have quantities
            
            lw_df.loc[idx, 'Match Type'] = 'Fuzzy Match'
            lw_df.loc[idx, 'Matched Products'] = ', '.join(matched_products)
            lw_df.loc[idx, 'Total'] = total_qty
            lw_df.loc[idx, 'Grading Count'] = grading_count
            lw_df.loc[idx, 'Match Percentage'] = match_percentage
        
        # Save the results
        output_file = 'LW-Updated-ipads-with-rules.xlsx' if rules_enabled else 'LW-Updated-ipads-no-rules.xlsx'
        lw_df.to_excel(output_file, index=False)
        
        # Process unmatched products using GPT
        result_df = process_unmatched_products(lw_df, pr_df)
        
        return result_df  # Return the processed DataFrame for API response
        
    except FileNotFoundError as e:
        debug_log(f"Error: Could not find Excel file - {str(e)}")
        raise
    except KeyError as e:
        debug_log(f"Error: {str(e)}")
        raise
    except Exception as e:
        debug_log(f"Unexpected error: {str(e)}")
        raise

def process_unmatched_products(lw_products, pr_products):
    """Process products that didn't get matched and extract their information using GPT."""
    try:
        # Clear the debug log first
        with open('debug.log', 'w') as f:
            f.write('')
        
        # Find unmatched products - those with empty 'Matched Products'
        unmatched_mask = (lw_products['Matched Products'].isna()) | (lw_products['Matched Products'] == '')
        unmatched_indices = lw_products[unmatched_mask].index
        
        if len(unmatched_indices) == 0:
            debug_log("No unmatched products found.")
            return lw_products
            
        debug_log(f"Found {len(unmatched_indices)} unmatched products.")
        
        # Create a copy of the DataFrame to avoid modifying the original during iteration
        updated_products = lw_products.copy()
        
        # Process unmatched products
        for idx in unmatched_indices:
            try:
                product_name = updated_products.loc[idx, 'Model']
                debug_log(f"\nProcessing unmatched product: {product_name}")
                
                # Get GPT info
                info = extract_product_info_gpt(product_name)
                if info:
                    debug_log("Extracted Information:")
                    for key, value in info.items():
                        debug_log(f"{key}: {value}")
                    
                    # Create standardized name
                    standardized_name = create_standardized_name(info)
                    if standardized_name:
                        debug_log(f"Created standardized name: {standardized_name}")
                        
                        # Try matching again with standardized name
                        matches, match_percentage, grade_counts = match_products(
                            standardized_name, 
                            pr_products,
                            threshold=85,
                            rules_enabled=False
                        )
                        
                        if matches:
                            debug_log(f"Found matches using standardized name!")
                            # Format grade counts string
                            grading_count = f"BN:{grade_counts.get('BN', 0)},GA:{grade_counts.get('GA', 0)},GB:{grade_counts.get('GB', 0)},OB:{grade_counts.get('OB', 0)},LN:{grade_counts.get('LN', 0)}"
                            
                            # Calculate total
                            total_qty = sum(grade_counts.values())
                            
                            # Update each column individually
                            updated_products.at[idx, 'Matched Products'] = ', '.join(matches)
                            updated_products.at[idx, 'Match Percentage'] = float(match_percentage)
                            updated_products.at[idx, 'Grading Count'] = grading_count
                            updated_products.at[idx, 'Total'] = total_qty
                        else:
                            debug_log("No matches found with standardized name")
                else:
                    debug_log("Failed to extract information")
            except Exception as e:
                debug_log(f"Error processing product at index {idx}: {str(e)}")
                continue
                
        # Save updated matches to new file
        output_file = 'LW-Updated-ipads-with-rules.xlsx'
        updated_products.to_excel(output_file, index=False)
        debug_log(f"Saved updated matches to {output_file}")
        
        return updated_products
                
    except Exception as e:
        debug_log(f"Error in process_unmatched_products: {str(e)}")
        return lw_products

def extract_product_info_gpt(product_name):
    """Extract product information from a product name string using ChatGPT."""
    try:
        debug_log(f"Sending request to GPT for product: {product_name}")
        
        system_prompt = '''You are a product information extraction assistant specialized in iPad products.
                    Extract information in the exact format requested. Be precise and only include information that is explicitly present.
                    Always return the Generation value as a string with the suffix, like "4th", "5th", etc.'''
                    
        user_prompt = f'''Extract the following information from this product name: "{product_name}"
                    Return ONLY a Python dictionary in this exact format:
                    {{
                        "Manufacturer": "manufacturer name",
                        "Model": "model name (e.g. iPad Pro, iPad Air, iPad Mini)",
                        "Screen_Size": screen size number only (e.g. 12.9, 11),
                        "Generation": "generation with suffix as string (e.g. '4th', '5th')",
                        "Storage": "storage number in GB or TB (always with correct unit like 256GB or 1TB)",
                        "Cellular": true/false (should only be true if the product has explicitely mentioned Cellular, 4G, 3G, 5G),
                        "Wifi": true/false (should only be true if the product has explicitely mentioned Wifi OR Wifi OR wi-fi)
                    }}'''

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        response_text = response.choices[0].message.content.strip()
        debug_log(f"GPT Response: {response_text}")

        # Try multiple parsing approaches
        parsed_dict = None
        
        # First attempt: Try json.loads
        try:
            import json
            # Replace Python bool literals with JSON bool literals
            json_text = response_text.replace('True', 'true').replace('False', 'false')
            parsed_dict = json.loads(json_text)
            debug_log("Successfully parsed using json.loads")
        except json.JSONDecodeError as e:
            debug_log(f"Failed to parse using json.loads: {str(e)}")
            
            # Second attempt: Clean and try ast.literal_eval
            try:
                cleaned_text = response_text.strip()
                parsed_dict = ast.literal_eval(cleaned_text)
                debug_log("Successfully parsed using ast.literal_eval")
            except (ValueError, SyntaxError) as e2:
                debug_log(f"Failed to parse using ast.literal_eval: {str(e2)}")
                
                # Third attempt: Use regex with careful boolean handling
                try:
                    import re
                    dict_pattern = r'\{[^}]+\}'
                    dict_match = re.search(dict_pattern, response_text)
                    if dict_match:
                        dict_str = dict_match.group(0)
                        # Normalize boolean values
                        dict_str = dict_str.replace('true', 'True').replace('false', 'False')
                        parsed_dict = ast.literal_eval(dict_str)
                        debug_log("Successfully parsed using regex and ast.literal_eval")
                except Exception as e3:
                    debug_log(f"Failed to parse using regex: {str(e3)}")
                    return None

        if parsed_dict:
            # Ensure boolean values are proper Python booleans
            if isinstance(parsed_dict, dict):
                for key in ['Cellular', 'Wifi']:
                    if key in parsed_dict:
                        if isinstance(parsed_dict[key], str):
                            parsed_dict[key] = parsed_dict[key].lower() == 'true'
            
            debug_log(f"Final parsed dictionary: {parsed_dict}")
            
            # Log extracted information
            debug_log("Extracted Information:")
            for key, value in parsed_dict.items():
                debug_log(f"{key}: {value}")
            
            return parsed_dict
        else:
            debug_log("Failed to parse GPT response into dictionary")
            return None

    except Exception as api_error:
        debug_log(f"OpenAI API error: {str(api_error)}")
        return None

def create_standardized_name(info: Dict) -> Optional[str]:
    """
    Create a standardized product name from GPT-extracted information.
    Format: "<Manufacturer> <Model> <Screen_Size>-inch <Generation>th Gen (<Storage>GB) [Connectivity]"
    Example: "Apple iPad Pro 12.9-inch 3rd Gen (256GB) [Wifi + Cellular]"
    """
    try:
        # Validate required fields
        required = ['Manufacturer', 'Model', 'Generation', 'Storage']
        if not all(key in info and info[key] for key in required):
            debug_log(f"Missing required fields. Have: {list(info.keys())}")
            return None
            
        # Build the name components
        components = []
        
        # Add manufacturer and model
        components.extend([info['Manufacturer'], info['Model']])
        
        # Add screen size if present
        if info.get('Screen_Size'):
            components.append(f"{info['Screen_Size']}-inch")
            
        # Add generation (already in ordinal form from GPT)
        components.append(f"{info['Generation']} Gen")
        
        # Add storage (now includes unit from GPT)
        storage = f"({info['Storage']})"
        
        # Add connectivity
        connectivity = []
        if info.get('Wifi', True):
            connectivity.append('Wifi')
        if info.get('Cellular', False):
            connectivity.append('Cellular')
        
        # Combine all parts
        name = ' '.join(components)
        name = f"{name} {storage}"
        if connectivity:
            name = f"{name} [{' + '.join(connectivity)}]"
            
        return name
        
    except Exception as e:
        debug_log(f"Error creating standardized name: {str(e)}")
        return None

def format_product_name(gpt_response: Dict[str, str]) -> str:
    """
    Format product name based on GPT-extracted attributes
    Expected format: "Manufacturer Model Screen-size Generation (Storage)"
    Example: "Apple iPad Pro 12.9-inch 4th Gen (128GB)"
    """
    # Extract attributes with defaults
    manufacturer = gpt_response.get('Manufacturer', '')
    model = gpt_response.get('Model', '')
    screen_size = gpt_response.get('Screen Size', '')
    generation = gpt_response.get('Generation', '')
    storage = gpt_response.get('Storage', '')

    # Clean and standardize attributes
    if screen_size and not screen_size.lower().endswith('inch'):
        screen_size = f"{screen_size}-inch"

    # Convert generation number to ordinal if it's just a number
    if generation:
        generation = generation.strip()
        match = re.search(r'\d+', generation)
        if match:
            num = match.group()
            generation = generation.replace(num, convert_to_ordinal(num))
        if not generation.lower().endswith('gen'):
            generation = f"{generation} Gen"

    # Format storage
    if storage and not storage.endswith('B'):
        storage = f"{storage}B"

    # Build the formatted name
    components = [manufacturer, model, screen_size, generation]
    base_name = ' '.join(filter(bool, components))
    
    if storage:
        return f"{base_name} ({storage})"
    return base_name

def validate_gpt_response(response: Dict[str, str]) -> bool:
    """
    Validate that the GPT response contains the required fields
    """
    required_fields = ['Manufacturer', 'Model']
    return all(field in response and response[field].strip() for field in required_fields)

def process_gpt_response(response: Dict[str, str]) -> Optional[str]:
    """
    Process GPT response and return formatted product name
    Returns None if validation fails
    """
    if not validate_gpt_response(response):
        return None
        
    return format_product_name(response)

if __name__ == "__main__":
    # You can run with or without custom rules
    process_ipads(rules_enabled=True)  # Set to False to disable custom rules
