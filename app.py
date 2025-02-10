import ollama
import json
import time

# Define LLaMA model
MODEL = "llama3"

# Function to extract email details
def extract_email_details(email_text):
    start_time = time.time()  # Start execution time

    # Define the prompt
    prompt = f"""
    You are a text classification and extraction model. You will be provided with an email content in plain text. 
    Your task is to extract the following information in valid JSON format:

    - is_from_IndiaMart (Yes or No based on whether the email indicates it's from IndiaMart)
    - person_name (Full name of the client)
    - company_name (If mentioned, otherwise return "")
    - email (Valid email address, if not found return "")
    - city (City name)
    - state (State name)
    - location (City, State if available)
    - mobile_number (Valid phone number)
    - product (Product mentioned in the email)

    Return **only** a valid JSON object.

    Input Email:
    {email_text}
    """

    # Get response from LLaMA
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])

    # Extract content from response
    raw_output = response['message']['content']

    try:
        # Convert raw string output to JSON format
        structured_output = json.loads(raw_output)
    except json.JSONDecodeError:
        # If parsing fails, clean up text response and reformat as JSON
        structured_output = {
            "error": "Failed to parse model response",
            "raw_output": raw_output
        }

    # End execution time
    execution_time = time.time() - start_time
    structured_output["execution_time"] = round(execution_time, 4)  # Add execution time

    return structured_output


# Example email input (can be replaced with dynamic input)
email_text = """
Buy Lead through IndiaMART

Buyer's Contact Details:
Phone ✓ Email ✓
PRANAVA MUTHU
Pandian Traders, Theni, TN
Click to call: +91-6383218808
E-mail: pranavamuthu@gmail.com

Member Since: 7 months
Buylead Details:
Karla Katta
Material: Iron
Finish: Polished
"""

# Run the function
output = extract_email_details(email_text)

# Print the structured JSON output
print(json.dumps(output, indent=4))

