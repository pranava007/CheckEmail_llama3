import ollama
import json
import time

# Define LLaMA model
MODEL = "llama3"

# Function to interact with LLaMA and return JSON output
def extract_email_details(email_text):
    # Start execution time
    start_time = time.time()

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


# Example email input
email_text = input()

# Run the function
output = extract_email_details(email_text)

# Print the structured JSON output
print(json.dumps(output, indent=4))
