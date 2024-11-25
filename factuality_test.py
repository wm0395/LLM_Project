import requests
import pandas as pd
from tqdm import tqdm

api_key = "fc607a0ed5a44a2faca3440d70b48351"  # Replace with actual API key

def check_factuality_with_claimbuster(statement):
    """
    Check the factuality of a statement using ClaimBuster API.

    Args:
    - statement (str): The statement to be fact-checked.

    Returns:
    - float: The factuality score between 0 and 1.
    """
    # Define the API endpoint and headers
    api_endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
    request_headers = {"x-api-key": api_key}

    # Prepare the payload (the text to be fact-checked)
    payload = {"input_text": statement}

    # Send POST request to ClaimBuster API
    api_response = requests.post(url=api_endpoint, json=payload, headers=request_headers)

    # Check if the response is valid
    if api_response.status_code == 200:
        # Parse the JSON response from the API
        response_json = api_response.json()
        # Extract and return the factuality score
        return response_json.get("score", 0)
    else:
        # Print error message if API request fails
        print(f"Error: {api_response.status_code}, {api_response.text}")
        return None

def process_csv(input_csv, output_csv, sample_size=100):
    """
    Process the input CSV to add a "FactualityScore" column based on the "input" column.

    Args:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file.
    - sample_size (int): Number of datapoints to sample.
    """
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Randomly sample 100 datapoints
    sampled_df = df.sample(n=sample_size, random_state=1)

    # Initialize an empty list to store the factuality scores
    factuality_scores = []

    # Iterate over the rows of the "input" column with a progress bar
    for statement in tqdm(sampled_df["counter_narrative"], desc="Processing statements"):
        # Generate the factuality score for the statement
        score = check_factuality_with_claimbuster(statement)
        # Append the score to the list
        factuality_scores.append(score)

    # Add the factuality scores as a new column in the sampled DataFrame
    sampled_df["FactualityScore"] = factuality_scores

    # Save the updated sampled DataFrame to a new CSV file
    sampled_df.to_csv(output_csv, index=False)
# Example usage
# input_csv_path = "./CONAN-master/CONAN/CONAN.csv"  # Replace with the path to your input CSV file
# output_csv_path = "./CONAN_fact.csv"  # Replace with the path to your output CSV file

input_csv_path = "./CONAN-master/multitarget_KN_grounded_CN/multitarget_KN_grounded_CN.csv"  # Replace with the path to your input CSV file
output_csv_path = "./KG_fact.csv"  # Replace with the path to your output CSV file
process_csv(input_csv_path, output_csv_path)
