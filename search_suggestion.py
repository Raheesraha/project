from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import requests
from io import StringIO
import re
import os

# Fetch search terms from the URL
url = "https://assets.danubehome.com/media/danubehome_search_terms.csv"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Load the CSV data into a DataFrame
    search_terms_df = pd.read_csv(StringIO(response.text))
else:
    raise Exception(f"Failed to fetch CSV file from {url}. Status code: {response.status_code}")

# Initialize the FastAPI app
app = FastAPI()

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Cache the search term embeddings to avoid repeated computation
search_terms = search_terms_df['searchTerm'].tolist()
search_term_embeddings = model.encode(search_terms)

# Define a Pydantic model for the request body
class Query(BaseModel):
    query: str

@app.post('/suggest/')
async def suggest(query: Query) -> str:  # Specify return type
    # Get the input query
    input_query = query.query

    # Generate embedding for the input query (without 'await')
    query_embedding = model.encode([input_query])

    # Calculate cosine similarity between the input query and cached search term embeddings
    similarities = np.dot(search_term_embeddings, query_embedding.T).flatten()
    top_n_indices = np.argsort(similarities)[-10:][::-1]  # Get top 10 indices for faster GPT-4 processing

    # Get top 10 semantic suggestions
    top_semantic_suggestions = [search_terms[i] for i in top_n_indices]

    # Setup prompt for GPT-4
    gpt_prompt = f"Select the best match for '{input_query}' from these options: {', '.join(top_semantic_suggestions)}."

    # Load API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        # Call the OpenAI API to get the best suggestion
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.2,  # Set temperature to 0.2 for more deterministic responses
            messages=[
                {"role": "system", "content": "You are a helpful assistant that selects the most relevant search term from a list."},
                {"role": "user", "content": gpt_prompt}
            ]
        )
        # Extract only the term from the response
        best_suggestion = response['choices'][0]['message']['content'].strip()
        # Use regex to remove any extra text and extract only the term
        best_suggestion = re.sub(r"^.*is\s+'?\"?([^'\".]+)'?\"?.*$", r"\1", best_suggestion)

        # If the best suggestion is empty or "not available", fallback to the first top suggestion
        if not best_suggestion or best_suggestion.lower() == "not available":
            best_suggestion = top_semantic_suggestions[0]

    except Exception as e:
        # Handle OpenAI API error
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

    return best_suggestion  # Return just the best suggestion as plain text
