# AI-Powered Search Suggestion

This project is an AI-driven solution that enhances search functionality by suggesting relevant search terms based on user input. It utilizes FastAPI for the web framework, the Sentence Transformer model for semantic understanding, and the OpenAI GPT-4 model for refining suggestions.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoint](#api-endpoint)
- [Technologies Used](#technologies-used)
- [License](#license)

## Features
- Fetches search terms from a CSV file hosted online.
- Generates embeddings for search terms using a Sentence Transformer model.
- Provides the top 10 relevant search term suggestions based on cosine similarity.
- Utilizes OpenAI's GPT-4 model to refine suggestions and select the best match.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Raheesraha/project.git
   cd project
2.Install the required dependencies:
   pip install fastapi[all] sentence-transformers openai pandas requests
3.Set your OpenAI API key as an environment variable:
   export OPENAI_API_KEY='your_openai_api_key' 

## Usage

To run the application, use the following command:
   uvicorn main:app --reload

## API Endpoint

You can send a POST request to the /suggest/ endpoint 
with the following JSON payload:
Request:

{
  "query": "your_search_term"
}

## Technologies Used

    FastAPI: For building the web application.
    Sentence Transformers: For generating semantic embeddings.
    OpenAI GPT-4: For selecting the most relevant search term.
    Pandas: For data manipulation and CSV file handling.
    Requests: For making HTTP requests.

## License

This project is licensed under the MIT License.
