import os
import pdfreader
import json

import argparse

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

import dotenv
dotenv.load_dotenv('.env')

Settings.llm = AzureOpenAI(
    engine="gpt-4o-mini",
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_version="2024-05-01-preview",
)

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    deployment_name="text-embedding-3-small",
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
    api_version='2023-05-15',
)

def main():
    parser = argparse.ArgumentParser(description='Parse PDFs in a folder and save extracted text to a JSON file.')
    parser.add_argument('--pdf_folder_path', type=str, help='Path to the '
                                                           'folder containing PDFs to parse.')
    args = parser.parse_args()

    documents = SimpleDirectoryReader(args.pdf_folder_path).load_data()
    print('Parsed data:')
    index = VectorStoreIndex.from_documents(documents)
    print('Indexed data:')
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "What are the themes of 'KGAT: Knowledge Graph "
        "Attention Network for Recommendation' article?")
    print(response)

    with open('data.json', 'w') as f:
        json.dump(documents, f, indent=4)
    print('Data saved to data.json')

if __name__ == '__main__':
    main()