import os
import chromadb
import json

import argparse

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.legacy import StorageContext
from llama_index.legacy.vector_stores import ChromaVectorStore
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

    vec_store = os.path.join(
            args.pdf_folder_path, "chroma_db"
        )

    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path=vec_store)

    # create collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store)
    # if os.path.exists(vec_store):
    #     # load your index from stored vectors
    #     index = VectorStoreIndex.from_vector_store(
    #         vector_store, storage_context=storage_context
    #     )
    #     print('Loaded data from data.json')
    # else:

    # documents = SimpleDirectoryReader(args.pdf_folder_path).load_data()
    # print('Parsed data:')
    # # create your index
    # index = VectorStoreIndex.from_documents(
    #     documents, storage_context=storage_context
    # )

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    print('Indexed data:')
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "What are 10 themes from 'KGAT: Knowledge Graph "
        "Attention Network for Recommendation' article? Give full quotes to "
        "support your answer.")
    print(response)



if __name__ == '__main__':
    main()