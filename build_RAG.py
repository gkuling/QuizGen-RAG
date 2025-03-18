import os

import argparse

import numpy as np
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, \
    load_index_from_storage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import StorageContext
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter

import dotenv
dotenv.load_dotenv('.env')
chunk_sz = 1024
overlap = int(np.round(chunk_sz*0.125))
questions = 2
key_name = 'AZURE_OPENAI_API_KEY_C'
Settings.llm = AzureOpenAI(
    engine="gpt-4o",
    api_key=os.environ.get(key_name),
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_version="2024-05-01-preview",
)

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-3-small",
    api_key=os.environ.get(key_name),
    azure_endpoint=os.environ.get('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
    api_version='2023-05-15',
)

def build_RAG():
    parser = argparse.ArgumentParser(description='Parse PDFs in a folder and save extracted text to a JSON file.')
    parser.add_argument('--pdf_folder_path', type=str, help='Path to the '
                                                           'folder containing PDFs to parse.')
    args = parser.parse_args()

    # Check if vec_store exists
    vec_store = os.path.join(
            args.pdf_folder_path, "vec_store"
        )

    if not os.path.exists(os.path.join(vec_store, "docstore.json")):
        # create vec_store if it doesn't exist from scratch
        if not os.path.exists(vec_store):
            os.makedirs(vec_store)

        # load documents with text splitter
        text_splitter = TokenTextSplitter(
            separator=" ", chunk_size=chunk_sz, chunk_overlap=overlap
        )
        print('Beginning document parsing...')
        documents = SimpleDirectoryReader(args.pdf_folder_path).load_data()
        print('Finished document parsing.')

        # create vector store
        print('Indexing data...')
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True,
            transformations=[text_splitter]
        )
        print('Finished indexing data.')

        # persist index to vec_store
        index.storage_context.persist(persist_dir=vec_store)
    else:
        print('Loading index from vec_store...')
        # load index from vec_store if it exists
        storage_context = StorageContext.from_defaults(
            persist_dir=vec_store)

        # load index
        index = load_index_from_storage(storage_context)

    print('Indexed data:')
    # query index
    query_engine = index.as_query_engine()
    query = "What are the main themes Artifical Intelligence in Medicine?"
    response = query_engine.query(query)
    print('Query:', query)
    print('-'*50)
    print('Response:', response)



if __name__ == '__main__':
    build_RAG()