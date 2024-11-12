import random

from course_time_table import course_schedule
import os
import pandas as pd
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

Settings.llm = AzureOpenAI(
    engine="gpt-4o",
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_version="2024-05-01-preview",
)

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-3-small",
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.environ.get('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
    api_version='2023-05-15',
)

llm = Settings.llm

def generate_quiz(pdf_folder_path, week_number, num_questions, output_local):
    # Check if vec_store exists
    vec_store = os.path.join(
        pdf_folder_path, "vec_store"
    )

    print('Loading index from vec_store...')
    # load index from vec_store if it exists
    storage_context = StorageContext.from_defaults(
        persist_dir=vec_store)

    # load index
    index = load_index_from_storage(storage_context)
    # Function to generate quiz questions based on week number
    week_data = next((week for week in course_schedule["weeks"] if week["week"] == week_number), None)
    if not week_data:
        print(f"Week {week_number} not found in the course schedule.")
        return

    title = week_data["title"]
    concepts = week_data["concepts"]
    readings = week_data["required_readings"]

    questions_answers = []

    for i in range(num_questions):
        # Randomly select a concept and generate a question
        concept = random.choice(concepts)
        # query index
        query_engine = index.as_query_engine()
        query = f"What is {concept}?"
        facts = query_engine.query(query)

        query = (f"Given the information: {facts} \n\n Write me a quiz "
                 f"question-answer pair based on the concept {concept}. "
                 f"Format it as a python dictionary with keys 'question' and "
                 f"'answer' that will work with the eval() function." +
                 "For example: ```python \n {'question': 'What is the color "
                 "of a dove?, 'answer': 'White'}\n```")
        response = llm.complete(query)
        try:
            qa = eval(response.text.split('python')[-1].replace('`',''))
        except:
            print("Error in response:", response.choices[0].text)
            continue

        # Append question-answer pair
        questions_answers.append(qa)

    # Display the generated questions
    for i, qa in enumerate(questions_answers, 1):
        print(f"Q{i}: {qa['question']}")
        print(f"A{i}: {qa['answer']}")
        print()

    # Save to CSV file using pandas
    filename = os.path.join(output_local, f"week_{week_number}_quiz.csv")
    df = pd.DataFrame(questions_answers)
    df.to_csv(filename, index=False)

    print(f"Quiz questions saved to {filename}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate quiz questions '
                                                 'based on the course '
                                                 'schedule.')
    parser.add_argument('--week_number', type=int,
                        help='The week number for which to generate quiz '
                             'questions.')
    parser.add_argument('--num_questions', type=int, default=2,
                        help='The number of quiz questions to generate.')
    parser.add_argument('--pdf_folder_path', type=str,
                        help='Path to the folder containing the vec_store.')
    parser.add_argument('--output_local', type=str, default='.',)
    args = parser.parse_args()

    generate_quiz(args.pdf_folder_path, args.week_number, args.num_questions,
                  args.output_local)