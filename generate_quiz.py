import re
from BTQ_prompt import generate_quiz_question_prompt
from course_time_table import course_schedule
import os
import pandas as pd
import argparse

import numpy as np
from llama_index.core import Settings, load_index_from_storage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import StorageContext
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.llms import ChatMessage

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

### Setting up the RAG Model for knowledge base

vec_store = r"C:\Project_Data\AIM2_project\course_pdf_data\vec_store"

print('Loading index from vec_store...')
# load index from vec_store if it exists
storage_context = StorageContext.from_defaults(
    persist_dir=vec_store)

# load index
rag = load_index_from_storage(storage_context)
rag = rag.as_query_engine()
print('Finished loading index from vec_store.')

def postprocess_llmresposne(response):
        # Extract the question and answer from the response
        reformat_prompt = "Given the quesiton and answer pair: \n\n" + \
            response + "\n\n" + \
                f"Generate a Python dictionary containing the question and answer pair. Please use the following format: \n\n" + \
                    "{'question': 'The Question', 'answer': 'The Answer'} \n"
        response2 = llm.chat([
            ChatMessage(role='system', content='You are a helpful assistant designed to output Python dictionaries.'),
            ChatMessage(role='user', content=reformat_prompt)
        ])      
        code_match = re.search(r"```python(.*?)```", response2.message.content, re.DOTALL)
        if code_match:
            try:
                return eval(code_match.group(1).strip())   
            except:
                return None
        else:
            return None
        
def generate_quiz(pdf_folder_path, week_number, num_questions, output_local):

    # get week data
    question_options = []
    week_data = next((week for week in course_schedule["weeks"] if week["week"]
    == week_number), None)
    week_data['topics'] = list(zip(['concept']*len(week_data['concepts']), 
                                   week_data['concepts'])) + \
        list(zip(['article']*len(week_data['required_readings']), 
                 week_data['required_readings']))
    # generate question options based on topics and skill levels
    for topic in week_data["topics"]:
        # Add different skill levels for each topic based on Bloom's Taxonomy
        for skill_level in ["Knowledge", "Understanding", "Applying",
                            "Analyzing", "Evaluating", "Creating"]:
            question_options.append(
                {'topic': topic[1], 'topic_type': topic[0], 'skill_level': skill_level}
                )
    # Calculate the number of times we need to repeat the list
    repeat_count = int(np.ceil(num_questions / len(question_options)))

    # Repeat the list and slice to get the exact target length
    question_options = (question_options * repeat_count)
    np.random.shuffle(question_options)
    question_options = question_options[:num_questions]

    questions_list = []
    for i, question in enumerate(question_options):
        concept = question['topic']
        taxonomy = question['skill_level']
        concept_type = question['topic_type']

        # Step 3: Generate Questions Based on Skill Level
        question_prompt = generate_quiz_question_prompt(
            concept, concept_type,  taxonomy, rag
        )

        # Step 4: Generate Questions and Answers
        response = llm.chat([
            ChatMessage(role='system', content='You are a teacher preparing a quiz for your students.'),
            ChatMessage(role='user', content=question_prompt)
        ])
        qa = None
        iter_cnt = 0
        while qa is None and iter_cnt < 5:
            qa = postprocess_llmresposne(response.message.content)
            iter_cnt += 1
        if qa is None:
            print(f"Failed to generate question and answer for concept: {concept}")
            continue
        questions_list.append(qa)
        print(f"Q{i+1}: {qa['question']}")
        print(f"A{i+1}: {qa['answer']}")
        print()


    # Save to CSV file using pandas
    filename = os.path.join(output_local, f"week_{week_number}_quiz.csv")
    df = pd.DataFrame(questions_list)
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