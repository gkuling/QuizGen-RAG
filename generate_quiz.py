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

### Setting up the LLM Model for question generation
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

vec_store = r"C:\Project_Data\AIM2_project\course_pdf_store\vec_store"

print('Loading index from vec_store...')
# load index from vec_store if it exists
storage_context = StorageContext.from_defaults(
    persist_dir=vec_store)

# load index
rag = load_index_from_storage(storage_context)
rag = rag.as_query_engine()
print('Finished loading index from vec_store.')

def generate_quiz(week_number, num_questions, output_local):

    # get week data
    question_options = []
    week_data = next((week for week in course_schedule["weeks"] if week["week"]
    == week_number), None)
    # generate question options based on topics and skill levels
    for reading in week_data["required_readings"]:
        for opjective in week_data['concepts']:

        # Add different skill levels for each topic based on Bloom's Taxonomy
            for skill_level in ["Knowledge", "Understanding", "Applying",
                                "Analyzing", "Evaluating", "Creating"]:
                question_options.append(
                    {'reference': reading,
                     'objective': opjective,
                     'skill_level': skill_level}
                    )

    # Calculate the number of times we need to repeat the list
    repeat_count = int(np.ceil(num_questions / len(question_options)) if
                       num_questions > 0 else 1)

    # Repeat the list and slice to get the exact target length
    question_options = (question_options * repeat_count)
    if num_questions > 0:
        np.random.shuffle(question_options)
        question_options = question_options[:num_questions]

    # declare a system prompt
    system_prompt = """
    You are an expert educator creating short-answer quiz questions for a 
    college-level course. These questions are formative assessments, checking 
    students’ current understanding to guide further learning, not a final, 
    summative evaluation.  

    Each question should be answerable in around 200 words or fewer.

    Use Bloom’s Taxonomy to vary cognitive complexity:
    1. Knowledge (Recall, identify)
    2. Understanding (Explain, summarize)
    3. Applying (Use concepts or methods in real contexts)
    4. Analyzing (Compare, contrast)
    5. Evaluating (Critique, judge)
    6. Creating (Design, propose)

    You will receive:
    - A *reference* (the learning material)
    - A *learning objective*
    - A *Bloom's level*
    - Key *facts* from the reference

    Your task:
    1. Generate exactly ONE question aligned with the specified Bloom’s level.
    2. Provide a concise, accurate answer (under 200 words).
    3. Base the Q&A on the provided facts.
    4. Format your output as a JSON:
    {
      "question": "...",
      "answer": "..."
    }
    No extra text.
    """

    questions_list = []
    for i, question in enumerate(question_options):
        objective = question['objective']
        taxonomy = question['skill_level']
        reference = question['reference']

        # Step 3: Generate Questions Based on Skill Level
        question_prompt = generate_quiz_question_prompt(
            objective, reference,  taxonomy, rag
        )

        # Step 4: Generate Questions and Answers
        response = llm.chat([
            ChatMessage(role='system', content=system_prompt),
            ChatMessage(role='user', content=question_prompt)
        ])
        try:
            qa = response.message.content
            qa = qa.split('```json')[-1].replace('```', '')
            qa = eval(qa)
        except:
            continue
        questions_list.append({
            'learning_objective': objective,
            'reference': reference,
            'taxonomy': taxonomy,
            'promtp': question_prompt,
            'question': qa['question'],
            'answer': qa['answer']
        })
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
    parser.add_argument('--num_questions', type=int, default=-1,
                        help='The number of quiz questions to generate.')
    parser.add_argument('--pdf_folder_path', type=str,
                        help='Path to the folder containing the vec_store.')
    parser.add_argument('--output_local', type=str, default='.',)
    args = parser.parse_args()

    generate_quiz(args.week_number,
                  args.num_questions,
                  args.output_local)