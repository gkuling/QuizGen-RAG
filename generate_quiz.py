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

def generate_quiz_question_prompt(topic, skill_level):
    if skill_level == "Knowledge":
        # ask RAG for a definition
        facts = rag.query(
            f"Give me a key definition that would be in a textbook for {topic}"
        )

        # Ask LLM to generate a knowledge-based question
        prompt = (
            f"Given the facts: \n{facts} \nGenerate a question and the answer that tests students' "
            f"knowledge on the definition of {topic}."
        )
        # Expected outcome: Questions like “What is the definition of [topic]?”
    elif skill_level == "Understanding":
        # ask RAG for a simple explanation
        facts = rag.query(
            f"Give me a simple explanation of {topic}"
        )
        # Ask LLM to generate a comprehension-based question
        prompt = (
            f"Given the facts: \n{facts} \nGenerate a question and the answer "
            f"that asks students to explain the concept of {topic} in their own words."
        )
        # Expected outcome: Questions like “Explain in your own words what [topic] means.”
    elif skill_level == "Applying":
        # ask RAG for a real-world scenario
        facts = rag.query(
            f"Give me a real-world scenario where {topic} is applied"
        )

        # Ask LLM to generate an application-based question
        prompt = (
            f"Given the facts: \n{facts} \nGenerate a question and the answer "
            f"that asks students how they would apply {topic} in a real-world "
            f"scenario based on the facts."
        )
        # Expected outcome: Questions like “How would you use [topic] to solve a problem in [relevant field]?”
    elif skill_level == "Analyzing":
        # ask RAG for a comparison
        facts = rag.query(
            f"Give me a comparison of {topic} and something similar. "
        )
        # Ask LLM to generate a question that requires analysis and multiple perspectives
        prompt = (
            f"Given the facts: \n{facts} \nGenerate a question and the answer asking students to compare and contrast different aspects of {topic}."
        )
        # Expected outcome: Questions like “What are the differences between [aspect 1] and [aspect 2] in [topic]? Which factors influence these differences?”
    elif skill_level == "Creating":
        # ask RAG for a design-based question
        facts = rag.query(
            f"Give me a scenario where {topic} is used to design something new."
        )
        # Ask LLM to generate a creative and design-based question
        prompt = (
            f"Given the facts: \n{facts} \nGenerate a question and the answer that asks students to design or propose a new solution based on {topic}."
        )
        # Expected outcome: Questions like “How would you design a solution for [specific problem] using [topic]? What challenges would you anticipate?”
    elif skill_level == "Evaluating":
        # ask RAG for an evaluation-based question
        facts = rag.query(
            f"Give me a scenario where {topic} is evaluated for its effectiveness."
        )
        # Ask LLM to generate an evaluation-based question
        prompt = (
            f"Given the facts: \n{facts} \nGenerate a question and the answer that asks students to evaluate the effectiveness of {topic} in a given context."
        )
        # Expected outcome: Questions like “How effective is [topic] in solving [specific problem]? What are the limitations?”
    else:
        raise ValueError(f"Invalid skill level: {skill_level}")
    return prompt
def generate_quiz(pdf_folder_path, week_number, num_questions, output_local):

    # get week data
    question_options = []
    week_data = next((week for week in course_schedule["weeks"] if week["week"] == week_number), None)
    week_data['topics'] = week_data['concepts'] + week_data['required_readings']
    # generate question options based on topics and skill levels
    for topic in week_data["topics"]:
        # Add different skill levels for each topic based on Bloom's Taxonomy
        for skill_level in ["Knowledge", "Understanding", "Applying",
                            "Analyzing", "Evaluating", "Creating"]:
            question_options.append(
                {'topic': topic, 'skill_level': skill_level}
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

        # Step 3: Generate Questions Based on Skill Level
        question_prompt = generate_quiz_question_prompt(concept, taxonomy)

        question_prompt += (
            ' \nFormat the response as a python dictionary. For example: {'
            '"question": "What is the capital of France?", "answer": "Paris"}'
        )

        # Step 4: Generate Questions and Answers
        response = llm.chat([
            ChatMessage(role='system', content='You are a teacher preparing a quiz for your students.'),
            ChatMessage(role='user', content=question_prompt)
        ])

        # Extract the question and answer from the response
        try:
            qa = eval(response.message.content.replace('`','').split('python')[-1])
        except:
            print(f"Error generating question {i+1}. Skipping...")
            continue
        questions_list.append(qa)
        print(f"Q{i}: {qa['question']}")
        print(f"A{i}: {qa['answer']}")
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