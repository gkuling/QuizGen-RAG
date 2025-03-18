import re
from llama_index.core.llms import ChatMessage
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

import dotenv
dotenv.load_dotenv('.env')
key_name = 'AZURE_OPENAI_API_KEY_C'

engine = "gpt-4o-mini"
Settings.llm = AzureOpenAI(
    engine=engine,
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
llm = Settings.llm

vec_store = r"C:\Project_Data\ALAS_RAG\studentguides\vec_store"

print('Loading index from vec_store...')
# load index from vec_store if it exists
storage_context = StorageContext.from_defaults(
    persist_dir=vec_store)

# load index
rag = load_index_from_storage(storage_context)
rag = rag.as_query_engine()
print('Finished loading index from vec_store.')

agent_descriptions = {
    "Pedagogical Expert": (
        "You are an expert in Bloom’s Taxonomy and foundational learning objectives. "
        "Your deep understanding of hierarchical cognitive skills—from 'Knowledge' to 'Creating'—"
        "allows you to evaluate and align questions with the intended educational outcomes. "
        "Your role is to ensure that the question–answer pair challenges students appropriately, "
        "is clear, and adheres to proven pedagogical frameworks."
    ),
    "Content Accuracy": (
        "You are a medical science expert with knowledge equivalent to a first-year doctor's foundations course. "
        "You rigorously verify that all content is factually correct, evidence-based, and aligned with current scientific standards. "
        "Your insights ensure the question–answer pair reflects the highest level of accuracy and reliability."
    ),
    "Feedback Integration": (
        "You specialize in rubric building and assessment strategies, with a strong foundation in rewarding effort and encouraging growth. "
        "Drawing on Vygotsky’s ZPD (1978), Dweck (2006), Black & Wiliam (1998), Gay (2000), and OECD (2013), "
        "you focus on semantic clarity and inclusive grading practices. "
        "Your role is to integrate instructor feedback to refine the question–answer pair and enhance student learning."
    ),
    "Innovative Perspective": (
        "You bring a forward-thinking, education-focused outlook that challenges conventional methods. "
        "Your role is to propose creative strategies, leverage emerging technologies, and introduce interdisciplinary insights "
        "to invigorate the question–answer pair. By encouraging critical thinking and curiosity, you help transform standard assessments "
        "into dynamic, engaging learning experiences."
    )
}

def call_course_content_agent(qa_pair):
    """
    Uses the RAG model to retrieve relevant course content that can inform
    improvements to the given question–answer pair.
    """
    # Construct a query that directs the RAG model to retrieve relevant course context.
    query = (
        f"Retrieve relevant course material context to help improve the following short answer question and answer pair:\n"
        f"Question: {qa_pair['question']}\n"
        f"Answer: {qa_pair['answer']}\n"
        "Return a summary of the key concepts or facts that can be used to refine this Q&A."
    )
    course_context = rag.query(query)
    return course_context

def run_round_with_course_content_and_indepth_agents(
        qa_pair, feedback_report, round_number, context_discussion,
        course_context
):
    """
    Runs one round of discussion with all specialized agents, including the Course Content Agent.
    """

    suggestions = {}
    for role in agent_descriptions.keys():
        # Include the retrieved course context in each agent's prompt.
        prompt = f"""You are the {role} Agent.
You are tasked with reviewing the following short answer question and answer pair:
Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

The feedback report provided to the instructor (about the grade distribution of student answers) is:
"{feedback_report}"

The following course context was retrieved from the course materials:
{course_context}

{"Previous discussion: " + context_discussion if context_discussion else ""}

Based on your role, please provide a detailed suggestion for improving this question–answer pair.
Return your suggestion as plain text.
"""
        response = llm.chat([
            ChatMessage(
                role='system',
                content=agent_descriptions[role]),
            ChatMessage(role='user', content=prompt)
        ])
        suggestions[role] = response.message.content.strip()
    return suggestions
def run_round_with_course_content(qa_pair, feedback_report, round_number,
                                  context_discussion, course_context):
    """
    Runs one round of discussion with all specialized agents, including the Course Content Agent.
    """

    agent_roles = [
        "Pedagogical Expert",
        "Content Accuracy",
        "Feedback Integration",
        "Innovative Perspective"
    ]
    suggestions = {}
    for role in agent_roles:
        # Include the retrieved course context in each agent's prompt.
        prompt = f"""You are the {role} Agent.
You are tasked with reviewing the following short answer question and answer pair:
Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

The feedback report provided to the instructor (about the grade distribution of student answers) is:
"{feedback_report}"

The following course context was retrieved from the course materials:
{course_context}

{"Previous discussion: " + context_discussion if context_discussion else ""}

Based on your role, please provide a detailed suggestion for improving this question–answer pair.
Return your suggestion as plain text.
"""
        response = llm.chat([
            ChatMessage(role='system',
                        content='You are a helpful educational content review agent.'),
            ChatMessage(role='user', content=prompt)
        ])
        suggestions[role] = response.message.content.strip()
    return suggestions


def synthesize_suggestions(qa_pair, feedback_report, rounds=2):
    """
    Orchestrates multiple rounds of agent discussion and synthesizes the suggestions.
    """
    discussion_history = ""
    # First, get course context using the RAG model.
    course_content = call_course_content_agent(qa_pair)

    for i in range(rounds):
        suggestions = run_round_with_course_content_and_indepth_agents(
            qa_pair, feedback_report, i + 1,
            discussion_history, course_content
        )
        round_summary = f"Round {i + 1} suggestions:\n" + "\n".join(
            [f"{role}: {suggestions[role]}" for role in suggestions]) + "\n"
        discussion_history += round_summary + "\n"

    final_prompt = f"""You are a Question Writing Agent tasked with implementing the following suggestions for improving a short answer question and answer pair.

Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

Feedback Report (regarding the grade distribution of student answers): "{feedback_report}"

Below are the suggestions from multiple discussion rounds:
{discussion_history}

Please provide a list of 5 recommended question-answer pairs the user could use on a quiz for their students.
"""
    response = llm.chat([
        ChatMessage(role='system',
                    content='You are an expert synthesizer agent for educational content review.'),
        ChatMessage(role='user', content=final_prompt)
    ])
    return response.message.content.strip()


def generate_edit_suggestions(qa_pair, feedback_report):
    """
    Given a question–answer pair and a feedback report, this function returns a final list
    of suggested edits generated via a multi-agent round table discussion.
    """
    final_suggestions = synthesize_suggestions(qa_pair, feedback_report,
                                               rounds=2)
    return final_suggestions


# Example usage:
if __name__ == "__main__":
    # Input: a short answer question–answer pair and a feedback report
    qa_pair = {
        'question': 'What is the purpose of cancer screening? What are the key '
                    'elements that are required for cancer screening to be '
                    'effective?',
        'answer': 'The purpose of cancer screening is to identify and remove '
                  'precursor lesions before they become invasive carcinoma. '
                  'In order for screening to be effective, there must be (a) '
                  'a well-described and identifiable precursor lesion; (b) an '
                  'effective intervention (e.g., surgical excision); and (c) '
                  'sufficient time to perform the intervention before the '
                  'precursor lesion becomes invasive.'
    }
    feedback_report = (
        "The analysis of responses to the question on cancer screening reveals "
        "a diverse range of understanding across grade levels. There were no "
        "responses graded as 1, indicating a minimum baseline understanding "
        "among all students. Grade 2 responses show a basic comprehension of "
        "cancer screening's primary purpose for early detection but often lack "
        "detailed insights into the key elements necessary for effectiveness, "
        "such as identifiable precursor lesions and effective interventions. "
        "Some students confuse screening with broader health checks, "
        "underscoring a common misconception.\n\n"
        "Grade 3 responses exhibit a clearer grasp of early detection and the "
        "significance of precursor lesions, although there is still some "
        "confusion between prevention and detection. These responses touch on "
        "critical screening elements like sensitivity and methodology but "
        "require more depth and precision.\n\n"
        "Responses awarded a Grade 4 display a comprehensive and nuanced "
        "understanding, highlighting the importance of precursor lesions and "
        "lead time. They also include specific examples such as colonoscopy and "
        "Pap smears, which were largely absent from lower grades. Overall, while "
        "students generally grasp the importance of early detection, the depth "
        "and precision of their understanding improve significantly from Grades "
        "2 to 4.")

    suggestions = generate_edit_suggestions(qa_pair, feedback_report)
    print("Final Edit Suggestions:")
    print(suggestions)

    print('End of script.')