bloom_instructions_objective = {
    "Knowledge": "Generate a question that requires students to recall or identify specific facts from the learning objective '{topic}'.",
    "Understanding": "Generate a question that asks students to explain or summarize the key ideas from the learning objective '{topic}'.",
    "Applying": "Generate a question that requires students to apply the knowledge from the learning objective '{topic}' to a real-world scenario.",
    "Analyzing": "Generate a question that asks students to compare and contrast or identify relationships between ideas related to the learning objective '{topic}'.",
    "Evaluating": "Generate a question that asks students to critique, justify, or make a judgment based on the information in the learning objective '{topic}'.",
    "Creating": "Generate a question that asks students to design or propose a new idea or solution using knowledge from the learning objective '{topic}'."
}

bloom_instructions_reference = {
    "Knowledge": "Generate a question that requires students to recall or identify specific facts from the article reference '{topic}'.",
    "Understanding": "Generate a question that asks students to explain or summarize the key ideas presented in the article reference '{topic}'.",
    "Applying": "Generate a question that requires students to apply the knowledge from the article reference '{topic}' to a real-world scenario.",
    "Analyzing": "Generate a question that asks students to compare and contrast or identify relationships between ideas discussed in the article reference '{topic}'.",
    "Evaluating": "Generate a question that asks students to critique, justify, or make a judgment based on the information in the article reference '{topic}'.",
    "Creating": "Generate a question that asks students to design or propose a new idea or solution using knowledge from the article reference '{topic}'."
}
def generate_quiz_question_prompt(topic, topic_type, bloom_level, rag_gen):
    """
    This function generates a quiz question prompt based on the given topic and
    skill level. It uses a RAG (Retrieval-Augmented Generation) model to gather
    facts and an LLM (Large Language Model) to generate the quiz question
    prompt.

    Parameters: topic (str): The topic for which the quiz question prompt needs
    to be generated. skill_level (str): The skill level of the students for
    which the quiz question prompt needs to be generated.
        It can be one of the following: "Knowledge", "Understanding",
        "Applying", "Analyzing", "Creating", "Evaluating".
    rag_gen: An instance of the RAG model used to gather facts.

    Returns: str: The generated quiz question prompt.

    Raises: ValueError: If the provided skill level is not one of the valid
    options.
    """
    
    assert bloom_level in bloom_instructions_objective, f"Invalid skill level: {bloom_level}"
    
    if topic_type == "concept":
        topic_preamble = ("Given the class's learning objective: " + topic +
                          "\n\n")
    elif topic_type == "article":   
        topic_preamble = "Given the required reading reference: " + topic + "\n\n"
    else:
        raise ValueError(f"Invalid topic type: {topic_type}")
        
    facts = rag_gen.query(topic_preamble + "Give me the ten key concepts a student needs to know.")
    # Construct the final prompt
    if topic_type == "concept":
        prompt = (
            topic_preamble +
            f"And the following facts: \n{facts}\n\n" +
            bloom_instructions_objective[bloom_level].format(topic=topic) +
            " Provide a detailed answer to this question a graduate student would respond with in 200 words or less."
        )
    elif topic_type == "article":
        prompt = (
            topic_preamble +
            f"And the following facts: \n{facts}\n\n" +
            bloom_instructions_reference[bloom_level].format(topic=topic) +
            " Provide a detailed answer to this question a graduate student would respond with in 200 words or less."
        )
    return prompt
