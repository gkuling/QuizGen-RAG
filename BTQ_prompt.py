bloom_instructions = {
    "Knowledge": (
        "Generate a question that requires students to *recall, list, or identify* "
        "specific facts or definitions from the learning objective '{topic}'."
    ),
    "Understanding": (
        "Generate a question that asks students to *explain, interpret, or summarize* "
        "the key ideas from the learning objective '{topic}'."
    ),
    "Applying": (
        "Generate a question that requires students to *apply or demonstrate* knowledge "
        "from the learning objective '{topic}' in a real-world or practical scenario."
    ),
    "Analyzing": (
        "Generate a question that asks students to *compare, contrast, or break down* "
        "the relationships between ideas in the learning objective '{topic}'."
    ),
    "Evaluating": (
        "Generate a question that asks students to *critique, judge, or justify* "
        "their stance based on the information in the learning objective '{topic}'."
    ),
    "Creating": (
        "Generate a question that asks students to *design, propose, or construct* "
        "a new idea or solution using knowledge from the learning objective '{topic}'."
    )
}


def generate_quiz_question_prompt(obj, ref, level, rag_gen):

    assert level in bloom_instructions, f"Invalid skill level: {level}"

    # Step 1: Generate a prompt for the RAG model to extract key facts
    rag_prompt = (
        f"Reference:\n{ref}\n\n"
        f"Learning Objective:\n{obj}\n\n"
        f"Please return the top 10 facts directly relevant to the learning "
        f"objective above. Provide them as concise bullet points."
    )

    facts = rag_gen.query(rag_prompt)
    # Step 2: Generate a prompt for the LLM to create a quiz question

    # Retrieve the specialized instruction text for the specified Bloom level
    instruction_text = bloom_instructions.get(level, "").format(
        topic=obj)

    return f"""Reference: {ref}
    
    Learning Objective: {obj}
    
    Bloom's Taxonomy Level: {level}
    
    {instruction_text}
    
    Facts:
    {facts}
    """
