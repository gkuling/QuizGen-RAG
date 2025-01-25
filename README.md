# README: RAG-Based Quiz Generation System

## Overview

This project provides a Retrieval-Augmented Generation (RAG)-based system for creating short-answer quiz questions tailored to educational settings. The primary goal is to transform educational content into formative assessments, helping educators evaluate student understanding and guide learning effectively. Designed for technical and non-technical users, the system leverages Large Language Models (LLMs) and vector-based retrieval to generate questions aligned with Bloom's Taxonomy levels.

## Features

1. **RAG-Based Question Generation**:
   - Extracts relevant facts from educational materials.
   - Generates short-answer quiz questions tailored to cognitive levels (Knowledge, Understanding, Applying, Analyzing, Evaluating, Creating).

2. **Customizable Course Integration**:
   - Generates quiz questions for specific weeks or topics.
   - Pulls from predefined learning objectives and reference materials.

3. **Modular Design**:
   - Core functionality is split across key scripts for ease of use and customization.

## File Structure

### Core Scripts

1. **`build_RAG.py`**:
   - Handles the creation of a vector-based retrieval index from educational materials (PDFs).
   - Uses Azure OpenAI embeddings and LLM configurations.
   - Creates or loads a `vec_store` for efficient querying.

2. **`generate_quiz.py`**:
   - Generates quiz questions based on the RAG model outputs and predefined learning objectives.
   - Supports output in JSON and CSV formats.
   - Accepts command-line arguments for customization.

### Supporting Scripts

1. **`BTQ_prompt.py`**:
   - Provides specialized prompts for generating questions at various Bloom's Taxonomy levels.
   - Defines the logic for structuring RAG and LLM prompts.

2. **`course_time_table.py`**:
   - Contains the course schedule, including weekly topics, concepts, and required readings.
   - Acts as a blueprint for quiz generation.

## Dependencies

1. **Python Libraries**:
   - `numpy`
   - `pandas`
   - `dotenv`
   - `argparse`
   - `llama_index`
2. **Azure OpenAI Services**:
   - API key and endpoint for LLM and embeddings.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Create a `.env` file in the root directory.
   - Add the following:
     ```env
     AZURE_OPENAI_API_KEY=<your_api_key>
     AZURE_OPENAI_ENDPOINT=<your_api_endpoint>
     AZURE_OPENAI_EMBEDDING_ENDPOINT=<your_embedding_endpoint>
     ```

## Usage

### Step 1: Build the RAG Model

Use the `build_RAG.py` script to parse PDFs and create a vector-based retrieval index:
```bash
python build_RAG.py --pdf_folder_path <path_to_pdf_folder>
```

### Step 2: Generate Quiz Questions

Run the `generate_quiz.py` script to create quiz questions:
```bash
python generate_quiz.py --week_number <week_number> --num_questions <number_of_questions> --output_local <output_directory>
```
- `--week_number`: The week from the course schedule to generate questions for.
- `--num_questions`: Total number of questions (leave as `-1` to generate for all topics).
- `--output_local`: Directory to save the output file.

### Example

Generate 5 questions for Week 3:
```bash
python generate_quiz.py --week_number 3 --num_questions 5 --output_local ./quizzes
```

## Extending the System

1. **Adding New Courses**:
   - Update `course_time_table.py` with your custom schedule and learning objectives.

2. **Custom Bloom’s Levels**:
   - Modify `BTQ_prompt.py` to adjust question templates or add new skill levels.

3. **Changing LLM Settings**:
   - Update `build_RAG.py` and `generate_quiz.py` to modify LLM or embedding configurations.

## Troubleshooting

1. **Missing Dependencies**:
   - Ensure all libraries in `requirements.txt` are installed.

2. **API Errors**:
   - Verify your `.env` file contains valid Azure OpenAI credentials.

3. **Empty Output**:
   - Check if the `vec_store` was correctly built with the `build_RAG.py` script.

## Future Directions

- Integration with a user-friendly web interface.
- Support for additional embedding models.
- Advanced analytics for student performance.

## License

MIT License. See `LICENSE` for more details.

## Contributors

- Grey Kuling (Lead Developer)
- Marinka Zitnik (Advisor)

---
Happy Teaching!

