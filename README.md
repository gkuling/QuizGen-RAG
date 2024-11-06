# QuizGen-RAG


### Overview
**QuizGen-RAG** is a Retrieval-Augmented Generation (RAG) model designed to generate quiz questions from a database of research articles or educational materials. The model is capable of producing short-answer questions at varying levels of difficulty, providing a flexible solution for automated quiz generation across diverse fields of study.

### Project Goals
- **Text Extraction**: Extract structured text from research articles and educational PDFs.
- **Content Retrieval**: Build a searchable database of course content, enabling targeted retrieval for question generation.
- **Quiz Question Generation**: Generate quiz questions with customizable difficulty levels, adaptable to specific learning objectives.
- **Browser-Based Interface**: Create a user-friendly web app for students and educators to access quizzes and engage with course content.

### Folder Structure
- **/data/pdf_content/**: Folder for storing PDF files for processing.
- **/data/processed_text/**: Processed text extracted from the PDFs.
- **/app/**: Code for the browser-based interface.
- **/scripts/**: Scripts for content extraction, question generation, and database management.

### Getting Started
1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/QuizGen-RAG.git
cd QuizGen-RAG
```
2. **Install Dependencies**: Ensure Python 3.x is installed, and install required libraries:
```bash
Copy code
pip install -r requirements.txt
```
3. **Extract Text from PDFs**: Place PDF files in /data/pdf_content/ and run the extraction script:
```bash
Copy code
python scripts/extract_text.py
```

Generate Quiz Questions: Feature coming soon.

### Future Development
Integration with Knowledge Tracing to enable adaptive question selection.
User Interface for accessing quizzes via a web application.

### License

MIT License

