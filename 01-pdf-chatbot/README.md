# PDF Chatbot

A powerful Python-based chatbot that can answer questions about the content of PDF documents using advanced natural language processing and machine learning techniques.

## Features

- Extract text from PDF documents
- Process and chunk text for efficient analysis
- Use semantic search to find relevant content
- Generate accurate answers to questions about the PDF content
- Command-line interface for easy interaction

## Requirements

- Python 3.7+
- PyPDF2
- sentence-transformers
- transformers
- torch
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd 01-pdf-chatbot
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

You can use the chatbot in two ways:

### 1. Command Line Arguments

```bash
python main.py --pdf_path "path/to/your/document.pdf" --question "Your question here?"
```

### 2. Interactive Mode

Simply run the script and follow the prompts:

```bash
python main.py
```

The program will ask you to:
1. Enter the path to your PDF file
2. Enter your question about the PDF content

## How It Works

1. **Text Extraction**: The system extracts text from the provided PDF file using PyPDF2.
2. **Text Processing**: The extracted text is split into manageable chunks for efficient processing.
3. **Semantic Search**: When a question is asked, the system uses sentence transformers to find the most relevant text chunk.
4. **Answer Generation**: The relevant context and question are processed through a T5 model to generate an accurate answer.

## Technical Details

- Uses `google/flan-t5-xl` for question answering
- Employs `all-MiniLM-L6-v2` for semantic text similarity
- Implements efficient text chunking with configurable chunk sizes
- Handles PDF reading errors gracefully

## Error Handling

The system includes robust error handling for:
- Missing PDF files
- PDF reading errors
- Invalid input
- Processing errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

- PyPDF2 for PDF processing
- Hugging Face for the transformer models
- Sentence Transformers for semantic search capabilities 