# RAG Bot

A powerful Python-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about multiple PDF documents. This system combines advanced natural language processing, semantic search, and large language models to provide accurate answers based on document content.

## Features

- Process multiple PDF documents from a folder
- Extract and process text from PDFs efficiently
- Split text into semantic chunks for better context
- Use sentence transformers for semantic search
- Generate accurate answers using T5 model
- Command-line interface for easy interaction
- Detailed logging for monitoring and debugging

## Requirements

- Python 3.9+
- PyPDF2
- sentence-transformers
- transformers
- torch
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd 02-rag-bot
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

You can use the RAG bot in two ways:

### 1. Command Line Arguments

```bash
python main.py --folder_path "path/to/your/pdf/folder" --question "Your question here?"
```

### 2. Interactive Mode

Simply run the script and follow the prompts:

```bash
python main.py
```

The program will ask you to:
1. Enter the path to your folder containing PDF files
2. Enter your question about the PDF content

## How It Works

1. **PDF Processing**: The system processes all PDF files in the specified folder using PyPDF2.
2. **Text Extraction**: Text is extracted from each PDF and combined for processing.
3. **Text Chunking**: The combined text is split into semantic chunks for efficient processing.
4. **Embedding Generation**: Each text chunk is converted into embeddings using sentence transformers.
5. **Semantic Search**: When a question is asked, the system finds the most relevant chunks using cosine similarity.
6. **Answer Generation**: The relevant context and question are processed through a T5 model to generate an accurate answer.

## Technical Details

- Uses `google/flan-t5-xl` for question answering
- Employs `all-MiniLM-L6-v2` for semantic text similarity
- Implements efficient text chunking with configurable chunk sizes
- Processes multiple PDFs in a single folder
- Includes comprehensive logging for monitoring and debugging
- Handles PDF reading errors gracefully

## Error Handling

The system includes robust error handling for:
- Missing folders
- Invalid folder paths
- PDF reading errors
- No PDF files found in folder
- Processing errors
- Model generation errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

- PyPDF2 for PDF processing
- Hugging Face for the transformer models
- Sentence Transformers for semantic search capabilities
