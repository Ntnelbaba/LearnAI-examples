import os
from typing import List
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import argparse
import logging
import sys
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ----------- Step 1: Read multiple PDFs and extract text -----------

def extract_text_from_pdfs(pdf_paths: List[str]) -> str:
    """
    Reads multiple PDF files and extracts all text.
    """
    full_text = ""
    for path in pdf_paths:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
        logger.info(f"Opening PDF file: {path}")
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)
            logger.info(f"Found {total_pages} pages in the PDF")
            for i, page in enumerate(reader.pages, 1):
                logger.info(f"Processing page {i}/{total_pages}")
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    logger.info("Successfully extracted text from all PDFs")
    return full_text


# ----------- Step 2: Split text into manageable chunks -----------

def split_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Splits text into chunks of up to `max_chars`, using paragraph breaks.
    """
    logger.info("Splitting text into chunks")
    paragraphs = text.split('\n')
    chunks = []
    current = ""
    total_chars = len(text)
    processed_chars = 0
    
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += p + '\n'
        else:
            chunks.append(current.strip())
            current = p + '\n'
        processed_chars += len(p)
        progress = (processed_chars / total_chars) * 100
        logger.info(f"Text processing progress: {progress:.1f}%")
    
    if current:
        chunks.append(current.strip())
    
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks


# ----------- Step 3: Create embeddings for all chunks -----------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Free and fast

def embed_chunks(chunks: List[str]):
    """
    Creates embeddings for each chunk using SentenceTransformer.
    """
    logger.info("Creating embeddings for text chunks")
    return embedding_model.encode(chunks, convert_to_tensor=True)


# ----------- Step 4: Retrieve top-k similar chunks -----------

def get_top_k_chunks(question: str, chunks: List[str], embeddings, k: int = 3) -> List[str]:
    """
    Finds the top-k most relevant chunks to the input question.
    """
    logger.info("Finding most relevant text chunks for the question")
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)[0]
    top_k = torch.topk(scores, k=k)
    logger.info(f"Retrieved top {k} most relevant chunks")
    return [chunks[i] for i in top_k.indices]


# ----------- Step 5: Generate an answer from the context -----------

qa_model = pipeline("text2text-generation", model="google/flan-t5-xl")

def generate_answer(context_chunks: List[str], question: str) -> str:
    """
    Generates an answer using a language model and the relevant context.
    """
    logger.info("Generating answer using T5 model")
    context = "\n".join(context_chunks)
    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    try:
        output = qa_model(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
        logger.info("Answer generated successfully")
        return output.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "I apologize, but I encountered an error while generating the answer. Please try rephrasing your question."


# ----------- Step 6: Full pipeline function -----------

def run_rag_bot(pdf_paths: List[str], question: str) -> str:
    """
    Complete RAG pipeline: from PDFs and a question to an answer.
    """
    logger.info("Starting RAG bot process")
    text = extract_text_from_pdfs(pdf_paths)
    chunks = split_text(text)
    embeddings = embed_chunks(chunks)
    top_chunks = get_top_k_chunks(question, chunks, embeddings, k=3)
    answer = generate_answer(top_chunks, question)
    logger.info("RAG bot process completed")
    return answer

def extract_pdfs_from_folder(folder_path: str) -> List[str]:
    """
    Extracts all PDF files from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
        
    Returns:
        List[str]: List of paths to PDF files found in the folder
        
    Raises:
        FileNotFoundError: If the folder doesn't exist
        ValueError: If no PDF files are found in the folder
    """
    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not os.path.isdir(folder_path):
        logger.error(f"Path is not a directory: {folder_path}")
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Get all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in folder: {folder_path}")
        raise ValueError(f"No PDF files found in folder: {folder_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF files in folder: {folder_path}")
    for pdf_file in pdf_files:
        logger.info(f"Found PDF: {pdf_file}")
    
    return pdf_files

def main():
    parser = argparse.ArgumentParser(description='RAG-based PDF Question Answering Bot')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing PDF files')
    parser.add_argument('--question', type=str, help='Question to ask about the PDF content')
    
    args = parser.parse_args()
    
    # Get folder path if not provided
    folder_path = args.folder_path
    if not folder_path:
        folder_path = input("Enter the path to your folder containing PDF files: ").strip()
    
    # Get question if not provided
    question = args.question
    if not question:
        question = input("Enter your question about the PDFs: ").strip()
    
    try:
        logger.info("Initializing RAG bot")
        pdf_paths = extract_pdfs_from_folder(folder_path)
        answer = run_rag_bot(pdf_paths, question)
        print("\nAnswer:", answer)
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
    except ValueError as e:
        logger.error(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
