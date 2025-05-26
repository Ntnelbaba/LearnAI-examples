from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

qa_model = pipeline("text2text-generation", model="google/flan-t5-xl")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file is not found
        Exception: For other PDF reading errors
    """
    try:
        logger.info(f"Opening PDF file: {path}")
        # Create a PDF reader object
        reader = PdfReader(path)
        
        # Initialize an empty string to store the text
        text = ""
        total_pages = len(reader.pages)
        logger.info(f"Found {total_pages} pages in the PDF")
        
        # Iterate through all pages and extract text
        for i, page in enumerate(reader.pages, 1):
            logger.info(f"Processing page {i}/{total_pages}")
            text += page.extract_text()
            
        logger.info("Successfully extracted text from all pages")
        return text
        
    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {path}")
        raise FileNotFoundError(f"PDF file not found at path: {path}")
    except Exception as e:
        logger.error(f"Error reading PDF file: {str(e)}")
        raise Exception(f"Error reading PDF file: {str(e)}")

def split_text(text: str, max_chars: int = 1000) -> list[str]:
    logger.info("Splitting text into chunks")
    paragraphs = text.split('\n')
    chunks, current = [], ""
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

def find_most_relevant_chunk(chunks: list[str], question: str) -> str:
    logger.info("Finding most relevant text chunk for the question")
    logger.info(f"Number of chunks to search: {len(chunks)}")
    logger.info("Generating embeddings for text chunks")
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    logger.info("Generating embedding for the question")
    question_embedding = model.encode(question, convert_to_tensor=True)
    logger.info("Calculating similarity scores")
    scores = util.cos_sim(question_embedding, chunk_embeddings)
    best_idx = scores.argmax().item()
    best_score = scores[0][best_idx].item()
    logger.info(f"Found most relevant chunk (index: {best_idx}, similarity score: {best_score:.4f})")
    
    # Log a preview of the selected chunk
    selected_chunk = chunks[best_idx]
    preview = selected_chunk[:200] + "..." if len(selected_chunk) > 200 else selected_chunk
    logger.info(f"Selected chunk preview: {preview}")
    
    return chunks[best_idx]

def ask_question(context: str, question: str) -> str:
    logger.info("Generating answer using T5 model")
    logger.info(f"Context length: {len(context)} characters")
    logger.info(f"Question: {question}")
    
    # Clean and prepare the context
    context = context.strip()
    if len(context) > 1000:  # Truncate very long contexts
        logger.info("Context was too long, truncating to last 1000 characters")
        context = context[-1000:]
    
    prompt = f"Answer the question based on the context. Be specific and detailed.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    logger.info("Sending prompt to T5 model")
    
    try:
        response = qa_model(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
        logger.info(f"Raw model response: {response}")
        
        # Clean up the response
        response = response.strip()
        if not response or response == "[iv]":
            logger.warning("Model generated empty or invalid response, trying again with different parameters")
            response = qa_model(prompt, max_new_tokens=200, do_sample=True, temperature=0.9)[0]["generated_text"]
            response = response.strip()
        
        logger.info("Answer generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "I apologize, but I encountered an error while generating the answer. Please try rephrasing your question."

def chatbot_from_pdf(pdf_path: str, question: str) -> str:
    logger.info("Starting PDF chatbot process")
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    relevant = find_most_relevant_chunk(chunks, question)
    answer = ask_question(relevant, question)
    logger.info("PDF chatbot process completed")
    return answer

def main():
    parser = argparse.ArgumentParser(description='PDF Question Answering Chatbot')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--question', type=str, help='Question to ask about the PDF content')
    
    args = parser.parse_args()
    
    # Get PDF path if not provided
    pdf_path = args.pdf_path
    if not pdf_path:
        pdf_path = input("Enter the path to your PDF file: ").strip()
    
    # Get question if not provided
    question = args.question
    if not question:
        question = input("Enter your question about the PDF: ").strip()
    
    try:
        logger.info("Initializing PDF chatbot")
        answer = chatbot_from_pdf(pdf_path, question)
        print("\nAnswer:", answer)
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

