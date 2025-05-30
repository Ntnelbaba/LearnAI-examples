import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk

# Load summarization model and tokenizer
model_id = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def extract_text_from_url(url):
    """
    Extracts visible text content from the <p> tags of a web page.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    return text.strip()

def split_text_semantically(text, max_tokens=1024):
    """
    Split text into semantic chunks (based on sentence boundaries),
    while ensuring each chunk does not exceed the token limit.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        tentative_chunk = (current_chunk + " " + sentence).strip()
        input_ids = tokenizer.encode(tentative_chunk, truncation=False)

        if len(input_ids) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence  # Start new chunk
        else:
            current_chunk = tentative_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def summarize_text(text):
    """
    Splits text into semantic chunks and summarizes each chunk individually.
    """
    print("Summarizing with semantic chunking...\n")
    chunks = split_text_semantically(text)
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"→ Summarizing chunk {i+1}/{len(chunks)} ({len(tokenizer.encode(chunk))} tokens)")
        summary = summarizer(chunk, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    return "\n".join(summaries)

if __name__ == "__main__":
    # Download NLTK punkt tokenizer
    nltk.download("punkt")

    url = input("Enter a URL to summarize:\n> ")
    print("\nFetching article content...\n")
    text = extract_text_from_url(url)

    if len(text) < 200:
        print("Article content is too short or could not be extracted.")
    else:
        summary = summarize_text(text)
        print("\n=== Summary ===\n")
        print(summary)
