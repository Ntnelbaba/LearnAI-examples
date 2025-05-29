import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Try to extract article content
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    
    # Clean up whitespace
    return text.strip()

def summarize_text(text, max_chunk_length=1000):
    # Split the text into manageable chunks
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = [summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return "\n".join(summaries)

if __name__ == "__main__":
    url = input("Enter a URL to summarize:\n> ")
    print("\nFetching article content...\n")
    text = extract_text_from_url(url)
    
    if len(text) < 200:
        print("Article content is too short or could not be extracted.")
    else:
        print("\nSummarizing...\n")
        summary = summarize_text(text)
        print("=== Summary ===\n")
        print(summary)
