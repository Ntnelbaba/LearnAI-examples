# Web Article Summarization Agent 📝🔍

This is an AI agent that uses a transformer model to automatically summarize web articles. It extracts content from any given URL and generates a concise summary using a pre-trained model. The agent intelligently splits long articles into semantic chunks for better summarization quality.

## ✨ Features

- Extracts text content from web articles using BeautifulSoup
- Intelligently splits text into semantic chunks using NLTK
- Processes long articles in chunks while maintaining context
- Uses a lightweight pre-trained model (`distilbart-cnn-12-6`) for summarization
- Simple command-line interface for easy interaction

## 🧰 Requirements

- Python 3.9+
- Install dependencies:
  
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```text
transformers==4.40.1
torch>=2.1.0
beautifulsoup4==4.12.3
requests==2.31.0
nltk==3.8.1
```

## 🚀 How to Run

Run the agent from the terminal:

```bash
python main.py
```

Then input a URL when prompted:

```text
Enter a URL to summarize:
> https://example.com/article
```

The agent will:
1. Fetch the article content
2. Split it into semantic chunks
3. Summarize each chunk
4. Combine the summaries into a final result

## 📂 Project Structure

```
.
├── main.py              # Main script that runs the summarization agent
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Features

* **Web Scraping**: Extracts article content using BeautifulSoup
* **Semantic Text Processing**: 
  - Uses NLTK for intelligent sentence tokenization
  - Splits text into chunks while preserving semantic meaning
  - Ensures chunks don't exceed model's token limit
* **Summarization**: Uses DistilBART model to generate concise summaries
* **Progress Tracking**: Shows progress while processing multiple chunks

## 🧠 Model Used

* [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)

> You can easily switch to a different summarization model by changing the model ID in `main.py`.

## 📝 Example

```text
Enter a URL to summarize:
> https://example.com/article

Fetching article content...

Summarizing with semantic chunking...

→ Summarizing chunk 1/3 (850 tokens)
→ Summarizing chunk 2/3 (920 tokens)
→ Summarizing chunk 3/3 (780 tokens)

=== Summary ===
[Generated summary will appear here]
```

## 🧪 Testing Models

You can try other summarization models like:

* `facebook/bart-large-cnn`
* `google/pegasus-xsum`
* `microsoft/prophetnet-large-cnndm`

Just replace the model ID in `main.py`:

```python
model_id = "MODEL_ID_HERE"
```

## 📜 License

MIT License. Use responsibly.
