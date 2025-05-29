# Web Article Summarization Agent 📝🔍

This is an AI agent that uses a transformer model to automatically summarize web articles. It extracts content from any given URL and generates a concise summary using a pre-trained model.

## ✨ Features

- Extracts text content from web articles using BeautifulSoup
- Summarizes long articles by processing them in chunks
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
requests==2.31.0
beautifulsoup4==4.12.2
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

The agent will fetch the article content and generate a summary.

## 📂 Project Structure

```
.
├── main.py              # Main script that runs the summarization agent
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Features

* **Web Scraping**: Extracts article content using BeautifulSoup
* **Text Processing**: Handles long articles by splitting them into manageable chunks
* **Summarization**: Uses DistilBART model to generate concise summaries

## 🧠 Model Used

* [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)

> You can easily switch to a different summarization model by changing the model ID in `main.py`.

## 📝 Example

```text
Enter a URL to summarize:
> https://example.com/article

Fetching article content...

Summarizing...

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
model="MODEL_ID_HERE"
```

## 📜 License

MIT License. Use responsibly.
