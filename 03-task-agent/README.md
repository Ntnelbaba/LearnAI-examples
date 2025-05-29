# Task Breakdown Agent 🧠🔧

This is a simple AI agent that uses an LLM to break down a natural language task into specific commands, and then executes them using tools like Wikipedia search and a calculator.

## ✨ Features

- Converts user instructions into structured tool commands
- Executes commands using:
  - `Wikipedia` lookup for factual information
  - `Calculator` for mathematical expressions
- Uses a lightweight open-source model (`SmolLM2-1.7B-Instruct`) for local inference
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
accelerate==0.29.3
torch>=2.1.0
wikipedia==1.4.0
```

## 🚀 How to Run

Run the agent from the terminal:

```bash
python main.py
```

Then input a task such as:

```text
Get the capital of Brazil, the population of Canada, and calculate 7 * (8 + 2)
```

The model will respond with tool commands, and then the agent will execute them and print the results.

## 📂 Project Structure

```
.
├── main.py              # Main script that runs the agent
├── tools.py             # Tool implementations for Wikipedia & calculator
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Tools Available

* **wikipedia**: Searches Wikipedia for facts using the `wikipedia` Python package
* **calculator**: Evaluates basic math expressions using Python `eval()` (Note: This is a simplified implementation for demonstration purposes)

## 🧠 Model Used

* [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

> You can easily switch to a different `text-generation` model by changing the model ID in `main.py`.

## 📝 Example

```text
Input:
Get the capital of Germany and calculate 7 * (4 + 3)

Output:
→ use wikipedia: capital of Germany
→ use calculator: 7 * (4 + 3)

Executing...
[wikipedia] capital of Germany → The capital of Germany is Berlin.
[calculator] 7 * (4 + 3) → 49
```

## 🧪 Testing Models

You can try other models like:

* `microsoft/Phi-3-mini-4k-instruct`
* `google/gemma-2b-it`
* `tiiuae/falcon-rw-1b`

Just replace the model ID in `main.py`:

```python
model="MODEL_ID_HERE"
```

## 📜 License

MIT License. Use responsibly.
