import wikipedia

def lookup_wikipedia(query: str, sentences: int = 2) -> str:
    """
    Return the first few sentences from Wikipedia for the given query.
    """
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_expression(expr: str) -> str:
    """
    Very basic calculator (do not use eval in production!).
    """
    try:
        result = eval(expr, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Invalid expression: {str(e)}"
