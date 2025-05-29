from transformers import pipeline
from tools import lookup_wikipedia, calculate_expression
import re

# Load the language model
llm = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    do_sample=False,
    max_new_tokens=150
)

# Define available tools
TOOLS = {
    "wikipedia": lookup_wikipedia,
    "calculator": calculate_expression
}

def plan_tasks(user_input):
    """
    Use LLM to break down the user task into executable steps.
    Output format: 'use <tool>: <input>'
    """
    prompt = (
        "You are an assistant that breaks user tasks into tool commands.\n"
        "Available tools: wikipedia, calculator\n"
        "Output format: one command per line:\n"
        "use <tool>: <input>\n"
        "You may use multiple tools and the same tool multiple times.\n"
        "Example (Do not write this example in your response):\n"
        "Input: Get the capital of France, find the population of France and calculate 7 * (2 + 3)\n"
        "Steps:\n"
        "use wikipedia: capital of France\n"
        "use wikipedia: population of France\n"
        "use calculator: 7 * (2 + 3)\n\n"
        f"Real user input: {user_input}\n"
        "Steps:\n"
        "<model_output>\n"
    )
    output = llm(prompt)[0]["generated_text"]
    lines = output.strip().split("\n")
    # Return only lines starting with "use "
    start_index = 0
    for line in lines:
        if line.strip().startswith("<model_output>"):
            start_index = lines.index(line)
    return [line.strip() for line in lines[start_index:] if line.strip().startswith("use ")]

def execute_plan(plan_lines):
    """
    Execute each line using the matching tool.
    """
    results = []
    for line in plan_lines:
        match = re.match(r"use (\w+): (.+)", line.strip())
        if not match:
            results.append(f"Could not parse step: {line}")
            continue
        tool_name, tool_input = match.groups()
        tool_func = TOOLS.get(tool_name)
        if not tool_func:
            results.append(f"Unknown tool: {tool_name}")
            continue
        try:
            result = tool_func(tool_input)
            results.append(f"[{tool_name}] {tool_input} → {result}")
        except Exception as e:
            results.append(f"[{tool_name}] error: {str(e)}")
    return results

if __name__ == "__main__":
    task = input("What task would you like the agent to perform (Wikipedia and/or Calculator)?\n> ")
    print("\nPlanning...\n")
    steps = plan_tasks(task)
    for step in steps:
        print("→", step)

    print("\nExecuting...\n")
    results = execute_plan(steps)
    for res in results:
        print(res)
