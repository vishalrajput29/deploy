import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage
import re
## Code
#######
def extract_tasks(text):
    """Heuristic-based function to extract potential tasks from unstructured text."""
    sentences = text.split('.')
    tasks = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if any(keyword in sentence.lower() for keyword in [' has to ', ' should ', ' must ', ' needs to ']):
            task = {'task': sentence, 'entity': None, 'deadline': None}
            
            # Extract entity (who has to do it)
            words = sentence.split()
            if words:
                task['entity'] = words[0]  # Assuming first word is the subject
            
            # Extract deadline if available
            deadline_match = re.search(r'by\s([\w\s]+)', sentence, re.IGNORECASE)
            if deadline_match:
                task['deadline'] = deadline_match.group(1)
            
            tasks.append(task)
    
    return tasks

def categorize_tasks_with_llama(tasks):
    """Uses LLaMA 3.2 (1B) to categorize extracted tasks."""
    model = ChatOllama(model="llama3.2:1b")
    categorized_tasks = []
    
    for task in tasks:
        prompt = f"Classify the task: '{task['task']}' into categories like 'Personal', 'Work', 'Errand', etc."
        response = model([HumanMessage(content=prompt)])
        category = response.content.strip()
        task['category'] = category
        categorized_tasks.append(task)
    
    return categorized_tasks

def main():
    st.title("Task Extraction and Categorization")
    st.write("Enter a paragraph, and the app will extract and categorize tasks from it.")
    
    text_input = st.text_area("Enter your text here:")
    
    if st.button("Extract Tasks"):
        if text_input:
            tasks = extract_tasks(text_input)
            categorized_tasks = categorize_tasks_with_llama(tasks)
            
            st.subheader("Extracted Tasks")
            for task in categorized_tasks:
                st.write(f"**Task:** {task['task']}")
                st.write(f"**Entity:** {task['entity']}")
                st.write(f"**Deadline:** {task['deadline']}")
                st.write(f"**Category:** {task['category']}")
                st.write("---")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
