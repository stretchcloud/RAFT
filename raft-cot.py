import os
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Set environment variable to avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set your OpenAI API key
openai.api_key = "Replace with your actual API key"

class DocumentRetriever:
    def __init__(self, documents):
        self.documents = documents
        print("Initializing SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Encoding documents...")
        self.document_embeddings = self.model.encode(documents, show_progress_bar=True)

    def retrieve(self, query, k=2):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

def generate_cot_answer(question, context):
    prompt = f"""Answer the following question using the given context. Provide a step-by-step explanation.

Question: {question}

Context: {context}

Step-by-step answer:
1) First, let's identify the key information in the question:
2) Now, let's look at the relevant information provided in the context:
3) Let's reason about this information:
4) Finally, we can formulate our answer:

Detailed explanation:
"""

    print("Generating answer using OpenAI API...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides step-by-step explanations to questions based on given context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].message['content'].strip()

def main():
    questions = [
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What is the boiling point of water?",
        "Who invented the telephone?",
        "What is the chemical symbol for gold?",
        "What is the tallest mountain in the world?"
    ]
    documents = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is in Paris.",
        "William Shakespeare wrote many famous plays including 'Romeo and Juliet'.",
        "Romeo and Juliet is a tragedy written by Shakespeare.",
        "Jupiter is the largest planet in our solar system.",
        "Mars is often called the Red Planet.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "The Mona Lisa is displayed in the Louvre Museum in Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Alexander Graham Bell is credited with inventing the telephone.",
        "The chemical symbol for gold is Au.",
        "Mount Everest is the tallest mountain in the world.",
        "The boiling point of water can change with altitude.",
        "The first telephone call was made on March 10, 1876.",
        "Gold is a precious metal used in jewelry and electronics.",
        "Mount Everest is located in the Himalayas, between Nepal and Tibet."
    ]

    print("Initializing DocumentRetriever...")
    retriever = DocumentRetriever(documents)

    test_questions = [
        "What's in the Louvre Museum?",
        "Who is credited with inventing the telephone?",
        "What is the boiling temperature of water?"
    ]

    for test_question in test_questions:
        print(f"\nQuestion: {test_question}")
        retrieved_docs = retriever.retrieve(test_question)
        context = ' '.join(retrieved_docs)
        print("Retrieved context:", context)
        cot_answer = generate_cot_answer(test_question, context)
        print("Chain-of-Thought Answer:")
        print(cot_answer)

if __name__ == "__main__":
    main()