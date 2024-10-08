import os
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# Set your OpenAI API key
openai.api_key = "Replace with your actual API key"

# Set environment variable to avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def generate_raft_examples(retriever, questions, answers, n_examples=3):
    examples = []
    for q, a in zip(questions, answers):
        retrieved_docs = retriever.retrieve(q)
        context = ' '.join(retrieved_docs)
        example = f"Question: {q}\nContext: {context}\nAnswer: {a}\n\n"
        examples.append(example)
    return ''.join(random.sample(examples, min(n_examples, len(examples))))

def generate_raft_answer(question, context, few_shot_examples):
    prompt = f"""You are an AI assistant trained to answer questions based on given context. 
Use the following examples to understand the format, then answer the new question.

{few_shot_examples}
Question: {question}
Context: {context}
Answer: """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on given context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].message['content'].strip()

def main():
    # Training data
    train_questions = [
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the largest planet in our solar system?",
    ]
    train_answers = [
        "The capital of France is Paris.",
        "William Shakespeare wrote 'Romeo and Juliet'.",
        "Jupiter is the largest planet in our solar system.",
    ]
    
    # Test questions
    test_questions = [
        "What's in the Louvre Museum?",
        "Who is credited with inventing the telephone?",
        "What is the boiling temperature of water?",
    ]

    documents = [
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris.",
        "William Shakespeare wrote many famous plays including 'Romeo and Juliet'.",
        "Romeo and Juliet is a tragedy written by Shakespeare.",
        "Jupiter is the largest planet in our solar system.",
        "Mars is often called the Red Planet.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "The Mona Lisa is displayed in the Louvre Museum in Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Alexander Graham Bell is credited with inventing the telephone.",
        "The first telephone call was made on March 10, 1876.",
        "The boiling point of water can change with altitude.",
    ]

    print("Initializing DocumentRetriever...")
    retriever = DocumentRetriever(documents)

    print("Generating few-shot examples...")
    few_shot_examples = generate_raft_examples(retriever, train_questions, train_answers)

    print("Answering test questions...")
    for test_question in test_questions:
        print(f"\nQuestion: {test_question}")
        retrieved_docs = retriever.retrieve(test_question)
        context = ' '.join(retrieved_docs)
        print("Retrieved context:", context)
        raft_answer = generate_raft_answer(test_question, context, few_shot_examples)
        print("RAFT Answer:")
        print(raft_answer)

if __name__ == "__main__":
    main()