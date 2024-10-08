# <div align="left">RAFT Implementation: Sample RAFT Implementation using BERT, CoT, Few-Shot<div>

<div align="left">
<p><strong>An example and educational implementation of RAFT using BERT, Chain of Thought and Few-Shot!</strong></p>
<a href="https://arxiv.org/abs/2403.10131" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
<a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
<a href="https://linkedin.com/in/jit2600"><img src="https://img.shields.io/badge/LinkedIn-Connect-blue" alt="LinkedIn"></a>
<a href="https://twitter.com/stretchcloud"><img src="https://img.shields.io/twitter/follow/stretchcloud?label=Follow%20@stretchcloud&style=social" alt="Twitter"></a>
<a href="https://github.com/stretchcloud/RAFT/stargazers"><img src="https://img.shields.io/github/stars/stretchcloud/RAFT?style=social" alt="GitHub Stars"></a>
<a href="https://github.com/stretchcloud/RAFT/network/members"><img src="https://img.shields.io/github/forks/stretchcloud/RAFT?style=social" alt="GitHub Forks"></a>
<a href="https://github.com/stretchcloud/RAFT/issues"><img src="https://img.shields.io/github/issues/stretchcloud/RAFT" alt="GitHub Issues"></a>
<a href="https://github.com/stretchcloud/RAFT/pulls"><img src="https://img.shields.io/github/issues-pr/stretchcloud/RAFT" alt="GitHub Pull Requests"></a>
</div>


This repository contains four different implementations of the Retrieval-Augmented Fine-Tuning (RAFT) technique for question answering:

1. RAFT using BERT
2. RAFT with Few-Shot Learning
3. RAFT using FLAN-T5 model and Chain-of-Thought (CoT) Reasoning
4. RAFT with BERT and Chain-of-Thought (CoT) Reasoning

## 1. RAFT using BERT

![RAFT Flow using BERT](RAFT-BERT.png)

### Summary
This implementation uses BERT (Bidirectional Encoder Representations from Transformers) for the RAFT process. It fine-tunes a BERT model on a dataset of question-answer pairs, using retrieved documents as context.

### Steps
1. Initialize the DocumentRetriever with a set of documents
2. Prepare the dataset using RAFTDataset
3. Fine-tune the BERT model using the prepared dataset
4. Use the fine-tuned model to generate answers for new questions

### Key Components
- DocumentRetriever: Uses SentenceTransformer to encode documents and retrieve relevant ones
- RAFTDataset: Prepares the data for BERT fine-tuning
- train_raft: Fine-tunes the BERT model
- generate_answer: Uses the fine-tuned model to generate answers

### Usage
Run `python raft-bert.py` to execute this implementation.

## 2. RAFT with Few-Shot Learning

### Summary
This implementation uses few-shot learning with GPT-3.5-turbo to perform RAFT. It retrieves relevant documents and uses a few examples to guide the model in generating answers.

### Steps
1. Initialize the DocumentRetriever with a set of documents
2. Generate few-shot examples using training data
3. For each test question:
   a. Retrieve relevant documents
   b. Generate an answer using few-shot examples and retrieved context

### Key Components
- DocumentRetriever: Uses SentenceTransformer to encode documents and retrieve relevant ones
- generate_raft_examples: Creates few-shot examples for the model
- generate_raft_answer: Uses GPT-3.5-turbo to generate answers based on few-shot examples and context

### Usage
Run `python raft-few-shot.py` to execute this implementation.

## 3. RAFT with Chain-of-Thought (CoT) Reasoning

![Document Retrieval and Processing Pipeline](RAFT-CoT.png)

### Summary
This implementation combines RAFT with Chain-of-Thought (CoT) reasoning using GPT-4. It retrieves relevant documents and prompts the model to provide step-by-step explanations for its answers.

### Steps
1. Initialize the DocumentRetriever with a set of documents
2. For each test question:
   a. Retrieve relevant documents
   b. Generate a CoT answer using GPT-4 with the retrieved context

### Key Components
- DocumentRetriever: Uses SentenceTransformer to encode documents and retrieve relevant ones
- generate_cot_answer: Uses GPT-4 to generate step-by-step explanations and answers

### Usage
Run `python raft-cot.py` to execute this implementation.

## 4. RAFT with FLAN-T5 model and Chain-of-Thought (CoT) Reasoning

![RAFT with BERT and Chain-of-Thought Technical Architecture](raft-cot-architecture.jpeg)

### Summary
This implementation combines RAFT using BERT with Chain-of-Thought (CoT) reasoning. It fine-tunes a FLAN-T5 model on a dataset of question-answer pairs with reasoning steps, using retrieved documents as context.

### Steps
1. Initialize the DocumentRetriever with a set of documents
2. Prepare the dataset using RAFTCoTDataset, including reasoning steps
3. Fine-tune the FLAN-T5 model using the prepared dataset
4. Use the fine-tuned model to generate both reasoning and answers for new questions

### Key Components
- DocumentRetriever: Uses SentenceTransformer to encode documents and retrieve relevant ones
- RAFTCoTDataset: Prepares the data for FLAN-T5 fine-tuning, including reasoning steps
- train_raft_cot: Fine-tunes the FLAN-T5 model
- generate_answer_cot: Uses the fine-tuned model to generate reasoning and answers in two steps

### Architecture
The architecture diagram illustrates the following components:
- Document Retriever: Uses Sentence-BERT for embedding and cosine similarity for retrieval
- FLAN-T5 Model: The core of the system, fine-tuned for QA and CoT reasoning
- Training Data: Includes questions, contexts, answers, and reasoning steps
- Fine-tuning Process: Uses AdamW optimizer and linear scheduler
- Generation Process: Employs beam search and temperature sampling, implementing a two-step CoT approach

### Usage
Run `python raft-bert-cot.py` to execute this implementation.

## Comparison of Techniques

1. RAFT using BERT:
   - Pros: Fine-tuned on specific data, can be faster at inference time
   - Cons: Requires more data and computational resources for training

2. RAFT with Few-Shot Learning:
   - Pros: Requires less training data, more flexible for different types of questions
   - Cons: Relies on the quality of few-shot examples, may be less consistent

3. RAFT using FLAN-T5 model and Chain-of-Thought (CoT) Reasoning:
   - Pros: Provides detailed explanations, potentially more accurate for complex questions
   - Cons: May be slower due to generating longer responses, requires more powerful language model (GPT-4)

4. RAFT with BERT and Chain-of-Thought (CoT) Reasoning:
   - Pros: Combines the benefits of fine-tuning and CoT reasoning, potentially more accurate and faster than GPT-4 based approach
   - Cons: Requires more complex training process, may need larger dataset for effective fine-tuning

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- SentenceTransformer
- OpenAI API key (for Few-Shot and GPT-4 CoT implementations)

## Setup

1. Install the required packages: `pip install torch transformers sentence-transformers openai`
2. Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY=your_api_key_here`
3. Run the desired implementation as described in the Usage sections above.

## Note

These implementations are for educational purposes and may require further optimization for production use. Always consider the ethical implications and potential biases when using language models for question answering tasks.