import torch
import warnings
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
transformers.logging.set_verbosity_error()


class DocumentRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_embeddings = self.model.encode(documents)

    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

class RAFTDataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        for question, context, answer in zip(questions, contexts, answers):
            tokenized = tokenizer(
                question,
                context,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = tokenized['input_ids'].squeeze()
            attention_mask = tokenized['attention_mask'].squeeze()
            
            answer_start = context.lower().index(answer.lower())
            answer_end = answer_start + len(answer)
            
            tokenized_answer = tokenizer(answer, add_special_tokens=False)
            start_positions = input_ids.tolist().index(tokenized_answer['input_ids'][0])
            end_positions = start_positions + len(tokenized_answer['input_ids']) - 1
            
            self.inputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'start_positions': torch.tensor(start_positions),
                'end_positions': torch.tensor(end_positions)
            })

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def train_raft(model, train_dataloader, val_dataloader, device, optimizer, scheduler, num_epochs=5):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                start_positions=start_positions, end_positions=end_positions)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    return model

def generate_answer(model, tokenizer, retriever, question, max_length=512):
    retrieved_docs = retriever.retrieve(question)
    context = ' '.join(retrieved_docs)
    
    inputs = tokenizer(question, context, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    if end_index < start_index:
        end_index = start_index

    answer_tokens = input_ids[0][start_index:end_index+1]
    answer = tokenizer.decode(answer_tokens)

    return answer.strip()

def main():
    # Expanded sample data
    documents = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "The Mona Lisa is displayed in the Louvre Museum.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Mount Everest is the tallest mountain in the world.",
        "Mount Everest is located in the Himalayas.",
        "The Louvre Museum is in Paris, France.",
        "The Mona Lisa is a famous painting by Leonardo da Vinci.",
        "The boiling point of water changes with altitude."
    ]

    train_questions = [
        "What is the capital of France?",
        "Who painted the Mona Lisa?",
        "At what temperature does water boil?",
        "Where is the Eiffel Tower located?",
        "What is the tallest mountain in the world?",
        "In which museum is the Mona Lisa displayed?"
    ]

    train_contexts = [
        "Paris is the capital of France. The Eiffel Tower is located in Paris.",
        "Leonardo da Vinci painted the Mona Lisa. The Mona Lisa is displayed in the Louvre Museum.",
        "Water boils at 100 degrees Celsius at sea level. The boiling point of water changes with altitude.",
        "The Eiffel Tower is located in Paris. Paris is the capital of France.",
        "Mount Everest is the tallest mountain in the world. Mount Everest is located in the Himalayas.",
        "The Mona Lisa is displayed in the Louvre Museum. The Louvre Museum is in Paris, France."
    ]

    train_answers = [
        "Paris",
        "Leonardo da Vinci",
        "100 degrees Celsius",
        "Paris",
        "Mount Everest",
        "Louvre Museum"
    ]

    val_questions = [
        "Where is Mount Everest located?",
        "Who is the artist of the Mona Lisa?",
    ]

    val_contexts = [
        "Mount Everest is the tallest mountain in the world. Mount Everest is located in the Himalayas.",
        "The Mona Lisa is a famous painting by Leonardo da Vinci. The Mona Lisa is displayed in the Louvre Museum."
    ]

    val_answers = [
        "Himalayas",
        "Leonardo da Vinci"
    ]

    test_questions = [
        "What's the tallest mountain in the world?",
        "In which museum can you find the Mona Lisa?"
    ]

    print("Initializing DocumentRetriever...")
    retriever = DocumentRetriever(documents)

    print("Loading BERT model and tokenizer...")
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', 
                                                          return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', 
                                              use_fast=True, 
                                              clean_up_tokenization_spaces=True)

    print("Preparing datasets...")
    train_dataset = RAFTDataset(train_questions, train_contexts, train_answers, tokenizer)
    val_dataset = RAFTDataset(val_questions, val_contexts, val_answers, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Training RAFT model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * 5  # 5 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    trained_model = train_raft(model, train_dataloader, val_dataloader, device, optimizer=optimizer, scheduler=scheduler)

    print("\nTesting RAFT model...")
    for question in test_questions:
        answer = generate_answer(trained_model, tokenizer, retriever, question)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()