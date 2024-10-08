import torch
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class RAFTCoTDataset(Dataset):
    def __init__(self, questions, contexts, answers, reasoning_steps, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        for question, context, answer, reasoning in zip(questions, contexts, answers, reasoning_steps):
            input_text = f"Question: {question} Context: {context}"
            target_text = f"Reasoning: {reasoning} Answer: {answer}"
            
            model_inputs = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True)
            with self.tokenizer.as_target_tokenizer():
                model_targets = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True)
            
            self.inputs.append({
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "labels": model_targets["input_ids"],
            })

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.inputs[idx].items()}

def train_raft_cot(model, train_dataloader, val_dataloader, device, num_epochs=5):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    return model

def generate_answer_cot(model, tokenizer, retriever, question, max_length=512):
    logging.info(f"Generating answer for question: {question}")
    retrieved_docs = retriever.retrieve(question)
    context = ' '.join(retrieved_docs)
    
    # Generate reasoning
    input_text = f"Question: {question} Context: {context} Generate reasoning:"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True).input_ids.to(model.device)

    logging.info("Generating reasoning...")
    reasoning_ids = model.generate(
        input_ids, 
        max_length=max_length // 2,  # Limit reasoning length
        num_return_sequences=1, 
        do_sample=True, 
        temperature=0.7,
        num_beams=3,
        early_stopping=True
    )
    reasoning = tokenizer.decode(reasoning_ids[0], skip_special_tokens=True)
    
    # Generate answer
    input_text = f"Question: {question} Context: {context} Reasoning: {reasoning} Generate answer:"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True).input_ids.to(model.device)

    logging.info("Generating answer...")
    answer_ids = model.generate(
        input_ids, 
        max_length=max_length // 4,  # Limit answer length
        num_return_sequences=1, 
        do_sample=True, 
        temperature=0.3,  # Lower temperature for more focused answers
        num_beams=3,
        early_stopping=True
    )
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    logging.info("Answer generation completed")
    logging.info(f"Generated reasoning: {reasoning}")
    logging.info(f"Generated answer: {answer}")
    
    return reasoning.strip(), answer.strip()

def main():
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

    train_reasoning_steps = [
        "The context states that Paris is the capital of France.",
        "The context mentions that Leonardo da Vinci painted the Mona Lisa.",
        "According to the context, water boils at 100 degrees Celsius at sea level.",
        "The context clearly states that the Eiffel Tower is located in Paris.",
        "The context directly states that Mount Everest is the tallest mountain in the world.",
        "The context mentions that the Mona Lisa is displayed in the Louvre Museum."
    ]

    test_questions = [
        "What's the tallest mountain in the world?",
        "In which museum can you find the Mona Lisa?"
    ]

    print("Initializing DocumentRetriever...")
    retriever = DocumentRetriever(documents)

    print("Loading model and tokenizer...")
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Preparing datasets...")
    train_dataset = RAFTCoTDataset(train_questions, train_contexts, train_answers, train_reasoning_steps, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataset = RAFTCoTDataset(train_questions[-2:], train_contexts[-2:], train_answers[-2:], train_reasoning_steps[-2:], tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Fine-tuning the model...")
    trained_model = train_raft_cot(model, train_dataloader, val_dataloader, device)

    print("\nTesting RAFT model with Chain-of-Thought...")
    for question in tqdm(test_questions, desc="Processing test questions"):
        try:
            reasoning, answer = generate_answer_cot(trained_model, tokenizer, retriever, question)
            print(f"\nQuestion: {question}")
            print(f"Reasoning: {reasoning}")
            print(f"Answer: {answer}\n")
        except Exception as e:
            logging.error(f"Error processing question '{question}': {str(e)}")

if __name__ == "__main__":
    main()