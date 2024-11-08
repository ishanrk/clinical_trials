import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")  # Extract text from each page
    return text

pdf_text = extract_text_from_pdf("psych_trials.pdf")

import spacy

# Load a pre-trained NLP model
import spacy
from transformers import BertTokenizer, BertModel
import torch

# Load SpaCy and BERT
nlp = spacy.load("en_core_sci_sm")  # SpaCy model for scientific text
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Define functions for embedding extraction and entity relationship identification
def bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def build_knowledge_graph(text):
    entities = extract_entities(text)
    graph_data = {}
    for ent in entities:
        entity_embedding = bert_embedding(ent[0])  # Get embedding for each entity
        graph_data[ent[0]] = {"type": ent[1], "embedding": entity_embedding}
    return graph_data

graph_data = build_knowledge_graph(pdf_text)



## RETRIEVAL STEP

import openai
import pinecone

# Initialize Pinecone for storing BERT embeddings
pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")
index = pinecone.Index("ckg-psychiatry")

# Store entities in Pinecone
for entity, data in graph_data.items():
    index.upsert([(entity, data["embedding"])])

# Function to retrieve context for a new trial
def retrieve_relevant_context(trial_text, num_retrievals=5):
    trial_embedding = bert_embedding(trial_text)
    result = index.query(trial_embedding, top_k=num_retrievals, include_metadata=True)
    context = "\n".join([f"{match['id']}: {match['metadata']}" for match in result['matches']])
    return context

# Query ChatGPT with relevant context
def query_chatgpt_with_context(trial_text):
    context = retrieve_relevant_context(trial_text)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in psychiatric clinical trials."},
            {"role": "user", "content": f"Here’s relevant info from similar trials:\n{context}"},
            {"role": "user", "content": f"Now analyze this new trial:\n{trial_text}"}
        ]
    )
    return response.choices[0].message["content"]

# Test with a new trial
new_trial_text = "A study evaluates the impact of cognitive behavioral therapy on anxiety in adolescents."
print(query_chatgpt_with_context(new_trial_text))
