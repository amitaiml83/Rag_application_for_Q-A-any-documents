from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI
import streamlit as st

# reading the pdf file using langchain.
def read_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load()
    return docs

# split the text into chunks for the pdf.
def text_splitter_chunk(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # For ChromaDB
    docx_dict = {}
    content = []
    metadata = []
    idx = []
    i = 0
    for chunk in documents:
        content.append(str(chunk.page_content))
        metadata.append(chunk.metadata)
        i += 1
        idx.append(str(i))
    docx_dict["content"] = content
    docx_dict["metadata"] = metadata
    docx_dict["idx"] = idx

    return docx_dict

# Embedding the chunk using ChromaDB and default embedding function
def doc_embeding(content, metadata, idx):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="amit_dk")

    collection.add(
        documents=content,
        metadatas=metadata,
        ids=idx
    )
    return collection

# Semantic search from ChromaDB vector
def retrieve_useful_chunk(query, collection):
    results = collection.query(
        query_texts=query,
        n_results=5
    )

    context = " ".join(doc for doc in results["documents"][0])
    return context

def generate_questions_answers(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please answer the question based on the prompt."},
            {"role": "user", "content": prompt}
        ]
    )

    response_text = response.choices[0].message.content
    return response_text

# Initialize the OpenAI client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Streamlit app interface
st.title("Local LLM Model for Question Answering with PDF Upload")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    docs = read_pdf(uploaded_file)  # Calling the function to read the uploaded PDF
    z = text_splitter_chunk(docs)   # Splitting the document into chunks
    collection = doc_embeding(z['content'], z['metadata'], z['idx'])  # Embedding the chunks

    query = st.text_input(label="Please enter the query based on the uploaded PDF")

    if query:
        content = retrieve_useful_chunk(query, collection)
        response = generate_questions_answers(query, content)
        st.write(response)
