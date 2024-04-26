from flask import Flask, request
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

cached_llm = Ollama(model="llama3")

db_folder = "db"

embedding = GPT4AllEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template("""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Keep the answer concise.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved context: \n\n {context} \n\n                
    Here is the user question: {input} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
)

@app.route("/ai", methods=["POST"])
def aiPost():
    print("POST /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"Query: {query}")

    response = cached_llm.invoke(query)

    response_answer = {"answer": response}
    return response_answer

@app.route("/pdf", methods=["POST"])
def pdfPost():
    print("POST /pdf called")
    file = request.files["file"]
    filename = file.filename
    filepath = "pdf/" + filename
    file.save(filepath)
    print(f"filename: {filename}")

    loader = PDFPlumberLoader(filepath)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=db_folder)
    vector_store.persist()

    response = {
        "status": "Successfully Uploaded", 
        "filename": filename, 
        "docs_len": len(docs),
        "chunks": len(chunks)
    }
    return response

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("POST /ask_pdf called")
    
    json_content = request.json
    query = json_content.get("query")
    print(f"Query: {query}")


    print("Loading vector store")
    vector_store = Chroma(persist_directory=db_folder, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.2,
        }
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append({
            "source": doc.metadata["source"],
            "page_content": doc.page_content,
        })

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()