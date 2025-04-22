from langchain_community.vectorstores import FAISS
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion, load_repo, text_splitter
from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize components with error handling
def initialize_components():
    print("Loading documents...")
    documents = load_repo("repo/")
    if not documents:
        raise ValueError("No documents loaded from repository")
    
    print("Splitting text...")
    text_chunks = text_splitter(documents)
    if not text_chunks:
        raise ValueError("No text chunks created")
    
    print("Loading embeddings...")
    embeddings = load_embedding()
    # Test the embedding
    test_embed = embeddings.embed_query("test")
    if not test_embed:
        raise ValueError("Embedding failed - check API key")
    
    print("Creating vector store...")
    try:
        vectordb = FAISS.from_documents(
            documents=text_chunks,
            embedding=embeddings
        )
        return vectordb
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")
        print(f"Number of chunks: {len(text_chunks)}")
        if text_chunks:
            print(f"First chunk content: {text_chunks[0].page_content}")
        raise

try:
    vectordb = initialize_components()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.4,    
        max_output_tokens=500
    )
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), 
        memory=memory
    )
except Exception as e:
    print(f"Initialization failed: {str(e)}")
    qa = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=["POST"])
def gitRepo():
    if request.method == 'POST':
        user_input = request.form['question']
        try:
            repo_ingestion(user_input)
            # Use subprocess instead of os.system for better control
            from subprocess import run
            run(["python", "store_index.py"], check=True)
            return jsonify({"response": "Repository processed successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/get", methods=["POST"])
def chat():
    if not qa:
        return "System not initialized properly", 500
        
    msg = request.form["msg"]
    if msg == "clear":
        import shutil
        shutil.rmtree("repo", ignore_errors=True)
        return "Repository cleared"
    
    try:
        result = qa(msg)
        return str(result["answer"])
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)