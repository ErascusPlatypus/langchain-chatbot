import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
api = Api(app)

def extract_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

url = "https://brainlox.com/courses/category/technical"
raw_text = extract_data(url)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(texts, embeddings)

model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,  
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

app = Flask(__name__)
api = Api(app)

class ChatbotConversation(Resource):
    def post(self):
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        result = qa_chain.invoke(query)
        
        answer = result['result']
        formatted_answer = '\n'.join(line.strip() for line in answer.split('\n')[1:])
        
        response = {
            "answer": formatted_answer
        }
        
        return jsonify(response)

api.add_resource(ChatbotConversation, '/chat')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)