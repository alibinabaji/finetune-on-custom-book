from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the fine-tuned model and tokenizer
model_name = "finetuned_model_book1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/ask', methods=['POST'])
def ask_question():
    content = request.json
    book = content['book']
    question = content['question']

    # Process question using the model
    answer = generator(question, max_length=50, num_return_sequences=1)[0]['generated_text']

    # Set CORS headers for the response to allow all origins
    response = jsonify({"book": book, "question": question, "answer": answer})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True)
