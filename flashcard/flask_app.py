
import os
from flask import Flask, request, render_template, redirect, url_for
from pdfminer.high_level import extract_text
from pptxtopdf import convert
from transformers import pipeline
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Initialize the summarization and question generation pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")



# Initialize Flask app
app = Flask(__name__)


def ppt_to_pdf(input_file, output_folder):
    try:
        # Convert PPT/PPTX to PDF
        convert(input_file, output_folder)
        output_file = os.path.join(output_folder, os.path.basename(input_file).replace('.pptx', '.pdf').replace('.ppt', '.pdf'))
        text = extract_text(output_file)
        return text
    except Exception as e:
        raise e

def extract_text_mn(pdf_path):
    text = extract_text(pdf_path)
    return text

def abstractive_summarization(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Splitting the text into chunks that fit the model's max token limit
    max_chunk_size = 1024  # BART's maximum context size is 1024 tokens
    sentences = sent_tokenize(text)
    current_chunk = ""
    chunks = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    summary_text = ""
    for chunk in chunks:
        summary_text += summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] + " "
    return summary_text.strip()

def generate_flashcards(text):
    sentences = text.split('. ')
    flashcards = []
    for sentence in sentences:
        if sentence.strip():
            generated_questions = question_generator("generate question: " + sentence)
            question = generated_questions[0]['generated_text']
            flashcards.append({'question': question, 'answer': sentence})
    return flashcards

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        if filename.endswith('.ppt') or filename.endswith('.pptx'):
            output_folder = os.path.splitext(filepath)[0]
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            text = ppt_to_pdf(filepath, output_folder)
        elif filename.endswith('.pdf'):
            text = extract_text(filepath)
        else:
            return "Unsupported file type", 400

        # Generate summarized text
        summarized_text = abstractive_summarization(text)
        print(summarized_text)

        # Generate questions using T5 model
        generated_question = question_generator("generate question: " + summarized_text)
        question = generated_question[0]['generated_text']

        flashcards = generate_flashcards(summarized_text)
        return render_template('flashcards.html', flashcards=flashcards, generated_question=question)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
