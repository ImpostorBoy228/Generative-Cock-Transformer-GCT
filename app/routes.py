import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from threading import Thread
from . import app

routes = Blueprint('routes', __name__)

# Модель и токенизатор
MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Прогресс
progress_status = {"progress": 0}

# Проверка допустимости файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt'}

# Главная страница
@routes.route('/')
def index():
    return render_template('index.html', chat_history=[])

# Страница загрузки модели
@routes.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Файл не выбран.')
            return redirect(request.url)
        if allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            flash('Файл успешно загружен. Обучение начинается...')
            return redirect(url_for('routes.train_model', filepath=filepath))
        flash('Разрешены только текстовые файлы (.txt).')
        return redirect(request.url)
    return render_template('upload_model.html')

# Обучение модели
@routes.route('/train_model', methods=['GET'])
def train_model():
    filepath = request.args.get('filepath')
    if filepath:
        Thread(target=start_training, args=(filepath,)).start()
        flash('Обучение запущено. Следите за прогрессом.')
    return render_template('upload_model.html', message='Обучение модели...')

# Старт обучения в отдельном потоке
def start_training(filepath):
    global progress_status
    progress_status["progress"] = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        train_data = f.read()

    inputs = [f"summarize: {line}" for line in train_data.split('\n') if line.strip()]
    encodings = tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt", max_length=512)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encodings,
    )

    for i in range(10):  # Пример симуляции прогресса
        time.sleep(1)
        progress_status["progress"] = (i + 1) * 10

    trainer.train()
    progress_status["progress"] = 100

# Чат
@routes.route('/chat', methods=['POST'])
def chat():
    global progress_status
    progress_status["progress"] = 0
    user_message = request.form['message']

    input_ids = tokenizer(f"question: {user_message}", return_tensors="pt").input_ids
    progress_status["progress"] = 50

    outputs = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    progress_status["progress"] = 100
    return render_template('index.html', chat_history=[{"role": "user", "content": user_message}, {"role": "assistant", "content": response}])

# Получение статуса прогресса
@routes.route('/progress', methods=['GET'])
def progress():
    return jsonify(progress_status)

# Очистка истории
@routes.route('/clear-history', methods=['POST'])
def clear_history():
    return redirect(url_for('routes.index'))
