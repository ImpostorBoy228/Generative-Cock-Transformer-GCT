from flask import Flask, request, jsonify, render_template
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Инициализация модели и токенизатора
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Установка токена паддинга
tokenizer.pad_token = tokenizer.eos_token

# Проверьте, доступен ли GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Переместите модель на GPU

# Предварительно определённый список участников
participants = {
    'фейт имба': "Стиль фейта имбы",
    'Sussy Baka 1': "Стиль Sussy Baka 1",
    'Георгий': "Стиль Георгия",
    'К слову, работаю на Спидвагона': "Стиль Спидвагона",
    'СыроESHKA': "Стиль СыроESHKA",
    'Амогус 2': "Стиль Амогуса 2"
}

class ChatDataset(Dataset):
    def __init__(self, chats):
        # Токенизируем текст и создаем метки для обучения
        self.examples = tokenizer(chats, return_tensors='pt', padding=True, truncation=True, max_length=512)
        self.labels = self.examples['input_ids'].clone()  # Создаем метки равные input_ids

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.examples['input_ids'][idx].to(device),  # Перемещаем на одно устройство
            'attention_mask': self.examples['attention_mask'][idx].to(device),  # Перемещаем на одно устройство
            'labels': self.labels[idx].to(device),  # Перемещаем на одно устройство
        }

def parse_chat(chat_data):
    # Объединяем сообщения каждого участника в один текст для обучения
    # Используем только заранее определённый список участников
    chat_lines = chat_data.splitlines()
    chat_messages = {participant: [] for participant in participants.keys()}

    for line in chat_lines:
        for participant in participants.keys():
            if line.startswith(participant):
                message = line[len(participant):].strip()
                chat_messages[participant].append(message)

    return {k: "\n".join(v) for k, v in chat_messages.items()}

def train_model(chat_data):
    chat_messages = parse_chat(chat_data)

    # Объединяем все сообщения в один текст для обучения
    all_chats = "\n".join(chat_messages.values())
    dataset = ChatDataset(all_chats)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_steps=100,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

def generate_response(participant_style):
    prompt = f"{participant_style}:"  # Начнем с стиля участника
    inputs = tokenizer(prompt, return_tensors='pt').to(device)  # Перемещаем на одно устройство

    # Генерация текста с улучшенными параметрами
    outputs = model.generate(
        inputs['input_ids'],
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,  # Более низкое значение для большей предсказуемости
        top_k=50,         # Ограничение на количество токенов
        top_p=0.95,       # Выбор из верхних токенов
        do_sample=True    # Включаем случайность
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


@app.route('/')
def index():
    return render_template('index.html', participants=participants)

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return "Нет файла для загрузки", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "Нет выбранного файла", 400
    
    if file and file.filename.endswith('.txt'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Обработка файла для обучения модели
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = f.read()

        # Тренировка модели на основе загруженных данных
        train_model(chat_data)

        return "Обучение завершено", 200
    else:
        return "Неправильный формат файла", 400

@app.route('/generate', methods=['POST'])
def generate():
    selected_participant = request.form['participant']
    participant_style = participants[selected_participant]  # Получаем стиль участника
    
    # Генерация сообщения в стиле выбранного участника
    reply = generate_response(participant_style)

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
