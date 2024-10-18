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

# Список участников для имитации стиля
participants = {
    "фейт имба": "Стиль фейта имбы",
    "Sussy Baka 1": "Стиль Sussy Baka 1",
    "Георгий": "Стиль Георгия",
    "К слову, работаю на Спидвагона": "Стиль Спидвагона",
    "СыроESHKA": "Стиль СыроESHKA",
    "Амогус 2": "Стиль Амогуса 2"
}

class ChatDataset(Dataset):
    def __init__(self, chats):
        self.examples = tokenizer(chats, return_tensors='pt', padding=True, truncation=True, max_length=512)
        self.labels = self.examples['input_ids'].clone()

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.examples['input_ids'][idx],
            'attention_mask': self.examples['attention_mask'][idx],
            'labels': self.labels[idx],
        }

def parse_chat(chat_data):
    parsed_messages = []
    current_participant = None
    message_buffer = []

    for line in chat_data.splitlines():
        if any(participant in line for participant in participants.keys()):
            if message_buffer:
                parsed_messages.append((current_participant, " ".join(message_buffer)))
                message_buffer = []
            current_participant = line.split(' ')[0]
        else:
            message_buffer.append(line)

    if message_buffer:
        parsed_messages.append((current_participant, " ".join(message_buffer)))

    return parsed_messages

def train_model(chat_data):
    parsed_data = parse_chat(chat_data)
    training_texts = [f"{participant}: {text}" for participant, text in parsed_data]

    dataset = ChatDataset("\n".join(training_texts))

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
    prompt = f"{participant_style}:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)
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

        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = f.read()

        train_model(chat_data)
        return "Обучение завершено", 200
    else:
        return "Неправильный формат файла", 400

@app.route('/generate', methods=['POST'])
def generate():
    selected_participant = request.form['participant']
    participant_style = participants.get(selected_participant, "Unknown")

    reply = generate_response(participant_style)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
