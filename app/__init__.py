from flask import Flask

app = Flask(__name__)

# Импортируем маршруты
from .routes import routes
app.register_blueprint(routes)

# Настройки приложения
app.config['SECRET_KEY'] = app.secret_key = b'\x00Z\x0e\xe6vmB\xe4Z\xd9\xd5\xff\xd9\x96{6\x99\xb3\x94l\x8d\xfe\x9e\xd5'
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
