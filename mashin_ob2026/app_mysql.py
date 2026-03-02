from flask import Flask, request, jsonify, render_template_string
import pymysql
from pymysql import Error
import torch
import torch.nn as nn
import pickle
import re


DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '17421742',  
    'database': 'toxic_comments_db',
    'charset': 'utf8mb4'
}

MODEL_PATH = 'toxic_lstm_model.pth'
VOCAB_PATH = 'vocab.pkl'
MAX_LEN = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Загрузка модели и словаря ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return self.sigmoid(output)

print("[INFO] Loading model and vocabulary...")
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

model = LSTMClassifier(len(vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"[INFO] Model loaded successfully! Vocabulary size: {len(vocab)}")


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^а-яa-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_text(text, vocab, max_len):
    tokens = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor([tokens], dtype=torch.long).to(DEVICE)

def predict(text):
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0, 0
    encoded = encode_text(cleaned, vocab, MAX_LEN)
    with torch.no_grad():
        pred = model(encoded).cpu().item()
    return pred, int(pred > 0.5)


def get_db_connection():
    """Получение подключения к MySQL"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def save_comment_to_db(text, prediction, is_toxic):
    """Сохранение комментария в MySQL"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        query = "INSERT INTO comments (text, prediction, is_toxic) VALUES (%s, %s, %s)"
        cursor.execute(query, (text[:500], prediction, is_toxic))
        conn.commit()
        return True
    except Error as e:
        print(f"Error saving to MySQL: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_recent_comments(limit=20):
    """Получение последних комментариев из MySQL"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        query = """
            SELECT text, prediction, is_toxic, timestamp 
            FROM comments 
            ORDER BY timestamp DESC 
            LIMIT %s
        """
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        return results
    except Error as e:
        print(f"Error fetching from MySQL: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def get_stats():
    """Получение статистики"""
    conn = get_db_connection()
    if not conn:
        return {'total': 0, 'toxic': 0, 'non_toxic': 0}
    
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        cursor.execute("SELECT COUNT(*) as total FROM comments")
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as toxic FROM comments WHERE is_toxic = 1")
        toxic = cursor.fetchone()['toxic']
        
        non_toxic = total - toxic
        
        return {
            'total': total,
            'toxic': toxic,
            'non_toxic': non_toxic
        }
    except Error as e:
        print(f"Error getting stats: {e}")
        return {'total': 0, 'toxic': 0, 'non_toxic': 0}
    finally:
        cursor.close()
        conn.close()


app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Toxic Comment Detector</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 1000px; 
            margin: auto; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        h1, h2 { color: #333; }
        .container { 
            background: white; 
            padding: 25px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        textarea { 
            width: 100%; 
            padding: 12px; 
            margin: 10px 0; 
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #007bff;
        }
        button { 
            padding: 12px 30px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 5px;
            cursor: pointer; 
            font-size: 16px;
            font-weight: bold;
            transition: background 0.3s;
        }
        button:hover {
            background: #0056b3;
        }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
        }
        .toxic { 
            background-color: #ffebee; 
            color: #c62828; 
            border-left: 5px solid #c62828;
        }
        .non-toxic { 
            background-color: #e8f5e9; 
            color: #2e7d32; 
            border-left: 5px solid #2e7d32;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px; 
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }
        th { 
            background-color: #f8f9fa; 
            font-weight: bold;
            color: #333;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .badge-toxic {
            background-color: #c62828;
        }
        .badge-non-toxic {
            background-color: #2e7d32;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            flex: 1;
            text-align: center;
        }
        .stat-number {
            font-size: 32px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>🔍 Toxic Comment Classifier</h1>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number" id="totalCount">0</div>
            <div class="stat-label">Всего комментариев</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="toxicCount">0</div>
            <div class="stat-label">Токсичных</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="nonToxicCount">0</div>
            <div class="stat-label">Нетоксичных</div>
        </div>
    </div>

    <div class="container">
        <h2>📝 Проверить новый комментарий</h2>
        <form id="commentForm">
            <textarea id="comment" rows="4" placeholder="Введите комментарий для проверки..."></textarea><br>
            <button type="button" onclick="submitComment()">Проверить</button>
        </form>
        <div id="result" class="result-box" style="display: none;"></div>
    </div>

    <h2>📊 История проверок</h2>
    <table id="history">
        <thead>
            <tr>
                <th>Комментарий</th>
                <th>Вероятность</th>
                <th>Статус</th>
                <th>Время</th>
            </tr>
        </thead>
        <tbody></tbody>
    </table>

    <script>
        function loadStats() {
            fetch('/api/stats')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('totalCount').textContent = data.total;
                    document.getElementById('toxicCount').textContent = data.toxic;
                    document.getElementById('nonToxicCount').textContent = data.non_toxic;
                });
        }

        function loadHistory() {
            fetch('/api/history')
                .then(res => res.json())
                .then(data => {
                    const tbody = document.querySelector('#history tbody');
                    tbody.innerHTML = '';
                    data.forEach(row => {
                        const tr = document.createElement('tr');
                        const toxicClass = row.is_toxic ? 'badge-toxic' : 'badge-non-toxic';
                        const statusText = row.is_toxic ? 'Токсичный' : 'Нейтральный';
                        tr.innerHTML = `
                            <td>${row.text.substring(0, 100)}${row.text.length > 100 ? '...' : ''}</td>
                            <td>${(row.prediction * 100).toFixed(1)}%</td>
                            <td><span class="status-badge ${toxicClass}">${statusText}</span></td>
                            <td>${new Date(row.timestamp).toLocaleString()}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                });
        }

        function submitComment() {
            const comment = document.getElementById('comment').value;
            if (!comment.trim()) {
                alert('Пожалуйста, введите комментарий');
                return;
            }

            fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: comment})
            })
            .then(res => res.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                const toxicClass = data.is_toxic ? 'toxic' : 'non-toxic';
                const statusText = data.is_toxic ? 'Токсичный' : 'Нейтральный';
                
                resultDiv.style.display = 'block';
                resultDiv.className = `result-box ${toxicClass}`;
                resultDiv.innerHTML = `
                    <strong>Результат:</strong> ${statusText}<br>
                    <strong>Вероятность токсичности:</strong> ${(data.prediction * 100).toFixed(2)}%
                `;
                
                document.getElementById('comment').value = '';
                loadHistory();
                loadStats();
            })
            .catch(error => {
                alert('Ошибка при обработке запроса');
                console.error(error);
            });
        }

        window.onload = function() {
            loadHistory();
            loadStats();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    pred, is_toxic = predict(text)
    
    if save_comment_to_db(text, pred, is_toxic):
        print(f"[INFO] Saved: {text[:50]}... -> {'TOXIC' if is_toxic else 'NON-TOXIC'}")
    
    return jsonify({
        'prediction': pred,
        'is_toxic': is_toxic
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    comments = get_recent_comments(20)
    return jsonify(comments)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    return jsonify(get_stats())

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности"""
    status = {
        'app': 'running',
        'model': 'loaded',
        'model_vocab_size': len(vocab),
        'database': 'unknown'
    }
    
    conn = get_db_connection()
    if conn:
        status['database'] = 'connected'
        conn.close()
    else:
        status['database'] = 'disconnected'
    
    return jsonify(status)

if __name__ == '__main__':
    print(f"[INFO] Starting Flask app with MySQL on {DEVICE}")
    print(f"[INFO] Model vocabulary size: {len(vocab)}")
    
    # Проверка подключения к MySQL
    conn = get_db_connection()
    if conn:
        print("[INFO] MySQL connection successful!")
        conn.close()
    else:
        print("[WARNING] MySQL connection failed!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)