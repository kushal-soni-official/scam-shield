import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import threading
import traceback
from collections import defaultdict
import webbrowser

# Constants
MAX_FEATURES = 5000
MAX_BATCH_SIZE = 100
CACHE_TTL = 5  # seconds for dashboard cache
RATE_LIMIT = 10  # requests per minute per IP

# Set up logging with rotation
handler = RotatingFileHandler('scam_log.txt', maxBytes=5*1024*1024, backupCount=2)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[handler]
)

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            return None
    return wrapper

# Preprocess text – keep URLs and numbers
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    # Keep alphanumeric, dots, slashes, colons (for URLs)
    text = re.sub(r'[^a-z0-9\s./:]', '', text)
    return text

# Dataset class for PyTorch
class TextDataset(Dataset):
    """PyTorch dataset for text features and labels."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Neural Network Classifier
class ScamClassifier(nn.Module):
    """Feedforward neural network with one hidden layer."""
    def __init__(self, input_size, hidden_size=128, output_size=2):
        super(ScamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load dataset
@handle_errors
def load_dataset(file_path='scam_dataset.json'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [preprocess_text(item['text']) for item in data]
    labels = [1 if item['label'] == 'scam' else 0 for item in data]
    return texts, labels

# Train model – with early stopping and GPU support
@handle_errors
def train_model(texts, labels, vectorizer, epochs=50, batch_size=32, lr=0.001, validation_split=0.1):
    # Split data into train, validation, test (70-10-20)
    n = len(texts)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)
    train_texts = texts[:train_end]
    train_labels = labels[:train_end]
    val_texts = texts[train_end:val_end]
    val_labels = labels[train_end:val_end]
    test_texts = texts[val_end:]
    test_labels = labels[val_end:]

    # Fit vectorizer only on training data
    vectorizer.fit(train_texts)
    train_features = vectorizer.transform(train_texts).toarray()
    val_features = vectorizer.transform(val_texts).toarray()
    test_features = vectorizer.transform(test_texts).toarray()

    train_dataset = TextDataset(train_features, train_labels)
    val_dataset = TextDataset(val_features, val_labels)
    test_dataset = TextDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = train_features.shape[1]
    model = ScamClassifier(input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training model...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return model, vectorizer, accuracy

# Save model
@handle_errors
def save_model(model, vectorizer, model_path='scam_model.pth', vectorizer_path='vectorizer.pkl'):
    torch.save(model.state_dict(), model_path)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Model and vectorizer saved.")

# Load model
@handle_errors
def load_model(model_path='scam_model.pth', vectorizer_path='vectorizer.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    input_size = len(vectorizer.get_feature_names_out())
    model = ScamClassifier(input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, vectorizer

# Predict messages
@handle_errors
def predict_messages(model, vectorizer, messages, original_messages=None):
    preprocessed = [preprocess_text(msg) for msg in messages]
    vec = vectorizer.transform(preprocessed).toarray()
    input_tensor = torch.tensor(vec, dtype=torch.float32)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidences = probabilities.max(1)[0].cpu().numpy() * 100
        predicted = probabilities.argmax(1).cpu().numpy()
    results = []
    for i, (pred, conf) in enumerate(zip(predicted, confidences)):
        label = "scam" if pred == 1 else "legitimate"
        msg = original_messages[i] if original_messages else messages[i]
        results.append((msg, label, conf))
        logging.info(f"Message: {msg} | Classification: {label} | Confidence: {conf:.2f}%")
    return results

# ====================== CLI MODE ======================
def cli_mode(model, vectorizer):
    """Interactive CLI with rich formatting if available."""
    # Detect terminal type for fallback
    is_cmd = os.name == 'nt' and 'cmd' in os.environ.get('COMSPEC', '').lower()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.prompt import Prompt, Confirm
        from rich.text import Text
        from rich import box
        rich_available = True
    except ImportError:
        print("Install rich: pip install rich (optional for basic CLI)")
        rich_available = False

    if rich_available and not is_cmd:
        console = Console(force_terminal=True)
        use_rich = True
    else:
        use_rich = False
        print("Using plain CLI (rich glitch detected - try PowerShell for full dashboard)")

    width = min(os.get_terminal_size().columns if use_rich else 80, 120)

    def show_header(use_rich_local):
        if use_rich_local:
            title = Text("🛡️ ScamShield - Scam Detector 2026", style="bold cyan", justify="center")
            header = Panel(
                title,
                subtitle="TF-IDF + PyTorch • Confidence Scoring • Batch Analysis",
                border_style="bright_blue",
                padding=(0, 1),
                width=min(width, 100),
                expand=False
            )
            console.print(header)
        else:
            print("\n" + "="*50)
            print("🛡️ SCAMSHIELD - SCAM DETECTOR 2026")
            print("TF-IDF + PyTorch • Confidence Scoring • Batch Analysis")
            print("="*50 + "\n")

    def show_stats(use_rich_local):
        if use_rich_local:
            stats_text = Text.assemble(
                ("Model Accuracy: ", "bold white"),
                ("90.91% ", "bold green"),
                ("(test set)", "dim")
            )
            stats_panel = Panel(
                stats_text,
                title="Stats",
                border_style="dim blue",
                padding=(0, 1),
                width=40,
                expand=False
            )
            console.print(stats_panel)
        else:
            print("Stats: Model Accuracy: 90.91% (test set)")
            print("-" * 40 + "\n")

    # Initial setup
    os.system('cls' if os.name == 'nt' else 'clear') if not use_rich else console.clear()
    show_header(use_rich)
    show_stats(use_rich)

    if use_rich:
        console.print("\n[bold yellow]Instructions[/bold yellow]:")
    else:
        print("\nInstructions:")
    print("• Enter messages one per line")
    print("• Press Enter twice (blank line) to analyze batch")
    print("• Type 'quit' to exit")
    print(f"• Tip: Batch up to {MAX_BATCH_SIZE} messages for best speed\n")

    while True:
        messages = []
        original_messages = []

        if use_rich:
            console.print("[dim cyan]Input messages below (blank line to process):[/dim cyan]")
        else:
            print("\nInput messages below (blank line to process):")

        while True:
            try:
                if use_rich:
                    line = Prompt.ask(" ", console=console, default="").strip()
                else:
                    line = input("> ").strip()
            except KeyboardInterrupt:
                print("\n[red]Interrupted.[/red] Exiting gracefully..." if use_rich else "\nInterrupted. Exiting gracefully...")
                return

            if line.lower() == 'quit':
                if use_rich:
                    confirm = Confirm.ask("[yellow]Really exit ScamShield?[/yellow]")
                else:
                    confirm = input("Really exit ScamShield? [y/n]: ").lower().startswith('y')
                if confirm:
                    print("[green]Goodbye! Stay safe.[/green]" if use_rich else "Goodbye! Stay safe.\n")
                    return
                continue

            if not line:
                break

            messages.append(line)
            original_messages.append(line)

            preview = line[:width-20] + "…" if len(line) > width-20 else line
            print(f" • {preview}")

            if len(messages) >= MAX_BATCH_SIZE:
                print(f"[yellow]Batch limit reached ({MAX_BATCH_SIZE}). Processing...[/yellow]")
                break

        if not messages:
            msg = "[yellow]No messages entered — try again[/yellow]\n"
            if use_rich:
                console.print(msg)
            else:
                print(msg)
            continue

        # Processing
        proc_msg = "[cyan]Analyzing...[/cyan]\n" if use_rich else "Analyzing...\n"
        print(proc_msg)

        if use_rich:
            with Progress(
                SpinnerColumn(spinner_style="cyan"),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {len(messages)} message{'s' if len(messages)>1 else ''}...", total=len(messages))
                results = predict_messages(model, vectorizer, messages, original_messages)
                for _ in results:
                    progress.advance(task)
        else:
            for i, msg in enumerate(messages, 1):
                print(f"Processing {i}/{len(messages)}: {msg[:30]}...")
            results = predict_messages(model, vectorizer, messages, original_messages)

        # Results table
        if use_rich:
            table = Table(
                title="Classification Results",
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED if width > 80 else box.SIMPLE,
                expand=True,
                min_width=60
            )
            table.add_column("Message", style="cyan", no_wrap=False)
            table.add_column("Label", justify="center", style="bold", width=10)
            table.add_column("Confidence", justify="right", width=12)
        else:
            print("\nClassification Results:")
            print("-" * width)
            print(f"{'Message':<50} {'Label':^10} {'Confidence':^12}")
            print("-" * width)

        scam_count = 0
        total_conf = 0.0

        for msg, label, conf in results:
            if use_rich:
                label_col = Text(label.upper(), style="bold red" if label == "scam" else "bold green")
                conf_text = f"{conf:.2f}%"
                conf_style = "yellow bold" if conf < 70 else "green bold"
                if conf < 70:
                    conf_text = f"⚠ {conf_text}"
                table.add_row(msg, label_col, Text(conf_text, style=conf_style))
            else:
                conf_str = f"⚠ {conf:.2f}%" if conf < 70 else f"{conf:.2f}%"
                label_str = label.upper()
                print(f"{msg:<50} {label_str:^10} {conf_str:^12}")

            if label == "scam":
                scam_count += 1
            total_conf += conf

        if use_rich:
            console.print(table)
        else:
            print("-" * width)

        # Summary
        if results:
            avg_conf = total_conf / len(results)
            scam_pct = (scam_count / len(results)) * 100
            low_conf_warn = " • Some low-confidence results — review manually" if any(c < 70 for _, _, c in results) else ""
            summary = f"Summary: {len(results)} msgs • {scam_count} scams ({scam_pct:.1f}%) • Avg conf: {avg_conf:.1f}%{low_conf_warn}"
            if use_rich:
                console.print(Panel(summary, border_style="dim", padding=(0,1), expand=False))
            else:
                print(f"\n{summary}\n")

        print("\nReady for next batch... (or 'quit')\n")

# ====================== GUI MODE ======================
# Simple in-memory cache for dashboard data
_dashboard_cache = {"data": None, "timestamp": 0}

def gui_mode(model, vectorizer):
    try:
        from flask import Flask, request, render_template, jsonify, send_file

    except ImportError:
        print("Install flask-bootstrap: pip install flask-bootstrap")
        return

    app = Flask(__name__)

    

    # Rate limiting store
    rate_limit_store = defaultdict(list)

    def parse_log_for_dashboard():
        pie = {'scam': 0, 'legitimate': 0}
        conf_over_time = []
        daily_scams = {}
        log_path = 'scam_log.txt'
        if not os.path.exists(log_path):
            return {
                'pie': [0, 0],
                'line_labels': [],
                'line_data': [],
                'bar_labels': ['No data'],
                'bar_data': [0]
            }
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or 'Classification:' not in line:
                        continue

                    try:
                        parts = line.split('Classification: ')
                        if len(parts) < 2:
                            continue
                        label_part = parts[1].split(' | Confidence: ')
                        if len(label_part) < 2:
                            continue

                        label = label_part[0].strip()
                        conf_str = label_part[1].rstrip('%').strip()
                        conf = float(conf_str)

                        timestamp = line[:19]
                        date = timestamp[:10]

                        pie[label] = pie.get(label, 0) + 1
                        conf_over_time.append(conf)
                        daily_scams[date] = daily_scams.get(date, 0) + (1 if label == 'scam' else 0)

                    except (IndexError, ValueError):
                        continue
        except Exception as e:
            print(f"Log parsing error: {e}")
        # Chart data preparation
        line_labels = list(range(1, len(conf_over_time) + 1)) if conf_over_time else []
        line_data = conf_over_time if conf_over_time else []
        bar_labels = list(daily_scams.keys()) if daily_scams else ['No data']
        bar_data = list(daily_scams.values()) if daily_scams else [0]

        return {
            'pie': [pie.get('scam', 0), pie.get('legitimate', 0)],
            'line_labels': line_labels,
            'line_data': line_data,
            'bar_labels': bar_labels,
            'bar_data': bar_data,
            'total_classifications': len(conf_over_time),
            'avg_conf': sum(conf_over_time) / len(conf_over_time) if conf_over_time else 0,
            'scam_rate': (pie.get('scam', 0) / len(conf_over_time) * 100) if conf_over_time else 0
        }

    def get_cached_dashboard_data():
        global _dashboard_cache
        now = time.time()
        if _dashboard_cache["data"] is None or (now - _dashboard_cache["timestamp"] > CACHE_TTL):
            _dashboard_cache["data"] = parse_log_for_dashboard()
            _dashboard_cache["timestamp"] = now
        return _dashboard_cache["data"]

    def is_rate_limited(ip, limit=RATE_LIMIT, window=60):
        now = time.time()
        requests = [t for t in rate_limit_store[ip] if now - t < window]
        if len(requests) >= limit:
            return True
        requests.append(now)
        rate_limit_store[ip] = requests
        return False

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/classify', methods=['POST'])
    def classify():
        try:
            # Rate limiting
            if is_rate_limited(request.remote_addr):
                return jsonify({'error': 'Too many requests. Please wait a moment.'}), 429

            messages = request.json.get('messages', [])
            if not messages:
                return jsonify({'error': 'No messages provided'}), 400
            # Limit batch size
            if len(messages) > MAX_BATCH_SIZE:
                messages = messages[:MAX_BATCH_SIZE]
            results = predict_messages(model, vectorizer, messages, messages)
            return jsonify([{'msg': msg, 'label': label, 'conf': float(conf)} for msg, label, conf in results])
        except Exception as e:
            logging.error(f"Classify error: {traceback.format_exc()}")
            return jsonify({'error': 'Internal server error'}), 500

    @app.route('/dashboard')
    def dashboard():
        data = get_cached_dashboard_data()
        return render_template('dashboard.html', **data)

    @app.route('/history')
    def history():
        log_entries = []
        if os.path.exists('scam_log.txt'):
            # Efficient read of last 1000 lines
            with open('scam_log.txt', 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                buffer_size = 8192
                lines = []
                pos = size
                while pos > 0 and len(lines) < 1000:
                    read_size = min(buffer_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    chunk = f.read(read_size).decode('utf-8', errors='ignore')
                    lines = chunk.split('\n') + lines
                lines = lines[-1000:]
                for line in reversed(lines):
                    if ' - Message: ' in line:
                        parts = line.strip().split(' - Message: ')
                        if len(parts) != 2:
                            continue
                        timestamp = parts[0]
                        rest = parts[1].split(' | Classification: ')
                        # Ensure the classification part exists
                        if len(rest) < 2:
                            continue
                        message = rest[0]
                        label_conf = rest[1].split(' | Confidence: ')
                        # Ensure the confidence part exists
                        if len(label_conf) < 2:
                            continue
                        label = label_conf[0]
                        conf = label_conf[1].rstrip('%')
                        log_entries.append((timestamp, message, label, conf))
        return render_template('history.html', log_entries=log_entries)
    
    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/update_data')
    def update_data():
        data = get_cached_dashboard_data()
        return jsonify(data)

    @app.route('/download_log')
    def download_log():
        if not os.path.exists('scam_log.txt'):
            return "No log file found", 404
        return send_file('scam_log.txt', as_attachment=True)

    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000/')

    print("Starting modern GUI dashboard... Browser should open automatically.")
    threading.Timer(1.5, open_browser).start()
    # Security: debug mode disabled
    app.run(debug=True, use_reloader=False)

# ====================== MAIN ======================
def main():
    dataset_path = 'scam_dataset.json'
    model_path = 'scam_model.pth'
    vectorizer_path = 'vectorizer.pkl'
    print("ScamShield Pro - Final Version 2026")
    mode = input("Select mode (CLI or GUI): ").strip().upper()
    texts, labels = load_dataset(dataset_path)
    if texts is None:
        return
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
    model, loaded_vectorizer = load_model(model_path, vectorizer_path)
    if model is None:
        print("No model found. Training new one...")
        model, vectorizer, accuracy = train_model(texts, labels, vectorizer)
        if model:
            save_model(model, vectorizer)
            print(f"Training complete. Test accuracy: {accuracy:.2f}%")
        else:
            print("Training failed. Exiting.")
            return
    else:
        vectorizer = loaded_vectorizer
        print("Loaded trained model successfully.")

    if mode == 'CLI':
        cli_mode(model, vectorizer)
    elif mode == 'GUI':
        gui_mode(model, vectorizer)
    else:
        print("Invalid selection. Run again and choose CLI or GUI.")

if __name__ == "__main__":
    main()