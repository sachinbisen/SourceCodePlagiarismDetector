from flask import Flask, request, render_template, send_file, redirect, url_for, session
from difflib import SequenceMatcher, HtmlDiff, unified_diff
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import math
import os
import uuid
import requests
from datetime import datetime
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, auth

# Initialize Firebase
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

FIREBASE_API_KEY = "AIzaSyAlcjBIud00nlGfXulxQtVpJQD529LXXwM"

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

THRESHOLD = 0.7

# Preprocessing Functions
def remove_comments(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    return code

def normalize_identifiers(code):
    try:
        tree = ast.parse(code)
    except:
        return code  # Fallback if parsing fails

    id_maps = {
        'variables': defaultdict(lambda: f'v{len(id_maps["variables"])}'),
        'functions': defaultdict(lambda: f'f{len(id_maps["functions"])}'),
        'classes': defaultdict(lambda: f'c{len(id_maps["classes"])}'),
        'parameters': defaultdict(lambda: f'p{len(id_maps["parameters"])}')
    }

    PRESERVED_NAMES = {
        'print', 'range', 'len', 'str', 'int', 'float', 'list',
        'dict', 'set', 'tuple', 'open', 'super', '__init__',
        'True', 'False', 'None'
    }

    class Normalizer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in PRESERVED_NAMES:
                return node
            if isinstance(node.ctx, ast.Store):
                node.id = id_maps['variables'][node.id]
            return node

        def visit_FunctionDef(self, node):
            if node.name not in PRESERVED_NAMES:
                node.name = id_maps['functions'][node.name]
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            if node.name not in PRESERVED_NAMES:
                node.name = id_maps['classes'][node.name]
            self.generic_visit(node)
            return node

        def visit_arg(self, node):
            if node.arg not in PRESERVED_NAMES:
                node.arg = id_maps['parameters'][node.arg]
            return node

        def visit_Attribute(self, node):
            return node

    normalizer = Normalizer()
    normalized_tree = normalizer.visit(tree)
    
    try:
        return ast.unparse(normalized_tree)
    except:
        return code

def preprocess_code(code):
    code = remove_comments(code)
    code = normalize_identifiers(code)
    return code

# Similarity Calculation Functions
def similarity_difflib(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def similarity_cosine(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    dot_product = sum(a * b for a, b in zip(vectors[0], vectors[1]))
    norm1 = math.sqrt(sum(a * a for a in vectors[0]))
    norm2 = math.sqrt(sum(b * b for b in vectors[1]))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def similarity_ast(text1, text2):
    try:
        tree1 = ast.dump(ast.parse(text1))
        tree2 = ast.dump(ast.parse(text2))
        return SequenceMatcher(None, tree1, tree2).ratio()
    except Exception:
        return 0.0

def similarity_jaccard(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

# Visualization Functions
def generate_similarity_chart(score, output_path):
    labels = ['Similarity', 'Difference']
    sizes = [score * 100, 100 - (score * 100)]
    colors = ['#4caf50', '#f44336']
    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Similarity Score')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_all_methods_barchart(scores, output_path):
    methods = list(scores.keys())
    values = [scores[m] * 100 for m in methods]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, values, color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'])
    
    plt.title('Comparison Using All Methods')
    plt.ylabel('Similarity Score (%)')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Database Functions
def save_history_to_firestore(username, filename1, filename2, method, score):
    timestamp = datetime.now().isoformat()
    db.collection('history').add({
        'user': username,
        'timestamp': timestamp,
        'filename1': filename1,
        'filename2': filename2,
        'method': method,
        'score': round(score, 4)
    })

# PDF Report Class
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Code Similarity Report', 0, 1, 'C')
        self.ln(5)

    def add_section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)

    def add_text_block(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, self._safe_text(text))
        self.ln(3)

    def add_diff_block(self, text1, text2):
        self.set_font('Courier', '', 8)
        diff = list(unified_diff(
            text1.splitlines(), 
            text2.splitlines(), 
            lineterm='',
            fromfile='file1',
            tofile='file2'
        ))
        
        for line in diff:
            safe_line = self._safe_text(line)
            if safe_line.startswith('+'):
                self.set_text_color(0, 128, 0)  # Green
            elif safe_line.startswith('-'):
                self.set_text_color(255, 0, 0)   # Red
            elif safe_line.startswith('@@'):
                self.set_text_color(0, 0, 255)   # Blue
            else:
                self.set_text_color(0, 0, 0)     # Black
            
            self.multi_cell(0, 5, safe_line)
        
        self.set_text_color(0, 0, 0)
        self.ln(3)
    
    def add_image(self, path, w=100):
        """Add an image to the PDF with specified width"""
        if os.path.exists(path):
            try:
                self.image(path, w=w)
                self.ln(5)
            except Exception as e:
                self.add_text_block(f"[Could not insert image: {str(e)}]")
    
    def _safe_text(self, text):
        """Convert Unicode characters to their closest ASCII equivalents"""
        try:
            return text.encode('latin-1', 'replace').decode('latin-1')
        except:
            normalized = unicodedata.normalize('NFKD', text)
            ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
            return ascii_text or "[non-ascii-content]"
# Routes
@app.route('/')
def home():
    return render_template('intro.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")

        try:
            signup_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
            signup_payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            signup_response = requests.post(signup_url, json=signup_payload)

            if signup_response.status_code == 400:
                error_data = signup_response.json()
                error_message = error_data.get("error", {}).get("message", "")
                if error_message == "EMAIL_EXISTS":
                    return render_template("signup.html", error="Email already exists. Please log in.")
                return render_template("signup.html", error=error_message)

            signup_response.raise_for_status()
            signup_data = signup_response.json()
            id_token = signup_data["idToken"]

            verify_url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
            verify_payload = {
                "requestType": "VERIFY_EMAIL",
                "idToken": id_token
            }
            requests.post(verify_url, json=verify_payload)

            return render_template("signup.html", success="Account created. Please check your email to verify.")

        except Exception as e:
            return render_template('signup.html', error="An error occurred. Please try again later.")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        try:
            response = requests.post(
                f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}",
                json=payload
            )

            if response.status_code == 400:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "")
                if error_message in ["INVALID_PASSWORD", "EMAIL_NOT_FOUND"]:
                    return render_template('login.html', error="Incorrect credentials. Please try again.")
                else:
                    return render_template('login.html', error="Login failed: " + error_message)

            data = response.json()
            id_token = data.get("idToken")

            info_response = requests.post(
                f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}",
                json={"idToken": id_token}
            )
            info_data = info_response.json()

            if not info_data['users'][0].get('emailVerified', False):
                return render_template('login.html', error="Please verify your email before logging in.")

            session['user'] = email
            return redirect(url_for('dashboard'))

        except Exception as e:
            return render_template('login.html', error="Something went wrong. Please try again.")

    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        reset_url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
        payload = {
            "requestType": "PASSWORD_RESET",
            "email": email
        }
        response = requests.post(reset_url, json=payload)
        data = response.json()

        if "error" in data:
            return render_template("forgot_password.html", error=data["error"]["message"])
        return render_template("forgot_password.html", success="Password reset email sent successfully.")
    return render_template("forgot_password.html")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user_history = (
        db.collection('history')
        .where('user', '==', session['user'])
        .order_by('timestamp', direction=firestore.Query.DESCENDING)
        .stream()
    )
    
    entries = []
    for doc in user_history:
        timestamp = doc.to_dict()['timestamp']
        # Parse ISO format datetime
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        entries.append([
            dt.strftime('%Y-%m-%d'),  # Date only
            dt.strftime('%H:%M:%S'),  # Time only
            doc.to_dict()['filename1'],
            doc.to_dict()['filename2'],
            doc.to_dict()['method'],
            f"{doc.to_dict()['score']*100:.2f}%",  # Formatted percentage
            doc.id
        ])
    
    return render_template('history.html', entries=entries)

@app.route('/similarity', methods=['POST'])
def compare():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    method = request.form.get('method', 'difflib')
    generate_report = 'report' in request.form

    if not file1 or not file2:
        return render_template('results.html', method=method, score=0.0)

    original_text1 = file1.read().decode('utf-8')
    original_text2 = file2.read().decode('utf-8')
    
    preprocessed_text1 = preprocess_code(original_text1)
    preprocessed_text2 = preprocess_code(original_text2)
    
    all_scores = {
        'difflib': similarity_difflib(preprocessed_text1, preprocessed_text2),
        'cosine': similarity_cosine(preprocessed_text1, preprocessed_text2),
        'ast': similarity_ast(preprocessed_text1, preprocessed_text2),
        'jaccard': similarity_jaccard(preprocessed_text1, preprocessed_text2)
    }
    
    score = all_scores[method]
    save_history_to_firestore(session['user'], file1.filename, file2.filename, method, score)

    html_diff = HtmlDiff().make_table(
        preprocessed_text1.splitlines(), 
        preprocessed_text2.splitlines(), 
        file1.filename, 
        file2.filename, 
        context=True, 
        numlines=3
    )
    
    threshold_alert = score >= THRESHOLD

    if generate_report:
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{uuid.uuid4().hex}.pdf")
        chart_path = os.path.join(app.config['UPLOAD_FOLDER'], f"chart_{uuid.uuid4().hex}.png")
        bar_chart_path = os.path.join(app.config['UPLOAD_FOLDER'], f"barchart_{uuid.uuid4().hex}.png")
        
        generate_similarity_chart(score, chart_path)
        generate_all_methods_barchart(all_scores, bar_chart_path)
        
        pdf = PDFReport()
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Code Similarity Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        pdf.add_section_title("Analysis Summary")
        pdf.add_text_block(f"Comparison between: {file1.filename} and {file2.filename}")
        pdf.add_text_block(f"Selected Method: {method.upper()}")
        pdf.add_text_block(f"Similarity Score: {score:.2%}")
        pdf.add_text_block(f"Plagiarism Threshold ({THRESHOLD:.0%}) Exceeded: {'Yes' if threshold_alert else 'No'}")
        pdf.ln(5)
        
        pdf.add_section_title("Similarity Visualizations")
        pdf.add_image(chart_path, w=80)
        pdf.add_image(bar_chart_path, w=180)
        pdf.ln(5)
        
        pdf.add_section_title("Preprocessing Details")
        pdf.add_text_block("Before comparison, the following preprocessing steps were applied:")
        pdf.add_text_block("1. All comments were removed")
        pdf.add_text_block("2. All identifiers were normalized (variables → vN, functions → fN, etc.)")
        pdf.ln(3)
        
        pdf.add_section_title("Original vs Preprocessed Code")
        pdf.add_text_block(f"Original {file1.filename}:")
        pdf.set_font('Courier', '', 8)
        pdf.multi_cell(0, 5, original_text1[:1000] + ("..." if len(original_text1) > 1000 else ""))
        pdf.ln(3)
        
        pdf.add_text_block(f"Preprocessed {file1.filename}:")
        pdf.multi_cell(0, 5, preprocessed_text1[:1000] + ("..." if len(preprocessed_text1) > 1000 else ""))
        pdf.ln(5)
        
        pdf.add_text_block(f"Original {file2.filename}:")
        pdf.multi_cell(0, 5, original_text2[:1000] + ("..." if len(original_text2) > 1000 else ""))
        pdf.ln(3)
        
        pdf.add_text_block(f"Preprocessed {file2.filename}:")
        pdf.multi_cell(0, 5, preprocessed_text2[:1000] + ("..." if len(preprocessed_text2) > 1000 else ""))
        pdf.ln(5)
        
        pdf.add_section_title("Detailed Differences (Preprocessed Code)")
        pdf.add_diff_block(preprocessed_text1, preprocessed_text2)
        
        pdf.output(report_path)
        return send_file(report_path)

    return render_template('results.html', 
                         method=method.upper(), 
                         score=score,
                         original_text1=original_text1,
                         original_text2=original_text2,
                         preprocessed_text1=preprocessed_text1,
                         preprocessed_text2=preprocessed_text2,
                         html_diff=html_diff, 
                         threshold_alert=threshold_alert,
                         all_scores=all_scores)

if __name__ == '__main__':
    app.run(debug=True, port=5001)