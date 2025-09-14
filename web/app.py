from flask import Flask, request, render_template, send_file, redirect, url_for, session, jsonify
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
import tempfile
import shutil
import zipfile
from pathlib import Path
import unicodedata
import networkx as nx
from networkx.algorithms.components import connected_components

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, auth

def fix_nested_arrays(data):
    if isinstance(data, list):
        if any(isinstance(i, list) for i in data):
            return [{"items": sub} for sub in data]
        else:
            return data
    return data

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

def normalize_identifiers(code, base_code_tokens=None):
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
        normalized_code = ast.unparse(normalized_tree)
        
        # If base code tokens are provided, filter them out
        if base_code_tokens:
            # Tokenize the normalized code
            tokens = set(re.findall(r'\b\w+\b', normalized_code))
            # Remove base code tokens
            filtered_tokens = tokens - base_code_tokens
            # Reconstruct code with only non-base tokens
            # This is a simplified approach - in practice, you might want to use a more sophisticated method
            words = re.findall(r'\b\w+\b|\W+', normalized_code)
            filtered_code = ''.join([word if (word not in base_code_tokens or not word.isidentifier()) else '' for word in words])
            return filtered_code
        
        return normalized_code
    except:
        return code

def preprocess_code(code, base_code_tokens=None):
    code = remove_comments(code)
    code = normalize_identifiers(code, base_code_tokens)
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

def similarity_ast_with_exclusion(text1, text2, base_tokens):
    """Calculate AST similarity while excluding base code tokens"""
    try:
        # Parse and get AST dumps
        tree1 = ast.dump(ast.parse(text1))
        tree2 = ast.dump(ast.parse(text2))
        
        # Tokenize AST dumps and exclude base tokens
        tokens1 = set(re.findall(r'\b\w+\b', tree1)) - base_tokens
        tokens2 = set(re.findall(r'\b\w+\b', tree2)) - base_tokens
        
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    except Exception:
        return 0.0

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

# Bulk Analysis Functions
def extract_base_code_tokens(base_code_path):
    """Extract all tokens from base code for exclusion"""
    tokens = set()
    for root, _, files in os.walk(base_code_path):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    preprocessed = preprocess_code(content)
                    # Tokenize the code
                    tokens.update(set(re.findall(r'\b\w+\b', preprocessed)))
    return tokens

def cluster_similar_files(results, threshold):
    """
    Cluster files based on similarity scores using single-linkage clustering
    Returns a list of clusters and the detailed comparisons
    """
    # Create a graph where nodes are files and edges represent similarity above threshold
    G = nx.Graph()
    
    # Add all files as nodes and create edges for similar pairs
    for result in results:
        if result['score'] >= threshold:
            G.add_edge(result['file1'], result['file2'], weight=result['score'])
    
    # Find connected components (clusters)
    clusters = []
    for component in connected_components(G):
        if len(component) > 1:  # Only include clusters with at least 2 files
            clusters.append(list(component))
    
    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)
    
    return clusters, results

def generate_bulk_report(results, threshold):
    """Generate comprehensive PDF report for all comparisons"""
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], f"bulk_report_{uuid.uuid4().hex}.pdf")
    
    pdf = PDFReport()
    pdf.add_page()
    
    # Title and summary
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Bulk Code Similarity Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.add_section_title("Analysis Summary")
    pdf.add_text_block(f"Number of detected similarities: {len(results)}")
    pdf.add_text_block(f"Similarity threshold: {threshold:.0%}")
    pdf.add_text_block(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(5)
    
    # Summary table
    if results:
        pdf.add_section_title("Detected Similarities")
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(70, 10, "File 1", 1)
        pdf.cell(70, 10, "File 2", 1)
        pdf.cell(30, 10, "Score", 1, 1)
        
        pdf.set_font('Arial', '', 10)
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            pdf.cell(70, 10, result['file1'], 1)
            pdf.cell(70, 10, result['file2'], 1)
            pdf.cell(30, 10, f"{result['score']:.2%}", 1, 1)
    else:
        pdf.add_text_block("No significant similarities detected above the threshold.")
    
    # Detailed comparisons
    for i, result in enumerate(results):
        pdf.add_page()
        pdf.add_section_title(f"Detailed Comparison: {result['file1']} vs {result['file2']}")
        pdf.add_text_block(f"Similarity Score: {result['score']:.2%}")
        
        # Simple text representation of differences
        diff = list(unified_diff(
            result['preprocessed1'].splitlines(), 
            result['preprocessed2'].splitlines(), 
            lineterm='',
            fromfile=result['file1'],
            tofile=result['file2']
        ))
        
        pdf.add_text_block("Differences (simplified):")
        pdf.set_font('Courier', '', 8)
        for line in diff[:50]:  # Limit to first 50 lines
            pdf.multi_cell(0, 5, line)
        if len(diff) > 50:
            pdf.add_text_block("... (showing first 50 differences only)")
    
    pdf.output(report_path)
    return report_path

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
    
    # Check for error messages in session
    error_message = session.pop('error_message', '') if 'error_message' in session else ''
    return render_template('index.html', user=session['user'], error_message=error_message)

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
                if error_message in ["INVALID_PASSWORD", 'EMAIL_NOT_FOUND']:
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

@app.route('/progress')
def get_progress():
    return jsonify({
        'total': session.get('total_comparisons', 0),
        'completed': session.get('completed_comparisons', 0)
    })

@app.route('/heatmap')
def heatmap():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get history data
    user_history = (
        db.collection('history')
        .where('user', '==', session['user'])
        .order_by('timestamp', direction=firestore.Query.DESCENDING)
        .stream()
    )
    
    # Process data for heatmap
    files = set()
    comparisons = []
    
    for doc in user_history:
        data = doc.to_dict()
        file1 = data['filename1']
        file2 = data['filename2']
        score = data['score']
        
        files.add(file1)
        files.add(file2)
        comparisons.append((file1, file2, score))
    
    # Create heatmap data structure
    files = sorted(list(files))
    heatmap_data = [[0] * len(files) for _ in range(len(files))]
    
    for file1, file2, score in comparisons:
        i = files.index(file1)
        j = files.index(file2)
        heatmap_data[i][j] = score
        heatmap_data[j][i] = score  # Symmetric
    
    # Generate heatmap image
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], f"heatmap_{uuid.uuid4().hex}.png")
    
    plt.figure(figsize(10, 8))
    plt.imshow(heatmap_data, cmap='RdYlGn_r', interpolation='nearest')
    plt.colorbar(label='Similarity Score')
    plt.xticks(range(len(files)), files, rotation=45, ha='right')
    plt.yticks(range(len(files)), files)
    plt.title('Code Similarity Heatmap')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    
    return render_template('heatmap.html', 
                         heatmap_url=url_for('uploaded_file', filename=os.path.basename(heatmap_path)),
                         files=files,
                         data=heatmap_data)

@app.route('/bulk_similarity', methods=['POST'])
def bulk_similarity():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get uploaded folders
    submissions_folder = request.files.getlist('submissions_folder')
    base_code_folder = request.files.getlist('base_code_folder')
    threshold = float(request.form.get('threshold', THRESHOLD))
    
    # Server-side validation
    if not submissions_folder or len(submissions_folder) == 0 or submissions_folder[0].filename == '':
        session['error_message'] = "Please select a submissions folder before analyzing."
        return redirect(url_for('dashboard'))
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        submissions_path = os.path.join(temp_dir, 'submissions')
        base_code_path = os.path.join(temp_dir, 'base_code')
        os.makedirs(submissions_path)
        
        # Save uploaded files from folder
        for file in submissions_folder:
            if file.filename:  # Skip empty files
                # Create directory structure if needed
                file_path = os.path.join(submissions_path, file.filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
        
        # Process base code if provided
        base_code_tokens = set()
        if base_code_folder and base_code_folder[0].filename:
            os.makedirs(base_code_path)
            for file in base_code_folder:
                if file.filename:
                    # Create directory structure if needed
                    file_path = os.path.join(base_code_path, file.filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.save(file_path)
            base_code_tokens = extract_base_code_tokens(base_code_path)
        
        # Get all code files
        code_files = []
        for root, _, files in os.walk(submissions_path):
            for file in files:
                if file.endswith('.py'):
                    code_files.append(os.path.join(root, file))
        
        # Check if we have at least 2 files to compare
        if len(code_files) < 2:
            session['error_message'] = "Not enough Python files found for comparison. Please ensure your folder contains at least 2 Python files."
            return redirect(url_for('dashboard'))
        
        # Perform pairwise comparisons
        results = []
        total_comparisons = len(code_files) * (len(code_files) - 1) // 2
        completed = 0
        
        # Store progress in session
        session['total_comparisons'] = total_comparisons
        session['completed_comparisons'] = 0
        
        for i in range(len(code_files)):
            for j in range(i + 1, len(code_files)):
                file1 = code_files[i]
                file2 = code_files[j]
                
                try:
                    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
                        content1 = f1.read()
                        content2 = f2.read()
                    
                    # Preprocess code with base code exclusion
                    preprocessed1 = preprocess_code(content1, base_code_tokens)
                    preprocessed2 = preprocess_code(content2, base_code_tokens)
                    
                    # Calculate similarity with base code exclusion
                    similarity_score = similarity_ast_with_exclusion(
                        preprocessed1, preprocessed2, base_code_tokens
                    )
                    
                    # Store all results (not just those above threshold)
                    results.append({
                        'file1': os.path.basename(file1),
                        'file2': os.path.basename(file2),
                        'score': similarity_score,
                        'content1': content1,
                        'content2': content2,
                        'preprocessed1': preprocessed1,
                        'preprocessed2': preprocessed2
                    })
                except Exception as e:
                    # Skip files that can't be read
                    continue
                
                completed += 1
                session['completed_comparisons'] = completed
        
        # Cluster the results
        clusters, detailed_results = cluster_similar_files(results, threshold)
        
        # Store the analysis in Firestore
        
        timestamp = datetime.now().isoformat()
        report_data = {
            'user': session['user'],
            'timestamp': timestamp,
            'threshold': threshold,
            'clusters': fix_nested_arrays(clusters),
            'comparisons': fix_nested_arrays(detailed_results),
            'total_files': len(code_files)
        }

        print("REPORT DATA (Firestore safe):", report_data)

        
        # Add to Firestore and get the document ID
        doc_ref = db.collection('bulk_analyses').add(report_data)
        report_id = doc_ref[1].id  # Get the document ID
        
        # Redirect to the report view
        return redirect(url_for('bulk_report', report_id=report_id))

@app.route('/bulk_report/<report_id>')
def bulk_report(report_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get the report from Firestore
    doc_ref = db.collection('bulk_analyses').document(report_id)
    report_data = doc_ref.get().to_dict()
    
    # Check if the report exists and belongs to the current user
    if not report_data or report_data['user'] != session['user']:
        return render_template('error.html', error_message="Report not found or access denied.")
    
    return render_template('bulk_report.html', 
                         report=report_data,
                         report_id=report_id)

@app.route('/report_detail/<report_id>/<file1>/<file2>')
def report_detail(report_id, file1, file2):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get the report from Firestore
    doc_ref = db.collection('bulk_analyses').document(report_id)
    report_data = doc_ref.get().to_dict()
    
    # Check if the report exists and belongs to the current user
    if not report_data or report_data['user'] != session['user']:
        return render_template('error.html', error_message="Report not found or access denied.")
    
    # Find the specific comparison
    comparison = None
    for comp in report_data['comparisons']:
        if (comp['file1'] == file1 and comp['file2'] == file2) or \
           (comp['file1'] == file2 and comp['file2'] == file1):
            comparison = comp
            break
    
    if not comparison:
        return render_template('error.html', error_message="Comparison not found.")
    
    # Generate HTML diff for the comparison
    html_diff = HtmlDiff().make_table(
        comparison['preprocessed1'].splitlines(), 
        comparison['preprocessed2'].splitlines(), 
        comparison['file1'], 
        comparison['file2'], 
        context=True, 
        numlines=3
    )
    
    return render_template('results.html', 
                         method='AST', 
                         score=comparison['score'],
                         original_text1=comparison['content1'],
                         original_text2=comparison['content2'],
                         preprocessed_text1=comparison['preprocessed1'],
                         preprocessed_text2=comparison['preprocessed2'],
                         html_diff=html_diff, 
                         threshold_alert=comparison['score'] >= report_data['threshold'],
                         all_scores={'ast': comparison['score']})

@app.route('/similarity', methods=['POST'])
def compare():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    method = request.form.get('method', 'difflib')
    generate_report = 'report' in request.form

    # Server-side validation
    if not file1 or not file2 or file1.filename == '' or file2.filename == '':
        session['error_message'] = "Please select both files before comparing."
        return redirect(url_for('dashboard'))

    try:
        original_text1 = file1.read().decode('utf-8')
        original_text2 = file2.read().decode('utf-8')
    except UnicodeDecodeError:
        session['error_message'] = "Unable to read files. Please ensure they are text files."
        return redirect(url_for('dashboard'))
    
    # Reset file pointers for potential reuse
    file1.seek(0)
    file2.seek(0)
    
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