# 🔍 Code Similarity Detector

A comprehensive web application for detecting source code similarity using advanced NLP and text-processing techniques. Designed for educators, developers, and organizations to identify plagiarism or code reuse efficiently.

---

## 📌 Project Overview

This system compares source code files using multiple algorithms, offering powerful analysis and visual reporting. It supports secure user authentication, detailed history tracking, and customizable comparison options.

---

## 🚀 Key Features

### 🔍 Multiple Comparison Methods
- **Difflib** – Sequence matching
- **Cosine Similarity** – TF-IDF vector-based
- **AST Comparison** – Abstract Syntax Tree-based
- **Jaccard Similarity** – Token-based

### 🧹 Advanced Preprocessing
- Comment removal
- Identifier normalization
- Code structure analysis

### 👥 User Management
- Secure authentication using Firebase
- Email verification
- Password reset

### 📊 Visual Reporting
- Interactive HTML result display
- PDF report generation
- Visual charts & graphs

### 📁 History Tracking
- Stores all previous comparisons
- Similarity score visualization
- Time-based filtering

---

## 🛠 Technologies Used

### 🔧 Backend
- Python 3
- Flask
- Firebase (Authentication & Database)
- Scikit-learn (TF-IDF)
- AST Module
- Difflib
- FPDF
- Matplotlib

### 🎨 Frontend
- HTML5, CSS3, JavaScript
- Bootstrap 5
- Font Awesome
- Fully responsive design

---

## 🚢 Deployment

- Compatible with any WSGI-supported server (e.g., Gunicorn, uWSGI)
- Requires Firebase project setup

---

## ⚙️ Installation

### ✅ Prerequisites
- Python 3.7+
- Firebase project with Email/Password Authentication enabled
- Firebase Admin SDK credentials (`firebase_config.json`)

### 📥 Steps

1. **Clone the Repository**

```bash
git clone [https://github.com/yourusername/code-similarity-detector.git](https://github.com/Rohitgajbhiye2005/CodeSimDetect.git)
cd code-similarity-detector
cd web
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Firebase Setup**

- Create a Firebase project: https://console.firebase.google.com/
- Enable **Email/Password** authentication
- Download the **Service Account JSON** and save it as:

```bash
firebase_config.json
```

- Set your Firebase API key in `app.py`:

```python
FIREBASE_API_KEY = "your_api_key_here"
```

4. **Run the Application**

```bash
python app.py
```

Open in your browser:

```
http://localhost:5001
```

---

## 📂 Project Structure

```
code-similarity-detector/
├── web/
│   ├── app.py                   # Main application
│   ├── firebase_config.json     # Firebase credentials
│   ├── static/                  # Static files (CSS, JS, images)
│   │   ├── cool.css             # Main stylesheet
│   │   ├── logo.png             # Application logo
│   │   ├── background.jpeg      # Background image
│   │   └── sounds/
│   │       └── launch.mp3       # Sound effects
│   ├── templates/               # HTML templates
│   │   ├── intro.html           # Landing page
│   │   ├── index.html           # Dashboard
│   │   ├── login.html           # Login page
│   │   ├── signup.html          # Signup page
│   │   ├── forgot_password.html # Password reset
│   │   ├── uploads.html         # Upload page (legacy)
│   │   ├── results.html         # Comparison results
│   │   ├── history.html         # User history
│   │   └── pdf_template.html    # PDF export template
└── README.md                # Project documentation
```

---

## 📖 Usage

### 👤 Registration & Login
- Sign up through the UI
- Verify your email
- Log in to access the dashboard

### 🧪 Code Comparison
- Upload two code files
- Choose a similarity algorithm
- View detailed HTML results or export as PDF

### 🕓 History
- Track all past comparisons
- Filter by method or date
- Visualize score trends

---

## 📡 API Endpoints

| Method | Endpoint               | Description                   |
|--------|------------------------|-------------------------------|
| GET    | `/`                    | Landing page                  |
| GET    | `/dashboard`           | Main dashboard                |
| GET/POST | `/login`            | Login form and handler        |
| GET/POST | `/signup`           | Signup form and handler       |
| GET/POST | `/forgot-password`  | Reset password functionality  |
| GET    | `/logout`             | Logout and end session        |
| GET    | `/history`            | View past comparisons         |
| POST   | `/similarity`         | Trigger code comparison       |

---

## 🌟 Future Enhancements

- **Language Support**: Java, C++, JavaScript, and more
- **Visualization**: Side-by-side diff viewer
- **Batch Processing**: Upload and compare multiple files
- **AI Detection**: ML-based similarity scoring
- **Collaboration**: Team workspaces and shared results
- **Bulk Submission Support**: Upload and compare entire folders of code submissions
- **Pairwise Comparison**: Automatically compare each file with every other file
- **Clustering of Similar Submissions**: Group submissions based on similarity scores
- **Plagiarism Reports**: Detailed reports showing pairwise similarity percentages
- **Downloadable Reports via Email**: Get results in PDF/CSV format sent to your email
- **AST-Based Detection**: Currently supports Python; future support for Java, C++, etc.


---

## 👥 Contributors

- **Krutika Funde**
- **Raj Dhobale**  
- **Rohit S Gajbhiye**  
- **Sachin N Bisen**  
- **Pritam Chaudhari**  
- **Prathmesh N Ghormade**  

---

## 📜 License

This project is licensed under the [MIT License].

---

## 🙏 Acknowledgments

- **Firebase** – for authentication & backend services  
- **Python difflib** – for string matching  
- **Scikit-learn** – for TF-IDF implementation  
- **Flask** – for web framework support

---

> 💡 *Empowering developers and educators to ensure code integrity with intelligent comparison tools and techniques.*

