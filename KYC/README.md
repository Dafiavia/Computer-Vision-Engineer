# 🛡️ End-to-End AI KYC Verification System

## 📋 Project Overview

The **KYC (Know Your Customer) Verification System** is a fully automated digital identity verification solution built from scratch. This project simulates a professional fintech onboarding process by combining **Computer Vision (OCR)** for document data extraction and **Face Recognition** for biometric verification.

The goal of this project is to solve the inefficiency of manual identity checks by providing a real-time, accurate, and secure automated validation system.

---

## 🚀 Key Features

* **Smart OCR Extraction:** Utilizes *Deep Learning* (EasyOCR) to automatically detect, read, and extract key fields (ID Number, Name, DOB) from Identity Cards (Case Study: Spanish DNI).
* **Data Validation:** Implements automated logic validation, including *checksum verification* for ID numbers to prevent fraud.
* **Biometric Registration:** High-precision face detection using **MTCNN** and conversion of facial features into 512-dimensional *vector embeddings* using **Inception Resnet (V1)**.
* **Identity Matching:** Performs one-to-one verification using **Cosine Similarity** algorithms to match live selfies against the registered identity database.
* **Interactive UI:** A responsive web interface providing real-time visual feedback (bounding boxes) and verification status.

---

## 🛠️ Tech Stack

This project demonstrates a **Full-Cycle Data Science** approach, ranging from AI model implementation to web application deployment.

| Category | Technology |
| :--- | :--- |
| **Backend & API** | Python, **FastAPI** (RESTful API), Uvicorn |
| **AI / Machine Learning** | **PyTorch**, FaceNet, MTCNN, EasyOCR, Scikit-learn |
| **Data Processing** | NumPy, Pandas, Regex (Regular Expressions) |
| **Frontend** | HTML5, **Tailwind CSS**, Vanilla JavaScript (Fetch API) |
| **Database** | JSON & CSV (Optimized for Vector Retrieval & Local Storage) |
| **Version Control** | Git & GitHub |

---

## ⚙️ System Workflow

1.  **Document Registration (ID Card):**
    * User uploads an ID Card image.
    * System pre-processes the image (grayscale/thresholding) and extracts text.
    * Data is stored after passing Regex patterns and checksum validation.

2.  **Face Enrollment:**
    * User uploads a reference photo.
    * System detects the face, performs alignment, and generates a *vector embedding*.
    * The vector is stored in the search database.

3.  **Identity Verification (KYC Check):**
    * User uploads a verification selfie.
    * System compares the new face vector against the database using *Cosine Similarity*.
    * **Output:** Match Status (MATCH/NOT MATCH), Similarity Score, and User Details.

---

## 📸 Demo Screenshots

### 1. ID Card Registration & OCR
*(Paste a screenshot of Step 1 here)*
<p align="center">
  <img src="dataset/doc/Screenshot%202025-11-28%20180405.png" width="45%" />
  <img src="dataset/doc/Screenshot%202025-11-28%20215009.png" width="45%" /> 
</p>


### 2. Face Verification (Successful Match)
*(Paste a screenshot of Step 3 here, showing the green bounding box)*
<p align="center">
  <img src="dataset/doc/Screenshot 2025-11-28 213112.png" width="45%" />
  <img src="dataset/doc/Screenshot 2025-11-28 215101.png" width="45%" /> 
</p>

---

## 🧠 Challenges & Key Takeaways

During the development of this project, I tackled several key engineering challenges:

* **Deep Learning Integration:** Optimizing PyTorch model loading times to ensure efficient execution within the FastAPI HTTP server environment.
* **Data Cleaning:** Handling OCR noise (misread characters) by implementing strict **Regex Pattern Matching** and validation logic.
* **Fullstack Integration:** Bridging complex Python backend logic with a JavaScript frontend, handling CORS, and rendering dynamic Canvas visualizations for bounding boxes.

---

## 💻 Installation & Setup

To run this project on your local machine:

```bash
# 1. Clone the Repository
git clone [https://github.com/your-username/kyc-verification-system.git](https://github.com/your-username/kyc-verification-system.git)
cd kyc-verification-system

# 2. Create Virtual Environment
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the Serve
uvicorn main:app --reload