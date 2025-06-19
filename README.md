# 🔥 JurusanFinder – AI Major Recommendation System

<div align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW5mYzVjOGx0dWx5dWY0Z3RlZ3B6eGJjbnBqbnR4dGZ6aDZ5eWZ0eSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPEqDGUULpEU0aQ/giphy.gif" width="150">
  <p><em>Your smart academic advisor powered by Decision Tree AI</em></p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Flask-2.0-000000?style=for-the-badge&logo=flask&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white">
</div>

---

## 🌟 **Key Features**

<div class="features">
  <div class="feature">
    <h3>🧠 AI-Powered Analysis</h3>
    <p>Decision Tree algorithm with 85%+ accuracy</p>
  </div>
  <div class="feature">
    <h3>📊 Multi-Dimensional Scoring</h3>
    <p>Evaluates personality, skills, and interests</p>
  </div>
  <div class="feature">
    <h3>🎯 Personalized Results</h3>
    <p>Top 3 major recommendations with match scores</p>
  </div>
</div>

---

## 🖥 **Screenshots**

<div align="center">
  <img src="images/home.png" width="45%" alt="Homepage">
  <img src="images/form.png" width="45%" alt="Input Form">
  <br>
  <img src="images/hasil.png" width="45%" alt="Results">
  <img src="images/chart.png" width="45%" alt="Analytics">
</div>

---

## ⚙️ **System Architecture**

```mermaid
graph TD
    A[User Interface] --> B[Flask Server]
    B --> C[Input Processing]
    C --> D[Decision Tree Model]
    D --> E[Data Analysis]
    E --> F[Recommendation Engine]
    F --> G[Visualization]
    G --> H[Results Display]