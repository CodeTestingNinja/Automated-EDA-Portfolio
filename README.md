<div align="center">
  <img src="https://raw.githubusercontent.com/CodeTestingNinja/Automated-EDA-Portfolio/main/readme_banner.png" alt="Project Banner">
  <h1>Automated Exploratory Data Analysis Web App</h1>
  <p>
    <strong>A full-stack web application for interactive data cleaning, visualization, and reporting.</strong>
  </p>
  <p>
    Built from scratch with Django, Pandas, and modern frontend technologies.
  </p>
  <p>
    <a href="https://eda-portfolio-tool.onrender.com" target="_blank">
      <img src="https://img.shields.io/badge/Live_Demo-🚀-4A55A2?style=for-the-badge" alt="Live Demo">
    </a>
  </p>
</div>

---

## 🚀 Key Features

This project is a complete, end-to-end data analysis suite with a rich, interactive user interface.

| Feature                      | Description                                                                                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📊 **Initial Data Overview**  | Instantly view foundational statistics: Shape, `df.info()`, descriptive stats, and a missing values summary in a clean, card-based layout.                    |
| ✨ **Interactive Cleaning**  | A powerful, column-by-column interface to handle missing data and outliers, complete with a crucial "Undo" feature to promote confident experimentation.                 |
| 📈 **Rich Visualizations**   | A full suite of plots: Histograms, Box Plots, Scatter Plots, Heatmaps, and more, all generated dynamically on the backend and paired with relevant summary statistics. |
| 📄 **PDF Reporting**         | A capstone feature to download a comprehensive, professionally formatted PDF of the final analysis, including key stats and visualizations.                      |
| 🎨 **Polished UX**           | A modern, consistent design system applied across all pages, featuring a responsive, hover-to-expand sidebar with multi-level nested menus for easy navigation.     |

---

## 🛠️ Technology Stack

This project was built from the ground up, leveraging a modern technology stack to create a robust and interactive application.

| Area      | Technologies                                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Backend** | `Python`, `Django`, `Pandas`, `Matplotlib`, `Seaborn`, `Gunicorn`                                                                            |
| **Frontend**  | `HTML5`, `CSS3`, `Vanilla JavaScript`                                                                                                      |
| **Database**  | `SQLite` (for development and production session storage)                                                                                  |
| **Deployment**| `Render`, `WhiteNoise`                                                                                                                     |
| **Reporting** | `xhtml2pdf`                                                                                                                                |

---

## 📸 Project in Action

Here is a brief look at the application's clean and functional interface.

*(It is highly recommended to replace this static image with an animated GIF showing the workflow: Upload -> Analyze -> Download PDF)*

<div align="center">
  <img src="https://raw.githubusercontent.com/CodeTestingNinja/Automated-EDA-Portfolio/main/project_demo.png" alt="Project Demo Screenshot" width="800">
</div>

---

## 💡 Project Journey & Key Learnings

As someone with a background in Machine Learning and Deep Learning, I recognized that a significant amount of any data scientist's time is spent on the repetitive, yet critical, tasks of initial data cleaning and exploratory analysis. The goal of this project was to bridge my data science knowledge with full-stack web development skills by building a tool I would personally use—one that automates this workflow in an interactive and user-friendly way.

This was my **first project using the Django framework, HTML, CSS, and JavaScript**, and it represents a significant learning journey:

-   **Full-Stack Architecture:** I learned to build a complete web application, managing everything from backend URL routing and view logic in **Django** to creating structured, semantic frontend pages with **HTML**.
-   **Backend Data Processing:** The core of the application lies in its use of **Pandas** for all data manipulation. I implemented complex logic for cleaning data (handling missing values and outliers) and calculating a wide range of statistics.
-   **Dynamic Visualization:** A major challenge was generating plots on the server with **Matplotlib/Seaborn** and displaying them on the frontend. I learned how to save these plots to an in-memory buffer and encode them into Base64 strings, allowing for dynamic, data-driven image generation without saving files to disk.
-   **State Management & Deployment:** I mastered the use of **Django Sessions** for state management and overcame numerous real-world production challenges to deploy the application successfully on **Render**. This involved configuring `gunicorn`, `WhiteNoise` for static files, and debugging complex, platform-specific session and import issues.
-   **Modern UI/UX Design:** Starting with no CSS knowledge, I learned to build a complete design system from scratch. I implemented responsive layouts using **Flexbox** and **Grid**, created a professional, hover-to-expand sidebar with **JavaScript**, and meticulously polished every page to ensure a cohesive and intuitive user experience.
-   **Problem-Solving & Debugging:** This project was a masterclass in debugging. I systematically identified and fixed a wide range of issues, from simple typos to subtle production bugs that required deep analysis of server logs and deployment configurations.

This project was a journey from concept to a complete, polished, and powerful application. It has solidified my understanding of how to build data-driven tools and has given me a deep appreciation for the entire web development stack.

---

## 🚀 Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/CodeTestingNinja/Automated-EDA-Portfolio.git
    cd Automated-EDA-Portfolio
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Apply Database Migrations:**
    ```bash
    python manage.py migrate
    ```

5.  **Run the Development Server:**
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`.

---

## 🌟 Future Enhancements

While the project is complete in its current scope, there are several exciting features that could be added in the future:

-   **Dark Mode:** Implement a user-toggleable dark theme for better accessibility and user preference.
-   **More Advanced Data Cleaning:** Add tools for categorical encoding (One-Hot, Label), feature scaling, and log transformations.
-   **Interactive Plots:** Integrate a library like `Plotly` or `Bokeh` to allow users to zoom, pan, and hover over visualizations.

---

## 👨‍💻 Author

*   **Sayantan Dutta**
*   **LinkedIn:** `[Link to your LinkedIn profile]`
*   **GitHub:** `[Link to your GitHub profile]`
