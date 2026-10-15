# PlacementPrep AI

PlacementPrep AI is an AI-powered placement preparation assistant built
with **Python**, **Streamlit**, and the **Google Gemini API**. It helps
students prepare for technical interviews, improve their resumes, and
practice assessments in one place.

## Features

-   **AI Chat Assistant** --- Get help with placement preparation,
    programming, technical concepts, and interview questions.
-   **Resume Analyzer** --- Upload a text-based PDF resume, select a
    target role, optionally provide a job description, and receive
    strengths, skill gaps, improvement suggestions, rewritten resume
    bullets, and interview questions.
-   **Quiz & Assessment** --- Practice multiple-choice questions across
    aptitude, logical reasoning, verbal ability, OOP, DBMS/SQL,
    operating systems, computer networks, Python, Java, and data
    structures and algorithms.
-   **Explanations and scoring** --- Review quiz scores, correct
    answers, and explanations.
-   **Downloadable reports** --- Export resume analysis results as PDF
    or Markdown.
-   **Image input in chat** --- Submit supported images for relevant
    placement or study questions.

## Tech Stack

-   Python
-   Streamlit
-   Google Gemini API (`google-genai`)
-   SQLite
-   PyMuPDF
-   ReportLab
-   `python-dotenv`

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/VermaCodes2004/PlacementPrep-AI.git
cd PlacementPrep-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Create a `.env` file in the project root:

``` env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never commit `.env` or
publish your API key.

For Streamlit Community Cloud, add the key in the app's **Secrets**
settings using TOML format:

``` toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

### 5. Start the app

``` bash
python -m streamlit run app.py
```

Streamlit will show a local URL in the terminal. Open it in your
browser.

## Deployment

The app can be deployed using [Streamlit Community
Cloud](https://share.streamlit.io/):

1.  Push the project to GitHub.
2.  Create an app in Streamlit Community Cloud and select this
    repository.
3.  Set the main file path to `app.py`.
4.  Configure `GEMINI_API_KEY` in the app's Secrets settings.
5.  Deploy and open the generated `streamlit.app` URL.

Use a Python version compatible with the dependencies listed in
`requirements.txt`.

## Data and Privacy Notes

-   Keep API keys and other credentials out of source control.
-   The app uses a local SQLite database (`placementprep_history.db`)
    for chat history.
-   A local SQLite database on a cloud-hosted app should not be assumed
    to provide reliable permanent or shared storage across devices. Use
    a persistent database service if durable cross-device history is
    required.
-   Uploaded resumes and prompts may be sent to the configured Gemini
    API for analysis or responses. Do not upload documents unless you
    are comfortable sharing their contents with the configured service.

## Project Structure

A typical project layout is:

``` text
PlacementPrep-AI/
├── app.py
├── requirements.txt
├── .gitignore
├── .env                      # Local only; do not commit
└── placementprep_history.db  # Local data; do not commit
```

Additional files or folders may exist depending on your implementation.

## Troubleshooting

-   **Missing API key:** Check your local `.env` file or Streamlit Cloud
    Secrets.
-   **Dependency errors:** Confirm the required packages are listed in
    `requirements.txt` and reinstall them.
-   **Deployment errors:** Check the Streamlit Community Cloud logs for
    the exact error message.
-   **Gemini errors:** Verify that your API key is valid and that the
    configured model is available to your account.

## License

No license has been specified yet. Add a license before allowing others
to reuse or distribute this project.
