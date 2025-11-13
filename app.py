# app.py — Flask web app with FULL logic + Groq RAG

import os
from flask import Flask, request, jsonify, render_template, send_from_directory

# Import Groq RAG chain from server.py
from server import chat_engine

app = Flask(__name__)

# ------------------- Tables ------------------- #
GRADE_TABLE = """
| Grade | Grade Points | Marks Range (%) |
|-------|-------------|----------------|
| A+    | 4.00        | 85 and above   |
| A     | 3.75        | 81 - 84.99     |
| A-    | 3.50        | 77 - 80.99     |
| B+    | 3.25        | 73 - 76.99     |
| B     | 3.00        | 69 - 72.99     |
| B-    | 2.75        | 65 - 68.99     |
| C+    | 2.50        | 61 - 64.99     |
| C     | 2.25        | 57 - 60.99     |
| C-    | 2.00        | 50 - 56.99     |
| D     | 1.50        | 40 - 49.99     |
| F     | 0.00        | below 40       |
"""

PROMOTION_RULES_MARKDOWN = """
### Promotion / ATKT Eligibility Rules

| Allowed | Not Allowed |
|---------|-------------|
| 2F, 0D  | 3F, 0D      |
| 1F, 2D  | 2F, ≥1D     |
| 0F, 1D  | 1F, ≥3D     |
| 0F, 2D  | ≥3F, ≥1D    |
| 0F, 3D  | 4F, 0D      |

**Notes:**  
If in "Not Allowed", the student must clear pending courses.
"""


# ------------------- CHAT ------------------- #
@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.json
    user_q = data.get("input", "").lower().strip()

    if not user_q:
        return jsonify({"answer": "Please ask a question."})

    # Greetings
    import re


    if re.search(r'\b(hi|hello|hey)\b', user_q):
        return jsonify({"answer": "Hello! How can I assist you with the NMIMS SRB today?"})


    if "thank" in user_q:
        return jsonify({"answer": "You're very welcome! 😊"})

    # Grading system
    if any(k in user_q for k in ["grade", "grading", "cgpa", "marks"]):
        return jsonify({"answer": GRADE_TABLE})

    # ATKT / promotion
    if any(k in user_q for k in ["atkt", "promotion", "progression", "backlog","promoted"]):
        return jsonify({"answer": PROMOTION_RULES_MARKDOWN})

    # Query expansion for common academic terms
    if "project" in user_q or "guideline" in user_q:
        user_q += " academic project submission plagiarism rules"

    # Forms
    forms = {
        "migration": "migration_certificate.doc",
        "exchange": "exchange_program_form.doc",
        "clearance": "clearance_certificate.doc",
        "undertaking": "undertaking_form.doc",
        "absence": "absence_form.doc"
    }

    for key, file in forms.items():
        if key in user_q:
            path = os.path.join("forms", file)
            if os.path.exists(path):
                return jsonify({
                    "answer": f"Here is your **{key.title()} Form**:",
                    "download_link": f"/download_form/{key}"
                })
            return jsonify({"answer": f"{key.title()} form is missing."})

    # Query expansion for placement
    if "placement" in user_q or "internship" in user_q:
        user_q += " placement rules internship eligibility process"

    # ----- MAIN GROQ RAG RESPONSE -----
    result = chat_engine.invoke({"input": user_q})
    answer = result.get("answer", "").strip()

    if not answer:
        answer = "I’m sorry, the Student Resource Book does not provide that information."

    return jsonify({"answer": answer})


# ------------------- DOWNLOAD ------------------- #
@app.route('/download_form/<form_name>')
def download_form(form_name):
    mapping = {
        "migration": "migration_certificate.doc",
        "exchange": "exchange_program_form.doc",
        "clearance": "clearance_certificate.doc",
        "undertaking": "undertaking_form.doc",
        "absence": "absence_form.doc"
    }

    filename = mapping.get(form_name.lower())
    if not filename:
        return "Form not found.", 404

    return send_from_directory("forms", filename, as_attachment=True)



# ------------------- FRONTEND ------------------- #
@app.route('/')
def home():
    return render_template("index.html")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

