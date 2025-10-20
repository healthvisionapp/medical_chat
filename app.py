import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# --- load env ---
load_dotenv()  # loads OPENAI_API_KEY from .env if present

from openai import OpenAI
client = OpenAI()  # will read OPENAI_API_KEY from env

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are MedicalBot, a helpful medical information assistant. "
    "Answer concisely, cite common symptoms/causes/next steps when appropriate, "
    "and add: 'This is general information, not medical advice.'"
)

def call_openai(user_msg: str) -> str:
    """Call OpenAI and return assistant text (or raise)."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please type something.", 200

    try:
        answer = call_openai(msg)
        return answer, 200
    except Exception as e:
        # Print the exact error to your terminal so we can see what went wrong
        print("OPENAI ERROR:", repr(e))
        return (
            "Sorry, I couldn’t reach the medical model right now. "
            "Please try again in a moment.",
            200,
        )

if __name__ == "__main__":
    # Show whether the key is actually loaded
    print("OPENAI_API_KEY present:", bool(os.environ.get("OPENAI_API_KEY")))
    app.run(host="127.0.0.1", port=5000, debug=True)
