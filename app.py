from flask import Flask, jsonify, request, render_template
from model import NewsDetector

app = Flask(__name__)
detector = NewsDetector("data/news.csv")

SAMPLE_ARTICLES = [
    {
        "category": "Reliable Journalism",
        "title": "Medical Journal Report",
        "badge": "Factual",
        "text": "According to a study published in the New England Journal of Medicine, researchers found that regular cardiovascular exercise reduces the risk of heart disease by 32 percent across diverse age groups.",
    },
    {
        "category": "Reliable Journalism",
        "title": "Economic Federal Policy",
        "badge": "Factual",
        "text": "The Federal Reserve announced a 25 basis point interest rate adjustment following the Federal Open Market Committee meeting, citing moderation in core inflation figures and sustained labor market stability.",
    },
    {
        "category": "Fake / Clickbait",
        "title": "Miracle Health Secret",
        "badge": "Sensational",
        "text": "SHOCKING: You will never believe what doctors just admitted! This simple kitchen spice cures diabetes and cancer in 48 hours and Big Pharma is desperately trying to hide the secret from you!",
    },
    {
        "category": "Conspiracy Disinformation",
        "title": "Suppressed Coverup Claim",
        "badge": "Conspiracy",
        "text": "EXCLUSIVE: Declassified military files prove that the moon landings were completely filmed on a Hollywood sound stage to trick rival nations, and secret elites are covering up the truth!",
    },
    {
        "category": "Financial Scam / Phishing",
        "title": "International Lottery Prize",
        "badge": "Scam Alert",
        "text": "CONGRATULATIONS! You have been selected as the official lucky winner of $1,000,000 in our international lottery! Transfer $250 processing fee immediately to claim your prize now.",
    },
    {
        "category": "Account Phishing Alert",
        "title": "Bank Security Notice",
        "badge": "Phishing Alert",
        "text": "URGENT NOTICE: Your bank account has been flagged for suspicious activity. Click here right now and verify your online banking password and OTP to prevent immediate account termination.",
    },
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "status": "Ready",
        "model_type": "TF-IDF + Calibrated Linear Support Vector Classifier",
        "total_samples": detector.total_samples,
        "metrics": detector.metrics,
    })


@app.route("/api/samples", methods=["GET"])
def sample_data():
    return jsonify(SAMPLE_ARTICLES)


@app.route("/api/analyze", methods=["POST"])
def analyze_article():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some content to analyze."}), 400

    if len(text) < 10:
        return jsonify({"error": "Text is too short. Please provide at least 10 characters."}), 400

    try:
        result = detector.predict_article(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/history", methods=["GET"])
def history():
    # Privacy protection: History is maintained per-browser via localStorage.
    return jsonify({"count": 0, "items": []})


@app.route("/api/history/<path:analysis_id>", methods=["GET", "DELETE"])
def history_record_noop(analysis_id):
    return jsonify({"success": True})


@app.route("/api/history/clear", methods=["DELETE"])
def clear_all_history_noop():
    return jsonify({"success": True})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
