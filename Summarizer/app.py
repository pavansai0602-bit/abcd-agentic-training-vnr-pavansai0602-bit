"""
Smart Notes Summarizer - Flask Backend
Uses pure Python NLP (no external AI API or NLTK required)
"""

from flask import Flask, render_template, request, jsonify
import re
import string
from collections import Counter
import math

app = Flask(__name__)

# ── NLP Utilities ────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","get","got","had","hadn't","has","hasn't",
    "have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's","i","i'd",
    "i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
    "itself","let's","me","more","most","mustn't","my","myself","no","nor",
    "not","of","off","on","once","only","or","other","ought","our","ours",
    "ourselves","out","over","own","same","shan't","she","she'd","she'll",
    "she's","should","shouldn't","so","some","such","than","that","that's",
    "the","their","theirs","them","themselves","then","there","there's",
    "these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we",
    "we'd","we'll","we're","we've","were","weren't","what","what's","when",
    "when's","where","where's","which","while","who","who's","whom","why",
    "why's","will","with","won't","would","wouldn't","you","you'd","you'll",
    "you're","you've","your","yours","yourself","yourselves","also","just",
    "like","may","might","much","many","now","use","used","using","one",
    "two","three","however","therefore","thus","hence","whereas","although",
    "though","still","even","well","already","within","without","across",
    "among","between","against","along","around","behind","beside","beyond",
    "during","except","inside","near","outside","since","toward","upon"
}


def tokenize(text: str) -> list[str]:
    """Split text into clean word tokens."""
    text = text.lower()
    text = re.sub(r"[''`]s\b", "", text)   # possessives
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return [w.strip("-") for w in text.split() if w.strip("-")]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common abbreviations to avoid false splits
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.", r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]
    return sentences


def word_frequencies(tokens: list[str]) -> dict[str, float]:
    """Compute normalized word frequencies excluding stopwords."""
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    if not content:
        return {}
    counts = Counter(content)
    max_freq = counts.most_common(1)[0][1]
    return {word: count / max_freq for word, count in counts.items()}


def score_sentences(sentences: list[str], freq: dict[str, float]) -> list[tuple[float, int, str]]:
    """Score each sentence by summing its word frequencies."""
    scored = []
    for i, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        score = sum(freq.get(t, 0) for t in tokens if t not in STOPWORDS)
        # Normalize by sentence length to avoid bias toward long sentences
        word_count = max(len(tokens), 1)
        normalized_score = score / math.sqrt(word_count)
        scored.append((normalized_score, i, sentence))
    return scored


def extract_summary(text: str, num_points: int = 5) -> list[str]:
    """Extract the most important sentences as summary bullet points."""
    sentences = split_sentences(text)
    if not sentences:
        return ["No content to summarize."]

    if len(sentences) <= num_points:
        return [s for s in sentences if len(s.split()) > 4]

    tokens = tokenize(text)
    freq = word_frequencies(tokens)
    scored = score_sentences(sentences, freq)

    # Pick top N by score, then sort by original position for readable order
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:num_points]
    top = sorted(top, key=lambda x: x[1])  # restore reading order

    # Clean and cap each bullet
    results = []
    for _, _, sentence in top:
        sentence = sentence.strip()
        if len(sentence.split()) < 4:
            continue
        # Truncate very long sentences gracefully
        words = sentence.split()
        if len(words) > 40:
            sentence = " ".join(words[:40]) + "…"
        results.append(sentence)

    return results or ["Could not generate a meaningful summary. Try providing more detailed text."]


def extract_keywords(text: str, num_keywords: int = 10) -> list[str]:
    """Extract top keywords using TF-IDF-inspired scoring."""
    tokens = tokenize(text)
    freq = word_frequencies(tokens)

    # Filter: must be meaningful words (length > 3, not stopwords)
    candidates = {
        word: score
        for word, score in freq.items()
        if len(word) > 3 and word not in STOPWORDS
    }

    # Boost multi-character words and penalize very common short words
    boosted = {
        word: score * (1 + 0.1 * len(word))
        for word, score in candidates.items()
    }

    top = sorted(boosted, key=boosted.get, reverse=True)[:num_keywords]
    return [kw.capitalize() for kw in top]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    num_points = int(data.get("num_points", 5))
    num_keywords = int(data.get("num_keywords", 10))

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) < 40:
        return jsonify({"error": "Text is too short. Please provide more content."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text is too long (max 20,000 characters)."}), 400

    summary = extract_summary(text, num_points=num_points)
    keywords = extract_keywords(text, num_keywords=num_keywords)
    word_count = len(text.split())
    sentence_count = len(split_sentences(text))

    return jsonify({
        "summary": summary,
        "keywords": keywords,
        "stats": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "char_count": len(text),
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
