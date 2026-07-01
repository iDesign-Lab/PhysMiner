"""
Word Cloud generator for academic papers (PDF) in the current directory.
Uses PyMuPDF (fitz) for text extraction and wordcloud for visualization.
"""

import os
import re
import fitz  # PyMuPDF
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
PDF_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(PDF_DIR)), "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "wordcloud_highlyCited.png")
WIDTH, HEIGHT = 1600, 900
MAX_WORDS = 10
BACKGROUND = "white"
COLORMAP = "viridis"          # try: plasma, inferno, cool, Blues, tab10 …

# Extra domain-specific stopwords for fluid mechanics papers
EXTRA_STOPWORDS = {
    "figure", "fig", "table", "equation", "eq", "al", "et",
    "also", "using", "used", "one", "two", "three", "four", "five",
    "shown", "show", "shows", "however", "also", "results", "result",
    "see", "may", "can", "well", "within", "although", "since",
    "paper", "present", "presented", "study", "obtained",
    "section", "chapter", "ref", "reference", "references",
    "experimental", "numerical", "flow", "flows",  # keep or remove as needed
    "doi","org",  # keep or remove as needed
    "case", "cases", "value", "time", "Reynolds", "number", # keep or remove as needed
    "backward","facing", "step",   # keep or remove as needed
    "periodic","hill", "hills"   # keep or remove as needed
}


def extract_text_from_pdfs(directory: str) -> str:
    """Extract all text from every PDF found in *directory*."""
    all_text = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    for filename in pdf_files:
        path = os.path.join(directory, filename)
        print(f"  Reading: {filename}")
        try:
            doc = fitz.open(path)
            for page in doc:
                all_text.append(page.get_text())
            doc.close()
        except Exception as exc:
            print(f"  [WARNING] Could not read {filename}: {exc}")

    return "\n".join(all_text)


def clean_text(raw: str) -> str:
    """Remove numbers, single characters, and non-alphabetic tokens."""
    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", raw)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def build_stopwords() -> set:
    stopwords = set(STOPWORDS)
    stopwords.update(EXTRA_STOPWORDS)
    return stopwords


def generate_wordcloud(text: str, stopwords: set) -> None:
    wc = WordCloud(
        width=WIDTH,
        height=HEIGHT,
        background_color=BACKGROUND,
        stopwords=stopwords,
        max_words=MAX_WORDS,
        colormap=COLORMAP,
        collocations=True,       # allow bigrams
        min_word_length=3,
        prefer_horizontal=0.85,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    # ax.set_title(
    #     "Word Cloud — Academic Papers",
    #     fontsize=18, fontweight="bold", pad=14
    # )
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nWord cloud saved → {OUTPUT_FILE}")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Scanning PDFs in: {PDF_DIR}\n")
    raw_text = extract_text_from_pdfs(PDF_DIR)
    clean = clean_text(raw_text)
    total_words = len(clean.split())
    print(f"\nTotal words extracted: {total_words:,}")

    if total_words < 20:
        raise ValueError("Too few words extracted — check that the PDFs contain selectable text.")

    stopwords = build_stopwords()
    generate_wordcloud(clean, stopwords)


if __name__ == "__main__":
    main()
