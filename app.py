import os
import re
from math import log
import fitz  # PyMuPDF
import google.generativeai as genai
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    
    /* Custom Scrollbar for Previews */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 4px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def split_into_chunks(text: str, page_number: int, chunk_size=180, overlap=40):
    """Split page text into smaller overlapping sections with basic word cleanup."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append({"text": chunk, "page": page_number})

        if start + chunk_size >= len(words):
            break

    return chunks


def find_relevant_passages(question: str, chunks: list, top_n: int = 3):
    """
    Ranks passages using TF-IDF term scoring to give higher weight to rare, unique terms.
    """
    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "for",
        "and",
        "or",
        "on",
        "does",
        "do",
        "how",
        "why",
        "with",
        "about",
        "explain",
        "me",
        "this",
        "that",
        "can",
        "you",
        "from",
        "it",
        "by",
        "at",
    }

    question_words = [
        w
        for w in re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
        if w not in stop_words
    ]

    if not question_words or not chunks:
        return []

    # Calculate Document Frequency (DF) for question terms across chunks
    total_chunks = len(chunks)
    doc_freq = {}
    for word in set(question_words):
        df = sum(
            1
            for chunk in chunks
            if word in re.findall(r"\b[a-zA-Z0-9]+\b", chunk["text"].lower())
        )
        doc_freq[word] = df

    ranked = []
    for chunk in chunks:
        passage_words = re.findall(r"\b[a-zA-Z0-9]+\b", chunk["text"].lower())
        passage_set = set(passage_words)
        matched_words = set(question_words) & passage_set

        if not matched_words:
            continue

        # Score based on Term Frequency (TF) * Inverse Document Frequency (IDF)
        score = 0.0
        for word in matched_words:
            tf = passage_words.count(word) / max(len(passage_words), 1)
            idf = log((total_chunks + 1) / (doc_freq.get(word, 0) + 1)) + 1
            score += tf * idf

        ranked.append({**chunk, "score": score, "matched": matched_words})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_n]


def generate_ai_answer(question: str, context_passages: list):
    """Generate an answer using Google Gemini based strictly on context."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "⚠️ **API Key Missing**: Please configure your `GEMINI_API_KEY` "
            "environment variable or Streamlit secrets to generate AI answers."
        )

    genai.configure(api_key=api_key)

    context_str = "\n\n".join(
        [
            f"[Page {p['page']}]: {p['text']}"
            for p in context_passages
        ]
    )

    prompt = f"""
    Answer the user's question directly and concisely.
Use only information supported by the provided excerpts.
Do not treat metadata, extraction labels, or category tags
as factual answers unless they are relevant to the question.
If the excerpts do not contain enough evidence, say so.
Cite page numbers for supported claims.

    --- CONTEXT ---
    {context_str}

    --- QUESTION ---
    {question}
    """

    try:
        model = genai.GenerativeModel("gemini-3.8-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"


# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])

    st.markdown("---")
    st.markdown("### Chunking Settings")
    chunk_size = st.slider(
        "Chunk Size (words)", 100, 400, 180, step=20
    )
    chunk_overlap = st.slider(
        "Overlap (words)", 10, 100, 40, step=10
    )

    st.markdown("---")
    st.markdown("💡 **Tip**: Higher overlap preserves context across boundaries.")


# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("📄 AI Document Assistant")
st.markdown(
    "Upload a document in the sidebar to extract text, search relevant excerpts, and generate AI-powered answers."
)

if uploaded_file is None:
    st.info("👈 Please upload a PDF document from the sidebar to get started.")
else:
    # Process PDF and cache in session state to avoid re-extraction on interactions
    file_bytes = uploaded_file.getvalue()
    cache_key = f"{uploaded_file.name}_{uploaded_file.size}_{chunk_size}_{chunk_overlap}"

    if "doc_cache_key" not in st.session_state or st.session_state["doc_cache_key"] != cache_key:
        with st.spinner("Parsing PDF document..."):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            all_chunks = []
            full_text = []

            for page_index, page in enumerate(doc):
                page_text = page.get_text()
                full_text.append(page_text)
                all_chunks.extend(
                    split_into_chunks(page_text, page_index + 1, chunk_size, chunk_overlap)
                )

            st.session_state["doc_cache_key"] = cache_key
            st.session_state["page_count"] = len(doc)
            st.session_state["all_chunks"] = all_chunks
            st.session_state["full_text"] = "\n".join(full_text)

    # Top Overview Metrics Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-value">{st.session_state['page_count']}</div><div class="metric-label">Total Pages</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-value">{len(st.session_state['all_chunks'])}</div><div class="metric-label">Text Chunks</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        word_count = len(st.session_state["full_text"].split())
        st.markdown(
            f"""<div class="metric-card"><div class="metric-value">{word_count:,}</div><div class="metric-label">Total Words</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # Empty document warning check
    if not st.session_state["full_text"].strip():
        st.error(
            "⚠️ No readable text extracted. This PDF might contain scanned images or require OCR."
        )
    else:
        # App Tabs for better visual grouping
        tab_qa, tab_preview = st.tabs(["💬 Q&A Assistant", "📄 Document Preview"])

        with tab_qa:
            st.subheader("Ask a Question")
            question = st.text_input(
                "Query",
                placeholder="e.g., What are the main objectives of this project?",
                label_visibility="collapsed",
            )

            col_btn, _ = st.columns([1, 4])
            with col_btn:
                search_clicked = st.button("🔍 Search & Answer", type="primary", use_container_width=True)

            if search_clicked or question:
                if not question.strip():
                    st.warning("Please enter a question to search.")
                else:
                    results = find_relevant_passages(
                        question, st.session_state["all_chunks"]
                    )

                    if results:
                        # 1. AI Generated Answer Section
                        st.markdown("### 🤖 AI Summary Answer")
                        with st.spinner("Generating answer with Gemini..."):
                            ai_response = generate_ai_answer(question, results)
                            st.info(ai_response)

                        # 2. Source Context Passages Section
                        st.markdown("### 📚 Source Passages")
                        st.caption("Top matching excerpts retrieved from your document:")

                        for idx, res in enumerate(results, 1):
                            with st.container(border=True):
                                col_p1, col_p2 = st.columns([3, 1])
                                with col_p1:
                                    st.markdown(f"**Passage {idx}** (Page {res['page']})")
                                with col_p2:
                                    st.caption(f"Score: `{res['score']:.3f}`")

                                st.write(res["text"])
                                st.caption(
                                    "**Matched keywords:** "
                                    + ", ".join(sorted(res["matched"]))
                                )
                    else:
                        st.warning(
                            "No matching passages found. Try rephrasing your question or using different keywords."
                        )

        with tab_preview:
            st.subheader("Raw Extracted Text")
            st.text_area(
                "Document Content",
                st.session_state["full_text"],
                height=350,
                label_visibility="collapsed",
            )

            st.download_button(
                label="📥 Download Extracted Text (.txt)",
                data=st.session_state["full_text"],
                file_name=f"{uploaded_file.name}_extracted.txt",
                mime="text/plain",
            )