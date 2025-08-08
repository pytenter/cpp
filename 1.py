import os
import re
import tempfile
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import hashlib

os.makedirs("static", exist_ok=True)

#步骤一：修改requirements文件
#步骤二：重新构建 Docker 镜像
#必须重新构建，才能让容器安装新依赖：
#在终端输入以下代码
#cd D:\cpp
#docker build -t pdf-qa .
#步骤三：重新运行容器
#构建成功后再次运行：
#docker run -p 8501:8501 pdf-qa
#然后访问
#http://localhost:8501

#deepseek的API密钥
#sk-d7e0f2023a7b498c8d0381fa04e85298

# --- START OF CONFIGURATION AND TEXT/TRANSLATION SECTION ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


LANG = "en"

TEXT = {
    "en": {
        "title": "📚 PDF Question Answering System (LangChain Based)",
        "api_input": "OpenAI API Key",
        "api_set": "API key set successfully",
        "upload": "Upload PDF files (multiple supported)",
        "extracting": "Extracting and splitting PDF text...",
        "processed": "Processed {files} file(s), {chunks} text chunks",
        "indexing": "Building vector index...",
        "enter_question": "Enter your question:",
        "retrieving": "Retrieving and generating answer...",
        "answer": "Answer:",
        "sources": "Sources:",
        "accurate": "✅ Accurate",
        "inaccurate": "❌ Inaccurate",
        "no_api": "Please enter your OpenAI API key first",
        "show_highlights": "Show highlighted preview",
        "download_highlighted": "Download highlighted PDF",
        "highlight_preview": "Highlighted Preview (Page {page_num})",
        "history_header": "📜 History",
        "clear_history": "Clear History"


},
    "zh": {
        "title": "📚 基于LangChain的PDF问答系统",
        "api_input": "OpenAI API密钥",
        "api_set": "API密钥已设置",
        "upload": "上传PDF文件（支持多个）",
        "extracting": "提取文本并分块...",
        "processed": "已处理 {files} 个文件，{chunks} 个文本块",
        "indexing": "构建向量索引...",
        "enter_question": "请输入问题：",
        "retrieving": "检索并生成答案...",
        "answer": "回答：",
        "sources": "参考来源：",
        "accurate": "✅ 准确",
        "inaccurate": "❌ 不准确",
        "no_api": "请先输入OpenAI API密钥",
        "show_highlights": "显示高亮预览",
        "download_highlighted": "下载高亮版PDF",
        "highlight_preview": "高亮预览（第 {page_num} 页）",
        "history_header": "📜 历史记录",
        "clear_history": "清除历史记录",
        "preview_page": "Preview Page {page_num}"

    }
}
# --- END OF CONFIGURATION AND TEXT/TRANSLATION SECTION ---
import re

def expand_query_with_llm(query):
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )
    prompt = (
        f"请将下面这个提问，改写成更容易从技术文档中找到答案的问题，保留原意并补全关键词：{query}"
    )
    return llm.invoke(prompt).content



def clean_markdown(text):
    """
    Remove markdown/LaTeX formatting from text (like **bold**, $math$, \(...\), etc.)
    """
    # Remove **bold**
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Remove inline code like this
    text = re.sub(r'(.*?)', r'\1', text)

    # Remove LaTeX math \( ... \) or \[ ... \]
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
    text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)

    # Remove $...$
    text = re.sub(r'\$(.*?)\$', r'\1', text)

    # Replace <br> and <br/> with newlines
    text = re.sub(r'<br\s*/?>', '\n', text)

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    return text.strip()


# --- START OF CORE FUNCTIONS ---
def clean_text(text):
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'(\s)+', r'\1', text)
    text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)
    return text.strip()


def find_chapter_for_page(toc, page_num):
    current_chapter = "Introduction"
    for level, title, page in reversed(toc):
        if page <= page_num:
            current_chapter = title
            break
    return current_chapter


def load_and_split_pdf(pdf_files):
    documents = []
    st.session_state.pdf_bytes = {}

    for file in pdf_files:
        st.session_state.pdf_bytes[file.name] = file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(st.session_state.pdf_bytes[file.name])
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        doc = fitz.open(tmp_path)
        toc = doc.get_toc()
        for page_doc in docs:
            page_doc.page_content = clean_text(page_doc.page_content)
            page_num = page_doc.metadata["page"] + 1
            page_doc.metadata["chapter"] = find_chapter_for_page(toc, page_num)
            page_doc.metadata["source"] = file.name
        documents.extend(docs)
        doc.close()
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    st.session_state.toc_data = toc  # 保存目录结构

    return [doc for doc in splits if len(doc.page_content) > 50]


def calculate_similarity(query, text):
    query_emb = similarity_model.encode([query])
    text_emb = similarity_model.encode([text])
    return float(np.dot(query_emb, text_emb.T))


def create_vector_store(splits):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)


def create_hybrid_retriever(vector_store, splits, query):
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 4
    weights = [0.5, 0.5] if any(k in query for k in ["what", "who", "when", "how many"]) else [0.6, 0.4]
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights
    )


from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

def create_qa_chain():
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following reference documents to answer the user's question:\n"
                   "1. Prefer to answer strictly based on the document content.\n"
                   "2. If the answer requires basic reasoning or summarization, you may infer it from the context.\n"
                   "3. Always cite the source in the format [Document, Chapter, Page].\n"
                   "4. If truly not found, respond with 'No relevant information found'."),
        ("human", "{context}\n\nQuestion: {question}")
    ])

    return prompt | llm  # 使用新语法替代 LLMChain




def add_highlights(pdf_bytes, docs_to_highlight):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num, content in docs_to_highlight.items():
        page = doc.load_page(page_num)
        for text in content:
            areas = page.search_for(text)
            for area in areas:
                page.add_highlight_annot(area)
    output_bytes = doc.write()
    doc.close()
    return output_bytes
# --- END OF CORE FUNCTIONS ---

def show_pdf_preview(source_file, page_num, T):
    pdf_bytes = st.session_state.pdf_bytes.get(source_file)
    if not pdf_bytes:
        st.warning("PDF not found in session.")
        return
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        st.image(pix.tobytes(), use_column_width=True, caption=T['highlight_preview'].format(page_num=page_num + 1))


# --- START OF NEW DISPLAY FUNCTION ---
def display_qa_results(item, T):
    """
    Displays the question, answer, sources, and highlighting options for a given item.
    """
    st.subheader(T["answer"])
    raw_answer = item["answer"]

    if hasattr(raw_answer, "content"):
        answer_text = raw_answer.content
    elif isinstance(raw_answer, dict) and "content" in raw_answer:
        answer_text = raw_answer["content"]
    else:
        answer_text = str(raw_answer)

    cleaned_answer = clean_markdown(answer_text)
    st.write(cleaned_answer)

    # Feedback buttons with unique keys based on the query
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(T["accurate"], key=f"acc_{item['query']}")
    with col2:
        st.button(T["inaccurate"], key=f"inacc_{item['query']}")

    st.subheader(T["sources"])

    # Display sources and highlighting options from the item's stored data
    for source_file, docs in item["docs_by_source"].items():
        st.write(f"**Document: {source_file}**")
        unique_docs = {doc.page_content: doc for doc in docs}.values()

        for doc in unique_docs:
            page_num = doc.metadata.get('page', 0)
            st.write(f"  - Chapter: {doc.metadata['chapter']} | Page: {page_num + 1}")
            st.write(f"    Excerpt: {doc.page_content[:200]}...")
            # 🔍 原文预览按钮：添加在 excerpt 下方
            # 🔍 原文预览按钮：添加在 excerpt 下方（使用内容哈希避免重复 key）
            raw_key = f"{source_file}|{page_num}|{item['query']}|{doc.page_content[:80]}"
            uniq_suffix = hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:10]
            btn_key = f"preview_{uniq_suffix}"

            st.button(
                f"🔍 {T['highlight_preview'].format(page_num=page_num + 1)}",
                key=btn_key,
                on_click=show_pdf_preview,
                kwargs={
                    "source_file": source_file,
                    "page_num": page_num,
                    "T": T
                }
            )

        with st.expander(T["show_highlights"]):
            docs_to_highlight = defaultdict(list)
            for doc in unique_docs:
                page_index = doc.metadata.get('page', 0)
                docs_to_highlight[page_index].append(doc.page_content)

            original_pdf_bytes = st.session_state.pdf_bytes.get(source_file)
            if original_pdf_bytes:
                highlighted_pdf_bytes = add_highlights(original_pdf_bytes, docs_to_highlight)
                first_page_num = min(docs_to_highlight.keys())
                with fitz.open(stream=highlighted_pdf_bytes, filetype="pdf") as pdf_doc:
                    first_page = pdf_doc.load_page(first_page_num)
                    pix = first_page.get_pixmap(dpi=150)
                    st.write(T['highlight_preview'].format(page_num=first_page_num + 1))
                    st.image(pix.tobytes(), use_column_width=True)

                st.download_button(
                    label=T["download_highlighted"],
                    data=highlighted_pdf_bytes,
                    file_name=f"highlighted_{source_file}",
                    mime="application/pdf",
                    key=f"download_{source_file}_{item['query']}"
                )
        st.divider()
# --- END OF NEW DISPLAY FUNCTION ---


def main():
    global LANG
    st.set_page_config(page_title="PDF QA System", layout="wide")

    # --- Initialize session state variables ---
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_item" not in st.session_state:
        st.session_state.active_item = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "splits" not in st.session_state:
        st.session_state.splits = None
    if "processed_files_id" not in st.session_state:
        st.session_state.processed_files_id = None
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = {}

    LANG = st.radio("🌐 Language / 语言", options=["en", "zh"], format_func=lambda x: "English" if x == "en" else "中文", horizontal=True)
    T = TEXT[LANG]
    st.title(T["title"])

    st.subheader("🔍 Full Document Keyword Search")
    keyword = st.text_input("Enter keyword to search inside all uploaded PDFs")
    if keyword and "splits" in st.session_state and st.session_state.splits:
        keyword_hits = []
        for doc in st.session_state.splits:
            if keyword.lower() in doc.page_content.lower():
                keyword_hits.append({
                    "text": doc.page_content[:300],
                    "chapter": doc.metadata.get("chapter", "Unknown"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "source": doc.metadata.get("source", "Unknown")
                })

        if keyword_hits:
            st.write(f"Found {len(keyword_hits)} results for: **{keyword}**")
            for hit in keyword_hits:
                st.write(f"📄 **{hit['source']}** | 📘 Chapter: {hit['chapter']} | 📄 Page {hit['page']}")
                st.text_area("Excerpt", hit["text"], height=100)
                st.divider()
        else:
            st.warning("No matching results found.")

    with st.sidebar:
        api_key = st.text_input(T["api_input"] + " (e.g., DeepSeek Key)", type="password")

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"  # ✅ 加上这一行
            st.success(T["api_set"])

        # --- History Section ---
        st.divider()
        st.subheader(T["history_header"])
        if st.button(T["clear_history"]):
            st.session_state.history = []
            st.session_state.active_item = None
            st.rerun()

        for i, item in enumerate(st.session_state.history):
            button_label = item["query"][:40] + "..." if len(item["query"]) > 40 else item["query"]
            if st.button(button_label, key=f"history_btn_{i}"):
                st.session_state.active_item = item
                st.rerun()

        if "toc_data" in st.session_state and st.session_state.toc_data:
            st.subheader("📑 TOC Navigator")
            for level, title, page in st.session_state.toc_data:
                indent = "    " * (level - 1)
                st.markdown(f"{indent}- Page {page}: {title}")

    uploaded_files = st.file_uploader(T["upload"], type=["pdf"], accept_multiple_files=True)
    file_identifier = "".join(sorted([f.name for f in uploaded_files])) if uploaded_files else None

    if uploaded_files and os.environ["OPENAI_API_KEY"]:
        # Process files only if they have changed
        if st.session_state.processed_files_id != file_identifier:
            with st.spinner(T["extracting"]):
                st.session_state.splits = load_and_split_pdf(uploaded_files)
                st.success(T["processed"].format(files=len(uploaded_files), chunks=len(st.session_state.splits)))
            with st.spinner(T["indexing"]):
                st.session_state.vector_store = create_vector_store(st.session_state.splits)
            st.session_state.processed_files_id = file_identifier
            st.session_state.history = []
            st.session_state.active_item = None

        query = st.text_input(T["enter_question"], key="query_input")
        expanded_query = expand_query_with_llm(query)


        if st.button("🧠 Generate Section Summaries"):
            with st.spinner("Generating summaries..."):
                summaries = []
                grouped = defaultdict(list)
                for doc in st.session_state.splits:
                    chapter = doc.metadata.get("chapter", "Unknown")
                    grouped[chapter].append(doc.page_content)

                summarizer = ChatOpenAI(
                    model_name="deepseek-chat",
                    temperature=0.3,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_base=os.getenv("OPENAI_API_BASE")
                )

                for chapter, contents in grouped.items():
                    chunk = "\n".join(contents[:3])[:1500]  # 只截取每章部分内容避免超长
                    summary_prompt = f"Summarize the following content from the chapter '{chapter}':\n\n{chunk}"
                    result = summarizer.invoke(summary_prompt)
                    summaries.append((chapter, result.content))

                for chapter, summary in summaries:
                    st.markdown(f"### 📘 {chapter}")
                    st.write(summary)


        if st.button("📝 Generate Quiz"):
            with st.spinner("Generating quiz questions..."):
                quiz_generator = ChatOpenAI(
                    model_name="deepseek-chat",
                    temperature=0.3,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_base=os.getenv("OPENAI_API_BASE")
                )

                quiz_content = "\n".join(doc.page_content for doc in st.session_state.splits[:5])[:2000]
                quiz_prompt = (
                        "Based on the following content, generate 3 multiple-choice questions. "
                        "Each question should have 4 options and indicate the correct answer:\n\n"
                        + quiz_content
                )
                result = quiz_generator.invoke(quiz_prompt)
                st.markdown("### 🧪 Quiz")
                st.write(result.content)

        # Process a new query
        is_new_query = query and (not st.session_state.active_item or query != st.session_state.active_item.get("query"))
        if is_new_query:
            with st.spinner(T["retrieving"]):
                retriever = create_hybrid_retriever(st.session_state.vector_store, st.session_state.splits, expanded_query)

                docs = retriever.get_relevant_documents(query)
                scored_docs = [(doc, calculate_similarity(query, doc.page_content)) for doc in docs]
                top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:3]
                filtered_docs = [doc for doc, _ in top_docs]

                context_docs = []
                docs_by_source = defaultdict(list)
                for doc in filtered_docs:
                    page_num = doc.metadata.get('page', 0)
                    context_docs.append(f"[Document: {doc.metadata['source']}, Chapter: {doc.metadata['chapter']}, Page: {page_num + 1}]\n{doc.page_content}")
                    docs_by_source[doc.metadata['source']].append(doc)

                context = "\n\n".join(context_docs)
                qa_chain = create_qa_chain()
                answer = qa_chain.invoke({"context": context, "question": query})


                # Create and store the new history item
                new_item = {"query": query, "answer": answer, "docs_by_source": docs_by_source}
                st.session_state.history.insert(0, new_item)
                st.session_state.active_item = new_item
                st.rerun()

        # Display the active item (either new or from history)
        if st.session_state.active_item:
            display_qa_results(st.session_state.active_item, T)

    elif uploaded_files and not os.environ["OPENAI_API_KEY"]:
        st.warning(T["no_api"])


if __name__ == "__main__":
    main()