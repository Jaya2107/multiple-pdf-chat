
"""Playwright UI tests for the Multi-PDF Chatbot (Streamlit app)."""
import os

import pytest
from playwright.sync_api import Page, expect

SLOW = 30_000
VERY_SLOW = 180_000


def open_app(page: Page, url: str):
    page.goto(url)
    expect(page.get_by_role("heading", name="Multi-PDF Chatbot")).to_be_visible(timeout=SLOW)


def upload_and_process(page: Page, pdf_path):
    page.locator('input[type="file"]').set_input_files(str(pdf_path))
    expect(page.get_by_text("sample.pdf").first).to_be_visible(timeout=SLOW)
    page.get_by_role("button", name="Process PDFs").click()
    expect(page.get_by_text("Ready to chat!")).to_be_visible(timeout=VERY_SLOW)


# ---------- Smoke tests (no LLM needed) ----------

def test_page_loads_with_title_and_caption(page: Page, app_url):
    open_app(page, app_url)
    expect(page.get_by_text("Upload one or more PDFs")).to_be_visible()


def test_file_uploader_and_button_are_present(page: Page, app_url):
    open_app(page, app_url)
    expect(page.get_by_text("Upload PDF(s)")).to_be_visible()
    expect(page.get_by_role("button", name="Process PDFs")).to_be_visible()


def test_both_rag_modes_are_available(page: Page, app_url):
    open_app(page, app_url)
    expect(page.get_by_text("Standard RAG", exact=True)).to_be_visible()
    expect(page.get_by_text("Self-Correcting RAG (LangGraph)")).to_be_visible()


def test_chat_input_hidden_before_processing(page: Page, app_url):
    open_app(page, app_url)
    expect(page.get_by_placeholder("Ask something from the PDF(s)")).to_have_count(0)


# ---------- Error handling ----------

def test_process_without_pdf_shows_warning(page: Page, app_url):
    open_app(page, app_url)
    page.get_by_role("button", name="Process PDFs").click()
    expect(page.get_by_text("Please upload PDF(s) first.")).to_be_visible(timeout=SLOW)


# ---------- Core flow (needs embedding model, not the LLM) ----------

def test_upload_and_process_pdf_enables_chat(page: Page, app_url, sample_pdf):
    open_app(page, app_url)
    upload_and_process(page, sample_pdf)
    expect(page.get_by_placeholder("Ask something from the PDF(s)")).to_be_visible(timeout=SLOW)


# ---------- End-to-end answer quality (needs real LLM) ----------

llm = pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1",
    reason="Set RUN_LLM_TESTS=1 (and have Groq key / Ollama running) to run",
)


@llm
@pytest.mark.llm
def test_answer_comes_from_document(page: Page, app_url, sample_pdf):
    open_app(page, app_url)
    upload_and_process(page, sample_pdf)
    page.get_by_placeholder("Ask something from the PDF(s)").fill("What is the capital of Zorland?")
    page.keyboard.press("Enter")
    expect(page.get_by_text("Quillon").first).to_be_visible(timeout=VERY_SLOW)


@llm
@pytest.mark.llm
def test_out_of_scope_question_is_refused(page: Page, app_url, sample_pdf):
    open_app(page, app_url)
    upload_and_process(page, sample_pdf)
    page.get_by_placeholder("Ask something from the PDF(s)").fill("Who won the 2018 FIFA World Cup?")
    page.keyboard.press("Enter")
    expect(page.get_by_text("Not in the provided documents").first).to_be_visible(timeout=VERY_SLOW)