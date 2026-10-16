"""Selenium UI tests for the Multi-PDF Chatbot.

Same app, second tool. Playwright covers the deep flows;
Selenium is used here as an independent cross-check.

Run:            pytest tests/test_selenium_ui.py
See browser:    $env:SHOW_BROWSER = "1"   (PowerShell) then run pytest

Driver handling:
- Local (Windows): if <project root>/drivers/chromedriver.exe exists, it is used.
- CI (Linux): the file does not exist, so Selenium Manager finds the driver itself.
"""
import os
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

WAIT = 30  # seconds

# Optional local driver: <project root>/drivers/chromedriver.exe
DRIVER_PATH = Path(__file__).resolve().parent.parent / "drivers" / "chromedriver.exe"


@pytest.fixture
def driver():
    opts = Options()
    if os.getenv("SHOW_BROWSER") != "1":
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--no-sandbox")             # needed on CI Linux runners
    opts.add_argument("--disable-dev-shm-usage")  # avoids /dev/shm crashes on CI

    if DRIVER_PATH.exists():
        d = webdriver.Chrome(service=Service(executable_path=str(DRIVER_PATH)), options=opts)
    else:
        d = webdriver.Chrome(options=opts)  # CI/Linux: Selenium Manager finds the driver
    yield d
    d.quit()


def open_app(driver, url):
    driver.get(url)
    WebDriverWait(driver, WAIT).until(
        EC.visibility_of_element_located(
            (By.XPATH, "//h1[contains(., 'Multi-PDF Chatbot')]")
        )
    )


def test_selenium_page_loads_with_title(driver, app_url):
    open_app(driver, app_url)
    assert "Multi-PDF Chatbot" in driver.find_element(By.TAG_NAME, "h1").text


def test_selenium_both_rag_modes_visible(driver, app_url):
    open_app(driver, app_url)
    wait = WebDriverWait(driver, WAIT)
    wait.until(EC.visibility_of_element_located(
        (By.XPATH, "//p[normalize-space()='Standard RAG']")))
    wait.until(EC.visibility_of_element_located(
        (By.XPATH, "//p[contains(., 'Self-Correcting RAG (LangGraph)')]")))


def test_selenium_process_without_pdf_shows_warning(driver, app_url):
    open_app(driver, app_url)
    wait = WebDriverWait(driver, WAIT)
    button = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//button[.//p[normalize-space()='Process PDFs']]")))
    button.click()
    wait.until(EC.visibility_of_element_located(
        (By.XPATH, "//p[contains(., 'Please upload PDF(s) first.')]")))