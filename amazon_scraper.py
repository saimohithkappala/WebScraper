import asyncio
import sys
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
import torch
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import re

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def load_models():
    summarizer = pipeline("summarization", model="t5-small")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summarizer, sentiment_analyzer

    return summarizer, sentiment_analyzer

summarizer, sentiment_analyzer = load_models()

st.title("🔗 Ecommerce Product Scraper & Analyzer")

url = st.text_input("Enter a URL to summarize and analyze sentiment", "")

def get_flipkart_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup

def get_amazon_content(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    html = driver.page_source
    driver.quit()
    return BeautifulSoup(html, 'html.parser')

def clean_price(value):
    clean_value = re.sub(r"^[^\d₹]*|\s*[^\d₹.]+$", "", value).strip()
    return clean_value

def extract_amazon_data(soup):
    extracted_info = {}

    title_tag = soup.find("span", {"id": "productTitle"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(separator=" ", strip=True)

    price_tag = soup.find("span", {'class': 'a-price-whole'})
    if not price_tag:
        price_tag = soup.find("span", {"id": "priceblock_dealprice"})
    if price_tag:
        currency_tag = soup.find("span", {'class': 'a-price-symbol'})
        currency = currency_tag.get_text() if currency_tag else "$"
        extracted_info['Price'] = f"{currency} {price_tag.get_text(separator=' ', strip=True)}"

    mrp_tag = soup.find("span", class_="a-size-small aok-offscreen")
    if mrp_tag:
        extracted_info['MRP'] = clean_price(mrp_tag.get_text(separator=" ", strip=True))

    if 'Price' in extracted_info and 'MRP' in extracted_info:
        try:
            price_value = float(re.sub(r"[^\d.]", "", extracted_info['Price']))
            mrp_value = float(re.sub(r"[^\d.]", "", extracted_info['MRP']))
            if mrp_value > 0 and price_value < mrp_value:
                discount_percentage = ((mrp_value - price_value) / mrp_value) * 100
                discount_amount = mrp_value - price_value
                extracted_info['Discount Percentage'] = f"{discount_percentage:.2f}%"
                extracted_info['Discount Amount'] = f"₹ {discount_amount:.2f}"
        except ValueError:
            pass

    description_tag = soup.find("div", {"id": "productDescription"})
    if description_tag:
        extracted_info['Description'] = description_tag.get_text(separator=" ", strip=True)

    bullet_points_tag = soup.find("div", {"id": "feature-bullets"})
    if bullet_points_tag:
        features = []
        for li in bullet_points_tag.find_all("li"):
            features.append(li.get_text(separator=" ", strip=True))
        extracted_info['Key Features'] = "\n".join(features)

    rating_tag = soup.find("span", {"class": "a-icon-alt"})
    reviews_tag = soup.find("span", {"id": "acrCustomerReviewText"})
    if rating_tag and reviews_tag:
        extracted_info['Rating'] = rating_tag.get_text(separator=" ", strip=True)
        extracted_info['Reviews'] = reviews_tag.get_text(separator=" ", strip=True)

    return extracted_info

def extract_flipkart_data(soup):
    extracted_info = {}

    title_tag = soup.find("span", {"class": "VU-ZEz"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(separator=" ", strip=True)

    price = None
    price_classes = [
        "yRaY8j A6+E6v yKS4la",
        "Nx9bqj CxhGGd",
        "Nx9bqj CxhGGd yKS4la"
    ]
    for cls in price_classes:
        price_tag = soup.find("div", {"class": cls})
        if price_tag:
            price = price_tag.get_text(strip=True).replace("₹", "").replace(",", "")
            extracted_info['Price'] = f"₹ {price}"
            price = float(price)
            break

    mrp = None
    mrp_classes = [
        "yRaY8j A6+E6v",
        "yRaY8j A6+E6v yKS4la"
    ]
    for cls in mrp_classes:
        mrp_tag = soup.find_all("div", {"class": cls})
        for tag in mrp_tag:
            tag_text = tag.get_text(strip=True).replace("₹", "").replace(",", "")
            try:
                value = float(tag_text)
                if price is None or value > price:
                    mrp = value
                    extracted_info["MRP"] = f"₹{value:.2f}"
                    break
            except:
                continue
        if mrp:
            break

    if price and mrp:
        discount_amt = mrp - price
        discount_pct = (discount_amt / mrp) * 100
        extracted_info['Discount Amount'] = f"₹ {discount_amt:.2f}"
        extracted_info['Discount Percentage'] = f"{discount_pct:.2f}%"

    description_tag_amazon = soup.find("div", {"id": "productDescription"})
    if description_tag_amazon:
        extracted_info['Description'] = description_tag_amazon.get_text(separator=" ", strip=True)

    if not extracted_info.get('Description'):
        meta_description = soup.find("meta", attrs={"name": "Description"})
        if meta_description and meta_description.get("content"):
            extracted_info['Description'] = meta_description["content"].strip()

    if not extracted_info.get('Description'):
        description_tag_flipkart = soup.find("div", {"class": "_1mXcCf RmoJUa"})
        if description_tag_flipkart:
            extracted_info['Description'] = description_tag_flipkart.get_text(strip=True)

    rating_tag = soup.find("span", {"class": "a-icon-alt"})
    reviews_tag = soup.find("span", {"id": "acrCustomerReviewText"})
    if rating_tag and reviews_tag:
        extracted_info['Rating'] = rating_tag.get_text(separator=" ", strip=True)
        extracted_info['Reviews'] = reviews_tag.get_text(separator=" ", strip=True)

    if not extracted_info.get('Rating') or not extracted_info.get('Reviews'):
        rating_review_span = soup.find("span", string=lambda text: text and "Ratings" in text)
        if rating_review_span and rating_review_span.parent.name == "span":
            spans = rating_review_span.parent.find_all("span")
            if len(spans) >= 3:
                extracted_info['Rating'] = spans[0].get_text(strip=True)
                extracted_info['Reviews'] = spans[2].get_text(strip=True)

    return extracted_info

if url:
    try:
        st.info("Fetching content...")

        if "amazon" in url.lower():
            soup = get_amazon_content(url)
            extracted_info = extract_amazon_data(soup)
        elif "flipkart" in url.lower():
            soup = get_flipkart_content(url)
            extracted_info = extract_flipkart_data(soup)
        else:
            st.warning("The URL is neither an Amazon nor a Flipkart product page.")
            st.stop()

        st.write("Extracted Info:")
        st.write(extracted_info)

        main_content = "\n".join([f"{key}: {value}" for key, value in extracted_info.items()])
        clean_text = " ".join(main_content.split())
        max_input_length = 512
        truncated_text = clean_text[:max_input_length]

        with st.expander("🔍 Show extracted text"):
            st.write(truncated_text)

        st.info("Summarizing...")
        summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

        st.subheader("📝 Summary")
        st.write(summary)

        st.info("Analyzing sentiment...")
        sentiment = sentiment_analyzer(summary)[0]

        st.subheader("🛒 Product Details")
        product_details = {
            "Attribute": ["Title", "Price", "MRP", "Discount Percentage", "Discount Amount", "Description", "Rating", "Reviews"],
            "Value": [
                extracted_info.get("Title", "N/A"),
                extracted_info.get("Price", "N/A"),
                extracted_info.get("MRP", "N/A"),
                extracted_info.get("Discount Percentage", "N/A"),
                extracted_info.get("Discount Amount", "N/A"),
                extracted_info.get("Description", "N/A"),
                extracted_info.get("Rating", "N/A"),
                extracted_info.get("Reviews", "N/A"),
            ]
        }

        product_df = pd.DataFrame(product_details)
        st.dataframe(product_df, use_container_width=True)

        if 'Key Features' in extracted_info:
            st.subheader("🔑 Key Features")
            for feature in extracted_info['Key Features'].split("\n"):
                st.write(f"- {feature}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
