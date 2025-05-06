import asyncio
import nest_asyncio
import sys
import streamlit as st
import torch
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
import requests
import re

# Apply patch for nested asyncio loops (Streamlit compatibility)
nest_asyncio.apply()

# Optional: Windows-specific asyncio fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Cache model loading (only runs once per session)
@st.cache_resource(show_spinner="🔄 Loading ML models...")
def load_models():
    summarizer = pipeline("summarization", model="t5-small")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summarizer.model.to(device)
    sentiment_analyzer.model.to(device)
    return summarizer, sentiment_analyzer

st.title("🔗 Ecommerce Product Scraper & Analyzer")

url = st.text_input("Enter a URL to summarize and analyze sentiment", "")

def get_soup(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        raise RuntimeError(f"Requests failed: {e}")

def clean_price(value):
    clean_value = re.sub(r"^[^\d₹]*|\s*[^\d₹.]+$", "", value).strip()
    return clean_value if "₹" in clean_value else clean_value

def extract_amazon_data(soup):
    extracted_info = {}
    title_tag = soup.find("span", {"id": "productTitle"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(separator=" ", strip=True)
    price_tag = soup.find("span", {'class': 'a-price-whole'}) or soup.find("span", {"id": "priceblock_dealprice"})
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
        features = [li.get_text(separator=" ", strip=True) for li in bullet_points_tag.find_all("li")]
        extracted_info['Key Features'] = "\n".join(features)
    rating_tag = soup.find("span", {"class": "a-icon-alt"})
    reviews_tag = soup.find("span", {"id": "acrCustomerReviewText"})
    if rating_tag and reviews_tag:
        extracted_info['Rating'] = rating_tag.get_text(separator=" ", strip=True)
        extracted_info['Reviews'] = reviews_tag.get_text(separator=" ", strip=True)
    return extracted_info

def extract_flipkart_data(soup):
    extracted_info = {}
    title_tag = soup.find("span", {"class": "B_NuCI"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(separator=" ", strip=True)
    price_tag = soup.find("div", {"class": "Nx9bqj CxhGGd yKS4la"})
    if price_tag:
        extracted_info['Price'] = price_tag.get_text(separator=" ", strip=True)
    mrp_tag = soup.find("div", {"class": "yRaY8j A6+E6v yKS4la"})
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
    discount_tag = soup.find("div", {"class": "UkUFwK WW8yVX yKS4la"})
    if discount_tag:
        extracted_info['Discount'] = discount_tag.get_text(separator=" ", strip=True)
    description_tag = soup.find("div", {"class": "_1mXcCf"})
    if description_tag:
        extracted_info['Description'] = description_tag.get_text(separator=" ", strip=True)
    bullet_points_tag = soup.find("ul", {"class": "_1xgFaf"})
    if bullet_points_tag:
        features = [li.get_text(separator=" ", strip=True) for li in bullet_points_tag.find_all("li")]
        extracted_info['Key Features'] = "\n".join(features)
    rating_tag = soup.find("div", {"class": "_3LWZlK"})
    reviews_tag = soup.find("span", {"class": "_2_R_DZ"})
    if rating_tag and reviews_tag:
        extracted_info['Rating'] = rating_tag.get_text(separator=" ", strip=True)
        extracted_info['Reviews'] = reviews_tag.get_text(separator=" ", strip=True)
    return extracted_info

if url:
    try:
        st.info("📦 Fetching content...")

        # Get the soup using requests instead of Selenium
        soup = get_soup(url)

        # Load models after URL input to avoid slow startup
        summarizer, sentiment_analyzer = load_models()

        if "amazon" in url.lower():
            extracted_info = extract_amazon_data(soup)
        elif "flipkart" in url.lower():
            extracted_info = extract_flipkart_data(soup)
        else:
            st.warning("⚠️ Only Amazon and Flipkart product pages are supported.")
            st.stop()

        st.write("📑 **Extracted Info:**")
        st.write(extracted_info)

        main_content = "\n".join([f"{key}: {value}" for key, value in extracted_info.items()])
        clean_text = " ".join(main_content.split())
        max_input_length = 512
        truncated_text = clean_text[:max_input_length]

        with st.expander("🔍 Show Extracted Text"):
            st.write(truncated_text)

        st.info("✏️ Summarizing product information...")
        summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        st.subheader("📝 Summary")
        st.write(summary)

        st.info("🔍 Analyzing sentiment...")
        sentiment = sentiment_analyzer(summary)[0]
        st.subheader("💬 Sentiment Analysis")
        st.write(f"Sentiment: `{sentiment['label']}` with score `{sentiment['score']:.2f}`")

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
        st.error(f"❌ An error occurred: {e}")
