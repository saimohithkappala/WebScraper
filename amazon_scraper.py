import asyncio
import sys
import streamlit as st
import torch
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import re  # For regex cleaning

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def load_models():
    # Initialize pipelines
    summarizer = pipeline("summarization", model="t5-small")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Move the models to the device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summarizer.model.to(device)
    sentiment_analyzer.model.to(device)
    
    return summarizer, sentiment_analyzer

summarizer, sentiment_analyzer = load_models()

st.title("🔗 Ecommerce Product Scraper & Analyzer")

url = st.text_input("Enter a URL to summarize and analyze sentiment", "")

def get_soup_with_selenium(url):
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

# Function to clean MRP and Price values to only display the numeric value
def clean_price(value):
    # Remove any leading ellipsis or unwanted characters, then remove non-numeric characters except ₹ and dot
    clean_value = re.sub(r"^[^\d₹]*|\s*[^\d₹.]+$", "", value).strip()
    
    # Ensure that only numeric value and ₹ symbol are returned
    if "₹" in clean_value:
        return clean_value
    return clean_value

# Function to extract data from Amazon
def extract_amazon_data(soup):
    extracted_info = {}

    # Product Title
    title_tag = soup.find("span", {"id": "productTitle"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(separator=" ", strip=True)

    # Price - Amazon
    price_tag = soup.find("span", {'class': 'a-price-whole'})
    if not price_tag:
        price_tag = soup.find("span", {"id": "priceblock_dealprice"})
    if price_tag:
        currency_tag = soup.find("span", {'class': 'a-price-symbol'})
        currency = currency_tag.get_text() if currency_tag else "$"
        extracted_info['Price'] = f"{currency} {price_tag.get_text(separator=' ', strip=True)}"

    # MRP (Maximum Retail Price)
    mrp_tag = soup.find("span", class_="a-size-small aok-offscreen")
    if mrp_tag:
        extracted_info['MRP'] = clean_price(mrp_tag.get_text(separator=" ", strip=True))

    # Discount Calculation based on MRP and Price
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
            pass  # If conversion fails, no discount percentage or amount will be shown

    # Description
    description_tag = soup.find("div", {"id": "productDescription"})
    if description_tag:
        extracted_info['Description'] = description_tag.get_text(separator=" ", strip=True)

    # Key Features
    bullet_points_tag = soup.find("div", {"id": "feature-bullets"})
    if bullet_points_tag:
        features = []
        for li in bullet_points_tag.find_all("li"):
            features.append(li.get_text(separator=" ", strip=True))
        extracted_info['Key Features'] = "\n".join(features)

    # Rating and Reviews
    rating_tag = soup.find("span", {"class": "a-icon-alt"})
    reviews_tag = soup.find("span", {"id": "acrCustomerReviewText"})
    if rating_tag and reviews_tag:
        extracted_info['Rating'] = rating_tag.get_text(separator=" ", strip=True)
        extracted_info['Reviews'] = reviews_tag.get_text(separator=" ", strip=True)

    return extracted_info

# Function to extract data from Flipkart
def extract_flipkart_data(soup):
    extracted_info = {}

    # Product Title
    title_tag = soup.find("span", {"class": "B_NuCI"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(separator=" ", strip=True)

    # Price - Flipkart
    price_tag = soup.find("div", {"class": "Nx9bqj CxhGGd yKS4la"})
    if price_tag:
        extracted_info['Price'] = price_tag.get_text(separator=" ", strip=True)

    # MRP (Original price) - Flipkart
    mrp_tag = soup.find("div", {"class": "yRaY8j A6+E6v yKS4la"})
    if mrp_tag:
        extracted_info['MRP'] = clean_price(mrp_tag.get_text(separator=" ", strip=True))

    # Discount Calculation based on MRP and Price
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
            pass  # If conversion fails, no discount percentage or amount will be shown

    # Discount - Flipkart (if available)
    discount_tag = soup.find("div", {"class": "UkUFwK WW8yVX yKS4la"})
    if discount_tag:
        extracted_info['Discount'] = discount_tag.get_text(separator=" ", strip=True)

    # Description
    description_tag = soup.find("div", {"class": "_1mXcCf"})
    if description_tag:
        extracted_info['Description'] = description_tag.get_text(separator=" ", strip=True)

    # Key Features
    bullet_points_tag = soup.find("ul", {"class": "_1xgFaf"})
    if bullet_points_tag:
        features = []
        for li in bullet_points_tag.find_all("li"):
            features.append(li.get_text(separator=" ", strip=True))
        extracted_info['Key Features'] = "\n".join(features)

    # Rating and Reviews
    rating_tag = soup.find("div", {"class": "_3LWZlK"})
    reviews_tag = soup.find("span", {"class": "_2_R_DZ"})
    if rating_tag and reviews_tag:
        extracted_info['Rating'] = rating_tag.get_text(separator=" ", strip=True)
        extracted_info['Reviews'] = reviews_tag.get_text(separator=" ", strip=True)

    return extracted_info

if url:
    try:
        st.info("Fetching content with Selenium...")
        soup = get_soup_with_selenium(url)

        # Check if the URL is from Amazon or Flipkart
        if "amazon" in url.lower():
            extracted_info = extract_amazon_data(soup)
        elif "flipkart" in url.lower():
            extracted_info = extract_flipkart_data(soup)
        else:
            st.warning("The URL is neither an Amazon nor a Flipkart product page.")
            st.stop()

        # Debugging: Check the extracted_info dictionary
        st.write("Extracted Info:")
        st.write(extracted_info)

        # Display product details
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

        st.subheader("💬 Sentiment Analysis")
        st.write(f"Sentiment: {sentiment['label']}, with a score of {sentiment['score']:.2f}")

        st.subheader("🛒 Product Details")
        product_details = {
            "Attribute": ["Title", "Price", "MRP",
                        #    "Discount",
                             "Discount Percentage", "Discount Amount", "Description", "Rating", "Reviews"],
            "Value": [
                extracted_info.get("Title", "N/A"),
                extracted_info.get("Price", "N/A"),
                extracted_info.get("MRP", "N/A"),
                # extracted_info.get("Discount", "N/A"),
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
