import requests
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import streamlit as st

def load_models():
    summarizer = pipeline("summarization", model="t5-small")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
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
    return BeautifulSoup(res.text, 'html.parser')

def get_amazon_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    return BeautifulSoup(res.text, 'html.parser')

def clean_price(value):
    return re.sub(r"[^\d.]", "", value).strip()

def extract_amazon_data(soup):
    extracted_info = {}

    # Title
    title_tag = soup.find("span", {"id": "productTitle"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(separator=" ", strip=True)

    # Price
    price_tag = soup.find("span", {'class': 'a-price-whole'})
    if not price_tag:
        price_tag = soup.find("span", {"id": "priceblock_dealprice"})
    if price_tag:
        currency_tag = soup.find("span", {'class': 'a-price-symbol'})
        currency = currency_tag.get_text() if currency_tag else "₹"
        extracted_info['Price'] = f"{currency} {price_tag.get_text(separator=' ', strip=True)}"
        try:
            price_value = float(clean_price(price_tag.get_text()))
        except:
            price_value = None
    else:
        price_value = None

    # MRP
    mrp_value = None
    all_prices = soup.find_all("span", class_="a-offscreen")
    price_values = []
    for tag in all_prices:
        try:
            val = float(clean_price(tag.get_text()))
            price_values.append(val)
        except:
            continue

    if price_values and price_value:
        mrp_candidates = [val for val in price_values if val > price_value]
        if mrp_candidates:
            mrp_value = max(mrp_candidates)
            extracted_info['MRP'] = f"₹ {mrp_value:,.2f}"

    # Discount Calculation
    if price_value and mrp_value:
        discount_amt = mrp_value - price_value
        discount_pct = (discount_amt / mrp_value) * 100
        extracted_info['Discount Amount'] = f"₹ {discount_amt:.2f}"
        extracted_info['Discount Percentage'] = f"{discount_pct:.2f}%"

    # Description
    description_tag = soup.find("div", {"id": "productDescription"}) or soup.find("div", {"id": "productDescription_feature_div"})
    if not description_tag:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            extracted_info['Description'] = meta_desc["content"].strip()
    if description_tag and not extracted_info.get('Description'):
        extracted_info['Description'] = description_tag.get_text(separator=" ", strip=True)

    # Key Features
    bullet_points_tag = soup.find("div", {"id": "feature-bullets"})
    if bullet_points_tag:
        features = [li.get_text(separator=" ", strip=True) for li in bullet_points_tag.find_all("li")]
        extracted_info['Key Features'] = "\n".join(features)

    # Ratings and Reviews
    rating_tag = soup.find("span", {"class": "a-icon-alt"})
    reviews_tag = soup.find("span", {"id": "acrCustomerReviewText"})
    if rating_tag:
        extracted_info['Rating'] = rating_tag.get_text(strip=True)
    if reviews_tag:
        extracted_info['Reviews'] = reviews_tag.get_text(strip=True)

    return extracted_info

def extract_flipkart_data(soup):
    extracted_info = {}

    title_tag = soup.find("span", {"class": "VU-ZEz"})
    if title_tag:
        extracted_info['Title'] = title_tag.get_text(strip=True)

    price = None
    price_classes = [
        "yRaY8j A6+E6v yKS4la",
        "Nx9bqj CxhGGd",
        "Nx9bqj CxhGGd yKS4la"
    ]
    for cls in price_classes:
        price_tag = soup.find("div", {"class": cls})
        if price_tag:
            price = float(price_tag.get_text(strip=True).replace("₹", "").replace(",", ""))
            extracted_info['Price'] = f"₹ {price:,.2f}"
            break

    mrp = None
    mrp_classes = [
        "yRaY8j A6+E6v",
        "yRaY8j A6+E6v yKS4la"
    ]
    for cls in mrp_classes:
        for tag in soup.find_all("div", {"class": cls}):
            try:
                val = float(tag.get_text(strip=True).replace("₹", "").replace(",", ""))
                if not mrp or val > price:
                    mrp = val
                    extracted_info["MRP"] = f"₹ {val:,.2f}"
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
        extracted_info['Rating'] = rating_tag.get_text(strip=True)
        extracted_info['Reviews'] = reviews_tag.get_text(strip=True)

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
        truncated_text = clean_text[:512]

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
        st.dataframe(pd.DataFrame(product_details), use_container_width=True)

        if 'Key Features' in extracted_info:
            st.subheader("🔑 Key Features")
            for feature in extracted_info['Key Features'].split("\n"):
                st.write(f"- {feature}")

        st.subheader("🧠 Sentiment")
        st.write(f"**Label:** {sentiment['label']}  \n**Score:** {sentiment['score']:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
