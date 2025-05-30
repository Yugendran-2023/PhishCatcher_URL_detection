import re
import socket
import tldextract
import numpy as np
from urllib.parse import urlparse
from difflib import SequenceMatcher

# Popular brands for spoof detection
BRANDS = ["google", "facebook", "amazon", "netflix", "apple", "microsoft", "paypal", "instagram", "linkedin", "whatsapp", "flipkart"]

def contains_login_keyword(url):
    keywords = ["login", "signin", "account", "verify", "secure", "webscr"]
    return int(any(kw in url.lower() for kw in keywords))

def is_ip_address(domain):
    try:
        socket.inet_aton(domain)
        return True
    except socket.error:
        return False

def count_subdomains(hostname):
    return len(hostname.split(".")) - 2  # Subdomains = total parts - domain - TLD

def detect_spoofed_brand(domain):
    domain = domain.lower()
    for brand in BRANDS:
        similarity = SequenceMatcher(None, domain, brand).ratio()
        if brand in domain or similarity > 0.75:
            return 1
    return 0

def extract_features(url):
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        domain_info = tldextract.extract(url)
        domain = domain_info.domain

        features = [
            len(url),                                         # url_length
            int(parsed.scheme == "https"),                    # has_https
            url.count("."),                                   # num_dots
            url.count("-"),                                   # num_hyphens
            count_subdomains(hostname),                       # subdomain_count
            int(is_ip_address(hostname)),                     # is_ip_address
            len(parsed.query),                                # query_length
            contains_login_keyword(url),                      # contains_login_keyword
            detect_spoofed_brand(domain),                     # spoofed_brand_flag
        ]
        return np.array(features)
    except Exception as e:
        print(f"Error extracting features from URL {url}: {e}")
        return np.zeros(9)



