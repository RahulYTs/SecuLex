import os
import random
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import requests
from bs4 import BeautifulSoup
import trafilatura
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# User agents to rotate for avoiding detection
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]


def get_random_user_agent():
    """Return a random user agent from the list."""
    return random.choice(USER_AGENTS)


def search_web(query, num_results=8):
    """
    Search the web for the given query and return text from relevant websites.
    
    Args:
        query (str): The search query
        num_results (int): Number of websites to crawl (5-10 recommended)
        
    Returns:
        str: Concatenated text from all crawled websites
    """
    try:
        logger.debug(f"Starting web search for query: {query}")

        # Fallback text in case of network issues
        fallback_text = f"""
        I'm currently experiencing network connectivity issues and couldn't search the web for information about "{query}".
        
        This could be due to network restrictions, firewall settings, or temporary connectivity problems.
        
        I'm still able to learn from our conversations, and I'll store your questions to build my knowledge base.
        
        You can try:
        1. Checking your internet connection
        2. Asking a more specific question
        3. Trying again later
        """

        try:
            # Try to get search results with timeout
            search_urls = get_search_results(query, num_results)
            logger.debug(f"Found {len(search_urls)} search results")

            if not search_urls:
                return "No search results found for the query. Please try a different search term or check your internet connection."

            # Crawl websites in parallel
            all_text = crawl_websites(search_urls)

            if not all_text or len(all_text) < 100:
                logger.warning("No significant text extracted from search results")
                return fallback_text

            return all_text

        except requests.exceptions.RequestException as req_err:
            logger.error(f"Network error during web search: {str(req_err)}")
            return fallback_text

    except Exception as e:
        logger.error(f"Error in search_web: {str(e)}", exc_info=True)
        return f"Error searching the web: {str(e)}"


def get_search_results(query, num_results=8):
    """
    Get search result URLs for the given query.
    
    This function implements a basic web search without using any rate-limited API.
    It sends a request to DuckDuckGo and parses the HTML response.
    """
    try:
        # Format the query for the URL
        encoded_query = urllib.parse.quote(query)

        # URLs to try (fallbacks in case one gets blocked)
        search_engines = [
            f"https://html.duckduckgo.com/html/?q={encoded_query}",
            f"https://search.brave.com/search?q={encoded_query}",
            f"https://www.mojeek.com/search?q={encoded_query}"
        ]

        all_urls = []

        # Try each search engine until we get results
        for search_url in search_engines:
            try:
                headers = {
                    'User-Agent': get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                }

                response = requests.get(search_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Different selectors for different search engines
                    selectors = [
                        # DuckDuckGo
                        ('a.result__a', 'href'),
                        # Brave
                        ('a.snippet-title', 'href'),
                        # Mojeek
                        ('a.title', 'href')
                    ]

                    for selector, attr in selectors:
                        links = soup.select(selector)
                        if links:
                            for link in links:
                                url = link.get(attr)

                                # Process URLs based on search engine
                                if isinstance(search_url, str) and search_url.startswith('https://html.duckduckgo.com'):
                                    # DuckDuckGo uses redirects
                                    if isinstance(url, str) and url.startswith('/'):
                                        continue

                                    # Extract actual URL from DuckDuckGo redirect
                                    if isinstance(url, str) and 'uddg=' in url:
                                        url = urllib.parse.unquote(url.split('uddg=')[1].split('&')[0])

                                # Skip unwanted URLs
                                if is_valid_url(url):
                                    all_urls.append(url)

                            # If we found links, break the selector loop
                            break

                    # If we found links, break the search engine loop
                    if all_urls:
                        break

            except Exception as e:
                logger.warning(f"Error with search engine {search_url}: {str(e)}")
                continue

        # Take only unique URLs up to num_results
        unique_urls = []
        for url in all_urls:
            if url not in unique_urls:
                unique_urls.append(url)
                if len(unique_urls) >= num_results:
                    break

        return unique_urls

    except Exception as e:
        logger.error(f"Error in get_search_results: {str(e)}", exc_info=True)
        return []


def is_valid_url(url):
    """Check if a URL is valid and not a blacklisted domain."""
    if not url or not isinstance(url, str):
        return False

    # Skip URLs that don't start with http
    if not url.startswith(('http://', 'https://')):
        return False

    # Blacklist of domains to skip
    blacklist = [
        'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'pinterest.com', 'reddit.com', 'tiktok.com',
        'amazon.com', 'ebay.com', 'netflix.com', 'spotify.com',
        'apple.com', 'microsoft.com', 'login', 'signin', 'account'
    ]

    # Check if URL contains any blacklisted domain
    for domain in blacklist:
        if domain in url.lower():
            return False

    return True


def extract_text_from_url(url):
    """
    Extract clean text from a URL using trafilatura or BeautifulSoup as fallback.
    """
    try:
        headers = {'User-Agent': get_random_user_agent()}

        # Add a random delay to avoid looking like a bot
        time.sleep(random.uniform(0.5, 2.0))

        # Try with Trafilatura first (better quality extraction)
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text) > 200:  # Ensure we got meaningful content
                return f"From {url}:\n\n{text}\n\n"

        # Fallback to BeautifulSoup if trafilatura didn't work
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()

            # Get text and clean it up
            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            if text and len(text) > 200:  # Ensure we got meaningful content
                return f"From {url}:\n\n{text}\n\n"

        return ""

    except Exception as e:
        logger.warning(f"Error extracting text from {url}: {str(e)}")
        return ""


def crawl_websites(urls):
    """
    Crawl multiple websites in parallel and combine their text.
    
    Args:
        urls (list): List of URLs to crawl
        
    Returns:
        str: Combined text from all websites
    """
    all_text = []

    # Use ThreadPoolExecutor for parallel crawling
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_url = {executor.submit(extract_text_from_url, url): url for url in urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    all_text.append(text)
                    logger.debug(f"Successfully extracted text from {url} ({len(text)} chars)")
                else:
                    logger.debug(f"No usable text extracted from {url}")
            except Exception as e:
                logger.warning(f"Error processing {url}: {str(e)}")

    if not all_text:
        return "Could not extract useful information from the search results."

    # Combine all texts
    return "\n\n".join(all_text)
