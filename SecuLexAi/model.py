import os
import logging
import re
import random
from collections import Counter
import warnings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Try to download NLTK data, but handle it if there's an error
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {str(e)}")


def summarize(query, text, max_length=2048):
    """
    Summarize the given text based on the query using a smart extractive summarization.
    
    Args:
        query (str): The user's question
        text (str): The text to summarize (from web search)
        max_length (int): Maximum length of summary
        
    Returns:
        str: Summarized text
    """
    try:
        logger.debug(f"Summarizing text of length {len(text)}")

        if not text or len(text) < 100:
            return f"I couldn't find much information about '{query}'. Please try a different query."

        # Clean up the input text - remove redundant spaces, fix broken sentences
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.([A-Z])', r'. \1', text)  # Fix merged sentences

        # Remove source URLs and citation patterns commonly found in scraped text
        text = re.sub(r'From https?://[^\s]+:\s*', '', text)
        text = re.sub(r'\[citation needed\]|\[\d+\]', '', text)

        # Remove redundant newlines that often appear in scraped content
        text = re.sub(r'\n+', '\n', text)

        # Split into sections by newlines (often paragraphs in web content)
        sections = text.split('\n')

        # Join very short sections with the next one
        merged_sections = []
        buffer = ""
        for section in sections:
            if len(buffer) < 100 and buffer:
                buffer += " " + section
            else:
                if buffer:
                    merged_sections.append(buffer)
                buffer = section
        if buffer:
            merged_sections.append(buffer)

        # Split into sentences using NLTK (more accurate for natural language)
        try:
            all_sentences = []
            for section in merged_sections:
                all_sentences.extend(sent_tokenize(section))
        except:
            # Fall back to simple splitting if NLTK fails
            all_sentences = []
            for section in merged_sections:
                all_sentences.extend([s.strip() for s in re.split(r'[.!?]+', section) if s.strip()])

        # Filter out very short sentences (often not meaningful)
        sentences = [s for s in all_sentences if len(s.split()) > 3]

        # Extract keywords from the query
        query_words = set(word.lower() for word in re.findall(r'\w+', query))

        # Calculate sentence scores based on multiple factors
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = set(word.lower() for word in re.findall(r'\w+', sentence))

            # Core scoring factors
            relevance_score = len(words.intersection(query_words)) / max(1, len(query_words))
            position_score = 1.0 / (i / 10 + 1)  # Less steep decay in importance
            length_score = min(1.0, len(sentence) / 150)  # Prefer medium-length sentences

            # Informational sentence indicators
            has_numbers = 1.5 if any(c.isdigit() for c in sentence) else 1.0  # Boost sentences with facts/numbers
            has_entities = 1.3 if re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
                                            sentence) else 1.0  # Boost named entities

            # Combine scores with weights
            sentence_scores[i] = (
                                         (0.5 * relevance_score) +
                                         (0.2 * position_score) +
                                         (0.1 * length_score)
                                 ) * has_numbers * has_entities

        # Select top sentences (with a minimum of important ones, but don't make it too long)
        num_sentences = max(5, min(20, int(len(sentences) * 0.25)))
        top_sentences = sorted(sentence_scores.keys(), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]

        # Maintain original order of sentences
        selected_indices = sorted(top_sentences)

        # Ensure we don't have just scattered sentences - try to include context
        context_indices = set(selected_indices)
        for idx in selected_indices:
            # Include the sentence before selected sentence if it exists
            if idx > 0 and idx - 1 not in context_indices:
                context_indices.add(idx - 1)
            # And sometimes the one after it
            if idx < len(sentences) - 1 and random.random() < 0.3 and idx + 1 not in context_indices:
                context_indices.add(idx + 1)

        # Recreate in order with context
        final_indices = sorted(list(context_indices))

        # Join selected sentences into a summary
        summary_parts = [sentences[i] for i in final_indices[:30]]  # Limit to prevent excessive length
        summary = " ".join(summary_parts)

        # Truncate if still too long
        if len(summary) > max_length:
            # Try to truncate at a sentence boundary
            last_period = summary[:max_length - 3].rfind('.')
            if last_period > max_length / 2:  # If we can find a good breaking point
                summary = summary[:last_period + 1]
            else:
                summary = summary[:max_length - 3] + "..."

        # Format the response as an answer to the query
        response = format_answer(query, summary)

        return response

    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}", exc_info=True)
        return f"I encountered an error while processing your query. Please try again with a more specific question."


def format_answer(query, summary):
    """Format the summary as an answer to the query with professional HTML formatting and structure."""
    # Clean up the summary
    response = summary.strip()

    # If summary is too short, return it with a simple intro
    if len(response) < 100:
        return f"<div class='doc-section'><h3>Quick Answer</h3><p><strong>Based on my search:</strong> {response}</p></div>"

    # Extract the query topic
    query = query.rstrip('?')
    query_topic = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_topic))

    # Remove common question words from the topic
    question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'do', 'does', 'did',
                      'can', 'could', 'would', 'should', 'has', 'have', 'had'}
    topic_words = [w for w in query_words if w not in question_words and len(w) > 2]
    topic_phrase = ' '.join(topic_words[:3]) if topic_words else query_topic

    # Identify query type from query text
    question_type = identify_query_type(query)

    # Check for special cases that require more structured output
    if is_ranking_query(query):
        # Attempt to create a ranking-style answer with better structure
        structured_answer = create_ranking_answer(query, summary)
        if structured_answer:
            return structured_answer

    # Check for city/location queries
    if "city" in query.lower() or "cities" in query.lower() or "cleanest" in query.lower():
        city_answer = create_city_ranking_answer(query, summary)
        if city_answer:
            return city_answer

    # Create a professional documentation-style introduction based on question type
    intros = {
        "person": [
            f"<div class='doc-section'><h3>About {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Profile Information</h3><p><strong>Overview: </strong>",
            f"<div class='doc-section'><h3>Research Results</h3><p>"
        ],
        "location": [
            f"<div class='doc-section'><h3>Location Information: {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Geographical Data</h3><p><strong>Location: </strong>",
            f"<div class='doc-section'><h3>Location Analysis</h3><p>"
        ],
        "time": [
            f"<div class='doc-section'><h3>Timeline: {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Historical Context</h3><p>",
            f"<div class='doc-section'><h3>Chronological Overview</h3><p>"
        ],
        "reason": [
            f"<div class='doc-section'><h3>Analysis: Why {topic_phrase.title()}?</h3><p>",
            f"<div class='doc-section'><h3>Key Factors</h3><p>",
            f"<div class='doc-section'><h3>Root Causes</h3><p>"
        ],
        "process": [
            f"<div class='doc-section'><h3>Process: {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Workflow Overview</h3><p>",
            f"<div class='doc-section'><h3>Step-by-Step Guide</h3><p>"
        ],
        "comparison": [
            f"<div class='doc-section'><h3>Comparison: {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Key Differences</h3><p>",
            f"<div class='doc-section'><h3>Comparative Analysis</h3><p>"
        ],
        "recommendation": [
            f"<div class='doc-section'><h3>Recommendations: {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Best Practices</h3><p>",
            f"<div class='doc-section'><h3>Expert Suggestions</h3><p>"
        ],
        "ranking": [
            f"<div class='doc-section'><h3>Rankings: {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Top Results</h3><p>",
            f"<div class='doc-section'><h3>Ranked List</h3><p>"
        ],
        "informational": [
            f"<div class='doc-section'><h3>Information: {topic_phrase.title()}</h3><p>",
            f"<div class='doc-section'><h3>Overview</h3><p>",
            f"<div class='doc-section'><h3>Key Facts</h3><p>"
        ]
    }

    # Select a random intro from the appropriate category
    selected_intro = random.choice(intros.get(question_type, intros["informational"]))

    # Try to extract ranked items, lists, or other structured content
    extracted_list = extract_list_items(summary)
    direct_answer = extract_direct_answer(query, summary)

    # Pre-process the response text for better formatting

    # Identify potential key points for bullet lists (starting with common markers)
    potential_bullets = []

    # Try to find list-like patterns in the text
    bullet_patterns = [
        r'(?:^|\n)(?:\d+\.\s+|\*\s+|:\s+)([A-Z][^.!?]*[.!?])',  # Numbered/bullet points
        r'(?:^|\n)(?:First|Second|Third|Finally|Lastly)[,:]?\s+([A-Z][^.!?]*[.!?])',  # Sequence markers
        r'(?<=[.!?])\s+([A-Z][^.!?]*? (?:include|includes|are|is|was|were):[^.!?]*[.!?])'  # Definition patterns
    ]

    for pattern in bullet_patterns:
        bullet_matches = re.findall(pattern, response)
        if bullet_matches:
            potential_bullets.extend(bullet_matches)

    # If we don't have enough bullet points but we have extracted list, use those
    if (not potential_bullets or len(potential_bullets) < 3) and extracted_list:
        potential_bullets = extracted_list

    # Find key sentences that have important facts or definitions
    fact_patterns = [
        r'([^.!?]*? is [^.!?]*?\.)',  # Definition patterns
        r'([^.!?]*? are [^.!?]*?\.)',
        r'([^.!?]*? was [^.!?]*?\.)',
        r'([^.!?]*? were [^.!?]*?\.)',
        r'([^.!?]*? has [^.!?]*?\.)',
        r'([^.!?]*? have [^.!?]*?\.)',
        r'([^.!?]*? contains [^.!?]*?\.)',
        r'([^.!?]*? includes [^.!?]*?\.)',
        r'([^.!?]*? consists of [^.!?]*?\.)',
    ]

    key_facts = []
    for pattern in fact_patterns:
        fact_matches = re.findall(pattern, response)
        if fact_matches:
            key_facts.extend(fact_matches[:2])  # Limit to avoid over-highlighting

    # IMPORTANT: Start with a direct answer if we have one
    formatted_response = ""
    if direct_answer:
        formatted_response = f"""
        <div class='doc-section'>
            <h3>Direct Answer</h3>
            <p class='direct-answer'><strong>{direct_answer}</strong></p>
        </div>
        """

    # Now begin building the main content with sections
    formatted_response += selected_intro

    # If we found potential bullet points, format them with professional styling
    if potential_bullets and len(potential_bullets) >= 3:
        # Create a professional bullet list section if we found enough points
        bullet_html = "</p></div>\n<div class='doc-section'>\n"

        # Choose appropriate title based on query type
        if question_type == "ranking" or is_ranking_query(query):
            bullet_html += "<h3>Top Results</h3>\n"
        elif "cleanest city" in query.lower() or "best city" in query.lower():
            bullet_html += "<h3>Top Cleanest Cities</h3>\n"
        else:
            bullet_html += "<h3>Key Points</h3>\n"

        # Create a numbered or unordered list based on query type
        if question_type == "ranking" or is_ranking_query(query) or "top" in query.lower() or "best" in query.lower():
            bullet_html += "<ol class='key-points'>\n"
        else:
            bullet_html += "<ul class='key-points'>\n"

        for bullet in potential_bullets[:8]:  # Limit to prevent excessive lists
            # Enhance bullet points with better formatting
            bullet_text = bullet.strip()

            # Look for potential highlightable terms in bullet points
            highlight_terms = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b', bullet_text)

            # Highlight important terms if found
            if highlight_terms:
                for term in highlight_terms:
                    # Don't highlight common words or short terms
                    if len(term) > 3 and term.lower() not in ['this', 'that', 'these', 'those', 'there', 'their',
                                                              'which', 'where', 'when', 'what']:
                        bullet_text = bullet_text.replace(term, f"<em>{term}</em>", 1)

            bullet_html += f"<li>{bullet_text}</li>\n"

        # Close the list
        if question_type == "ranking" or is_ranking_query(query) or "top" in query.lower() or "best" in query.lower():
            bullet_html += "</ol>\n</div>\n"
        else:
            bullet_html += "</ul>\n</div>\n"

        # Add the bullet list after the introduction
        formatted_response += response[:150] + "...</p></div>\n" + bullet_html
    else:
        # If we don't have bullets, just use the regular content
        formatted_response += response + "</p></div>\n"

    # Create a separate "Key Facts" section if we found substantial facts
    important_facts = [fact for fact in key_facts if len(fact) > 20]

    if len(important_facts) >= 2:
        facts_html = "<div class='doc-section'>\n"
        facts_html += "<h3>Important Facts</h3>\n"
        facts_html += "<ul class='fact-list'>\n"

        for fact in important_facts[:4]:  # Limit to top 4 facts
            # Clean up the fact text and emphasize key parts
            fact_text = fact.strip()

            # Look for specific patterns to emphasize
            if " is " in fact_text:
                parts = fact_text.split(" is ", 1)
                subject = parts[0].strip()
                predicate = parts[1].strip()
                fact_text = f"<strong>{subject}</strong> is {predicate}"
            elif " are " in fact_text:
                parts = fact_text.split(" are ", 1)
                subject = parts[0].strip()
                predicate = parts[1].strip()
                fact_text = f"<strong>{subject}</strong> are {predicate}"

            facts_html += f"<li>{fact_text}</li>\n"

        facts_html += "</ul>\n</div>\n"

        # Add the facts section 
        formatted_response += facts_html

    # If this is about a specific city or topic, add a details section
    if direct_answer and len(direct_answer) > 0:
        details_html = "<div class='doc-section'>\n"
        details_html += f"<h3>Details: {direct_answer}</h3>\n"

        # Extract sentences about the direct answer
        related_sentences = []
        for sentence in re.split(r'(?<=[.!?])\s+', summary):
            if direct_answer in sentence and len(sentence) > 30:
                related_sentences.append(sentence.strip())

        if related_sentences:
            details_html += "<ul class='details-list'>\n"
            for sentence in related_sentences[:3]:  # Limit to 3 details
                details_html += f"<li>{sentence}</li>\n"
            details_html += "</ul>\n"
        else:
            details_html += f"<p>Additional information about {direct_answer} is not available in the current search results.</p>\n"

        details_html += "</div>\n"

        # Add the details section
        formatted_response += details_html

    return formatted_response


def extract_list_items(text):
    """Extract list items from text based on patterns."""
    items = []

    # Method 1: Look for numbered items like "1. Item" or "1) Item"
    numbered = re.findall(r'(?:^|\n|\. )([1-9][0-9]?)[\.|\)]\s+([A-Z][^.!?]+)', text)
    if numbered:
        # Convert to a dictionary to handle possible duplicates
        numbered_dict = {}
        for num_str, item_text in numbered:
            try:
                num = int(num_str)
                if 1 <= num <= 20:  # Reasonable list size
                    numbered_dict[num] = item_text.strip()
            except ValueError:
                continue

        # Convert back to ordered list if we have enough items
        if len(numbered_dict) >= 3:
            # Sort by the numbers
            items = [item for _, item in sorted(numbered_dict.items())]

    # Method 2: Look for bullet points
    if not items:
        bullets = re.findall(r'(?:^|\n)[\*\-•]\s+([A-Z][^.!?]+)', text)
        if len(bullets) >= 3:
            items = [item.strip() for item in bullets]

    # Method 3: Look for city names followed by descriptions
    if not items:
        cities = re.findall(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is|was|has been)\s+(?:ranked|rated|known|named|called|considered|recognized)',
            text)
        if len(cities) >= 3:
            items = [city.strip() for city in cities]

    return items


def extract_direct_answer(query, text):
    """Extract the direct answer to a question from text."""
    query = query.lower().strip()

    # For "cleanest city" type questions
    if "cleanest city" in query or "cleanest cities" in query:
        # Try different patterns for finding the cleanest city
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|has been|was|remains)\s+(?:the|recognized as the|consistently|rated as the)\s+cleanest\s+city',
            r'the\s+cleanest\s+city\s+(?:in|of)\s+[A-Z][a-z]+\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+tops\s+the\s+list\s+of\s+cleanest\s+cities',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+was\s+awarded\s+the\s+title\s+of\s+cleanest\s+city'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        # If no direct match, try to find the first city in a list context
        list_pattern = r'(?:1|1st|one|first)[\.\)]\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        list_match = re.search(list_pattern, text)
        if list_match:
            return list_match.group(1)

    # For capital city questions
    elif "capital" in query and ("city" in query or "what is" in query):
        country_match = re.search(r'capital of ([A-Za-z]+)', query)
        if country_match:
            country = country_match.group(1)
            capital_pattern = rf'(?:capital|capital city) of {country} is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            capital_match = re.search(capital_pattern, text, re.IGNORECASE)
            if capital_match:
                return capital_match.group(1)

    # For "what is" or definition queries
    elif query.startswith("what is"):
        topic = query.replace("what is", "").replace("?", "").strip()
        if topic:
            definition_pattern = rf'{topic} (?:is|refers to|means) ([^.!?]+)'
            definition_match = re.search(definition_pattern, text, re.IGNORECASE)
            if definition_match:
                return definition_match.group(1).strip()

    # For questions about tallest, shortest, biggest, etc.
    elif any(word in query for word in ["tallest", "highest", "biggest", "largest", "smallest", "shortest"]):
        for word in ["tallest", "highest", "biggest", "largest", "smallest", "shortest"]:
            if word in query:
                pattern = rf'(?:the|world\'s|earth\'s) {word} ([^.!?]+) is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return f"{match.group(2)} ({match.group(1)})"

    # No direct answer found
    return None


def create_ranking_answer(query, summary):
    """Create a structured ranking answer for list-type queries."""
    # Extract list items from the summary
    items = extract_list_items(summary)

    # If we couldn't extract a proper list, return None
    if not items or len(items) < 3:
        return None

    # Format a structured answer
    result = "<div class='doc-section'>\n"

    # Create a header based on the query
    if "cleanest city" in query.lower() or "cleanest cities" in query.lower():
        result += "<h3>Top Cleanest Cities</h3>\n"
    elif "top" in query.lower():
        result += "<h3>Top Results</h3>\n"
    elif "best" in query.lower():
        result += "<h3>Best Results</h3>\n"
    else:
        result += "<h3>Ranked List</h3>\n"

    # Create a numbered list
    result += "<ol class='ranked-list'>\n"
    for item in items[:10]:  # Limit to top 10
        result += f"<li><strong>{item}</strong></li>\n"
    result += "</ol>\n</div>\n"

    # Add a brief explanation section
    result += "<div class='doc-section'>\n"
    result += "<h3>Context</h3>\n"
    result += f"<p>{summary[:300]}...</p>\n"
    result += "</div>\n"

    return result


def create_city_ranking_answer(query, summary):
    """Create a specialized answer for city ranking queries."""
    # Try to extract the direct answer (top city)
    top_city = extract_direct_answer(query, summary)

    # Try to extract a list of cities
    city_list = extract_list_items(summary)

    # If we don't have the necessary information, return None
    if not (top_city or (city_list and len(city_list) >= 3)):
        return None

    # Start building the response
    result = ""

    # Add the direct answer section if we found one
    if top_city:
        result += f"""
        <div class='doc-section'>
            <h3>Answer</h3>
            <p class='direct-answer'><strong>{top_city}</strong> is the cleanest city.</p>
        </div>
        """

    # Add the ranking list if available
    if city_list and len(city_list) >= 3:
        result += "<div class='doc-section'>\n"
        result += "<h3>Top 10 Cleanest Cities</h3>\n"
        result += "<ol class='ranked-list'>\n"

        for i, city in enumerate(city_list[:10]):  # Limit to top 10
            # Highlight the top city if it matches our direct answer
            if top_city and city == top_city:
                result += f"<li><strong>{city}</strong> (Winner)</li>\n"
            else:
                result += f"<li><strong>{city}</strong></li>\n"

        result += "</ol>\n</div>\n"

    # If we have a top city, add a details section
    if top_city:
        result += "<div class='doc-section'>\n"
        result += f"<h3>{top_city}'s Cleanliness Initiatives</h3>\n"

        # Extract sentences about the top city
        city_sentences = []
        for sentence in re.split(r'(?<=[.!?])\s+', summary):
            if top_city in sentence and len(sentence) > 30:
                city_sentences.append(sentence.strip())

        if city_sentences:
            result += "<ul class='initiative-list'>\n"
            for sentence in city_sentences[:4]:  # Limit to 4 points
                result += f"<li>{sentence}</li>\n"
            result += "</ul>\n"
        else:
            # If we can't find specific sentences, add a general paragraph
            result += f"<p>{summary[:250]}...</p>\n"

        result += "</div>\n"

    # Add a brief explanation of the ranking system if available
    ranking_info = re.search(r'(?:[Tt]hese rankings|[Tt]he rankings|[Tt]he assessment|[Tt]he survey) [^.!?]+[.!?]',
                             summary)
    if ranking_info:
        result += "<div class='doc-section'>\n"
        result += "<h3>Ranking Methodology</h3>\n"
        result += f"<p>{ranking_info.group(0)}</p>\n"
        result += "</div>\n"

    return result


def identify_query_type(query):
    """Identify the type of query for better formatting."""
    query = query.lower()

    if any(x in query for x in ["who", "person", "people", "name"]):
        return "person"
    elif any(x in query for x in ["where", "location", "place", "country", "city"]):
        return "location"
    elif any(x in query for x in ["when", "date", "time", "year"]):
        return "time"
    elif any(x in query for x in ["how many", "count", "number", "total"]):
        return "quantity"
    elif any(x in query for x in ["what is", "meaning", "define", "definition"]):
        return "definition"
    elif any(x in query for x in ["list", "top", "best", "ranked", "rating", "cleanest"]):
        return "ranking"
    elif any(x in query for x in ["why", "reason", "cause", "because"]):
        return "reason"
    elif any(x in query for x in ["how", "process", "steps", "way", "method", "procedure"]):
        return "process"
    elif any(x in query for x in ["difference", "compare", "versus", "vs"]):
        return "comparison"
    else:
        return "informational"


def is_ranking_query(query):
    """Check if this is a ranking-type query."""
    query = query.lower()
    return any(x in query for x in
               ["top", "best", "cleanest", "greatest", "largest", "smallest", "highest", "lowest", "rank", "list"])


def extract_list_items(text):
    """Extract list items from text based on patterns."""
    items = []

    # Method 1: Look for numbered items like "1. Item" or "1) Item"
    numbered = re.findall(r'(?:^|\n|\. )([1-9][0-9]?)[\.|\)]\s+([A-Z][^.!?]+)', text)
    if numbered:
        # Convert to a dictionary to handle possible duplicates
        numbered_dict = {}
        for num_str, item_text in numbered:
            try:
                num = int(num_str)
                if 1 <= num <= 20:  # Reasonable list size
                    numbered_dict[num] = item_text.strip()
            except ValueError:
                continue

        # Convert back to ordered list if we have enough items
        if len(numbered_dict) >= 3:
            # Sort by the numbers
            items = [item for _, item in sorted(numbered_dict.items())]

    # Method 2: Look for bullet points
    if not items:
        bullets = re.findall(r'(?:^|\n)[\*\-•]\s+([A-Z][^.!?]+)', text)
        if len(bullets) >= 3:
            items = [item.strip() for item in bullets]

    # Method 3: Look for city names followed by descriptions
    if not items:
        cities = re.findall(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is|was|has been)\s+(?:ranked|rated|known|named|called|considered|recognized)',
            text)
        if len(cities) >= 3:
            items = [city.strip() for city in cities]

    return items


def extract_direct_answer(query, text):
    """Extract the direct answer to a question from text."""
    query = query.lower().strip()

    # For "cleanest city" type questions
    if "cleanest city" in query or "cleanest cities" in query:
        # Try different patterns for finding the cleanest city
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|has been|was|remains)\s+(?:the|recognized as the|consistently|rated as the)\s+cleanest\s+city',
            r'the\s+cleanest\s+city\s+(?:in|of)\s+[A-Z][a-z]+\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+tops\s+the\s+list\s+of\s+cleanest\s+cities',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+was\s+awarded\s+the\s+title\s+of\s+cleanest\s+city'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        # If no direct match, try to find the first city in a list context
        list_pattern = r'(?:1|1st|one|first)[\.\)]\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        list_match = re.search(list_pattern, text)
        if list_match:
            return list_match.group(1)

    # For capital city questions
    elif "capital" in query and ("city" in query or "what is" in query):
        country_match = re.search(r'capital of ([A-Za-z]+)', query)
        if country_match:
            country = country_match.group(1)
            capital_pattern = rf'(?:capital|capital city) of {country} is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            capital_match = re.search(capital_pattern, text, re.IGNORECASE)
            if capital_match:
                return capital_match.group(1)

    # For "what is" or definition queries
    elif query.startswith("what is"):
        topic = query.replace("what is", "").replace("?", "").strip()
        if topic:
            definition_pattern = rf'{topic} (?:is|refers to|means) ([^.!?]+)'
            definition_match = re.search(definition_pattern, text, re.IGNORECASE)
            if definition_match:
                return definition_match.group(1).strip()

    # For questions about tallest, shortest, biggest, etc.
    elif any(word in query for word in ["tallest", "highest", "biggest", "largest", "smallest", "shortest"]):
        for word in ["tallest", "highest", "biggest", "largest", "smallest", "shortest"]:
            if word in query:
                pattern = rf'(?:the|world\'s|earth\'s) {word} ([^.!?]+) is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return f"{match.group(2)} ({match.group(1)})"

    # No direct answer found
    return None


def create_ranking_answer(query, summary):
    """Create a structured ranking answer for list-type queries."""
    # Extract list items from the summary
    items = extract_list_items(summary)

    # If we couldn't extract a proper list, return None
    if not items or len(items) < 3:
        return None

    # Format a structured answer
    result = "<div class='doc-section'>\n"

    # Create a header based on the query
    if "cleanest city" in query.lower() or "cleanest cities" in query.lower():
        result += "<h3>Top Cleanest Cities</h3>\n"
    elif "top" in query.lower():
        result += "<h3>Top Results</h3>\n"
    elif "best" in query.lower():
        result += "<h3>Best Results</h3>\n"
    else:
        result += "<h3>Ranked List</h3>\n"

    # Create a numbered list
    result += "<ol class='ranked-list'>\n"
    for item in items[:10]:  # Limit to top 10
        result += f"<li><strong>{item}</strong></li>\n"
    result += "</ol>\n</div>\n"

    # Add a brief explanation section
    result += "<div class='doc-section'>\n"
    result += "<h3>Context</h3>\n"
    result += f"<p>{summary[:300]}...</p>\n"
    result += "</div>\n"

    return result


def create_city_ranking_answer(query, summary):
    """Create a specialized answer for city ranking queries."""
    # Try to extract the direct answer (top city)
    top_city = extract_direct_answer(query, summary)

    # Try to extract a list of cities
    city_list = extract_list_items(summary)

    # If we don't have the necessary information, return None
    if not (top_city or (city_list and len(city_list) >= 3)):
        return None

    # Start building the response
    result = ""

    # Add the direct answer section if we found one
    if top_city:
        result += f"""
        <div class='doc-section'>
            <h3>Answer</h3>
            <p class='direct-answer'><strong>{top_city}</strong> is the cleanest city.</p>
        </div>
        """

    # Add the ranking list if available
    if city_list and len(city_list) >= 3:
        result += "<div class='doc-section'>\n"
        result += "<h3>Top 10 Cleanest Cities</h3>\n"
        result += "<ol class='ranked-list'>\n"

        for i, city in enumerate(city_list[:10]):  # Limit to top 10
            # Highlight the top city if it matches our direct answer
            if top_city and city == top_city:
                result += f"<li><strong>{city}</strong> (Winner)</li>\n"
            else:
                result += f"<li><strong>{city}</strong></li>\n"

        result += "</ol>\n</div>\n"

    # If we have a top city, add a details section
    if top_city:
        result += "<div class='doc-section'>\n"
        result += f"<h3>{top_city}'s Cleanliness Initiatives</h3>\n"

        # Extract sentences about the top city
        city_sentences = []
        for sentence in re.split(r'(?<=[.!?])\s+', summary):
            if top_city in sentence and len(sentence) > 30:
                city_sentences.append(sentence.strip())

        if city_sentences:
            result += "<ul class='initiative-list'>\n"
            for sentence in city_sentences[:4]:  # Limit to 4 points
                result += f"<li>{sentence}</li>\n"
            result += "</ul>\n"
        else:
            # If we can't find specific sentences, add a general paragraph
            result += f"<p>{summary[:250]}...</p>\n"

        result += "</div>\n"

    # Add a brief explanation of the ranking system if available
    ranking_info = re.search(r'(?:[Tt]hese rankings|[Tt]he rankings|[Tt]he assessment|[Tt]he survey) [^.!?]+[.!?]',
                             summary)
    if ranking_info:
        result += "<div class='doc-section'>\n"
        result += "<h3>Ranking Methodology</h3>\n"
        result += f"<p>{ranking_info.group(0)}</p>\n"
        result += "</div>\n"

    return result

    # For responses that don't have proper section structure yet, create professional sections
    if '<div class=\'doc-section\'>' not in response and '<div class="doc-section">' not in response:
        # Try to split into logical paragraphs and create proper sections
        sentences = re.split(r'(?<=[.!?])\s+', response)

        if len(sentences) > 5:
            # Create 2-3 sections depending on content length
            sections_needed = min(3, max(2, len(sentences) // 8))

            # Try to identify logical section breaks based on content
            section_titles = []

            # If we have enough sentences, try to extract potential section titles
            # by looking for sentences that might describe categories or topics
            potential_sections = []
            for i, sentence in enumerate(sentences):
                if i < len(sentences) - 3:  # Don't use sentences near the end
                    # Look for sentences that might introduce a new topic
                    if re.search(
                            r'\b(types|categories|examples|benefits|features|applications|uses|aspects|advantages|steps|stages|phases|components|elements|factors|parts)\b',
                            sentence.lower()):
                        potential_sections.append((i, sentence))

            # If we found potential section breaks, use them
            if len(potential_sections) >= sections_needed - 1:
                structured_response = ""

                # First section - Introduction
                start_idx = 0

                for section_idx, (break_idx, break_sentence) in enumerate(potential_sections[:sections_needed - 1]):
                    # Extract section text
                    section_text = ' '.join(sentences[start_idx:break_idx])

                    # Generate section title from the breaking sentence
                    title_match = re.search(
                        r'\b(types|categories|examples|benefits|features|applications|uses|aspects|advantages|steps|stages|phases|components|elements|factors|parts)\b',
                        break_sentence.lower())
                    if title_match:
                        title_word = title_match.group(0).title()
                        # Find what the title word applies to
                        topic_match = re.search(r'of\s+([^.,:;]+)', break_sentence)
                        if topic_match:
                            section_title = f"{title_word} of {topic_match.group(1).strip().title()}"
                        else:
                            section_title = f"Key {title_word}"
                    else:
                        section_title = f"Section {section_idx + 1}"

                    # Add the section
                    structured_response += f"<div class='doc-section'>\n<h3>{section_title}</h3>\n<p>{section_text}</p>\n</div>\n"

                    # Move to next section
                    start_idx = break_idx

                # Final section
                final_section = ' '.join(sentences[start_idx:])
                structured_response += f"<div class='doc-section'>\n<h3>Summary</h3>\n<p>{final_section}</p>\n</div>\n"

                response = structured_response
            else:
                # Fallback: create evenly divided sections
                sentences_per_section = len(sentences) // sections_needed

                structured_response = ""
                for i in range(sections_needed):
                    start_idx = i * sentences_per_section
                    # For the last section, include all remaining sentences
                    end_idx = (i + 1) * sentences_per_section if i < sections_needed - 1 else len(sentences)
                    section_text = ' '.join(sentences[start_idx:end_idx])

                    # Generate generic section titles
                    if i == 0:
                        section_title = "Overview"
                    elif i == sections_needed - 1:
                        section_title = "Conclusion"
                    else:
                        section_title = f"Details {i}"

                    structured_response += f"<div class='doc-section'>\n<h3>{section_title}</h3>\n<p>{section_text}</p>\n</div>\n"

                response = structured_response
        else:
            # For shorter responses, ensure there's at least one structured section
            if not response.startswith('<div class'):
                response = f"<div class='doc-section'>\n<h3>Quick Information</h3>\n<p>{response}</p>\n</div>"

    # Ensure any paragraphs have proper closing tags
    if response.count('<p>') > response.count('</p>'):
        response += '</p>'

    # Add the intro if not already present
    plain_intros = [intro.replace('<p>', '').replace('<strong>', '').replace('</strong>', '')
                    for intro in intros[question_type]]
    if not any(response.startswith(intro.strip()) for intro in plain_intros):
        # Remove existing paragraph tag if present to avoid nesting issues
        if response.startswith('<p>'):
            response = response[3:]
        response = selected_intro + response

    # Special handling for process/steps type questions - try to create ordered list
    if question_type == "process" and 'steps' in query_topic:
        # Look for numbers followed by text, which might be steps
        step_matches = re.findall(r'(?:^|\s)(\d+)[.)]?\s+([A-Z][^.!?]*[.!?])', response)
        if step_matches and len(step_matches) >= 3:
            # Create ordered list
            steps_html = "<ol>\n"
            for _, step_text in step_matches:
                steps_html += f"<li>{step_text.strip()}</li>\n"
            steps_html += "</ol>\n"

            # Find a good position to add the steps (after intro)
            intro_end = response.find('</p>')
            if intro_end > 0:
                response = response[:intro_end + 4] + "\n" + steps_html + response[intro_end + 4:]

    # Ensure we don't have any unclosed HTML tags
    if response.count('<p>') > response.count('</p>'):
        response += '</p>'

    return response
