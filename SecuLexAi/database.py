import os
import json
import logging
import re
import psycopg2
from psycopg2 import pool
from datetime import datetime
from collections import Counter
import hashlib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Create a connection pool
connection_pool = None


def init_db():
    """Initialize the PostgreSQL database with required tables."""
    global connection_pool

    try:
        # Setup connection pool
        if not connection_pool and DATABASE_URL:
            connection_pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)
            logger.info("Database connection pool created successfully")

        # Get a connection from the pool
        conn = get_connection()
        cursor = conn.cursor()

        # First, check if we need to migrate the old schema
        try:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='qa_pairs' AND column_name='query_hash'")
            exists = cursor.fetchone()

            if not exists:
                logger.info("Old schema detected. Dropping table to recreate with new schema.")
                cursor.execute("DROP TABLE IF EXISTS qa_pairs")
                conn.commit()
        except Exception as e:
            logger.warning(f"Error checking schema: {str(e)}")
            # Continue with table creation

        # Create improved table for storing Q&A pairs with more metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            query_hash VARCHAR(64) NOT NULL,
            query_keywords TEXT[] NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            use_count INTEGER DEFAULT 1,
            query_type VARCHAR(30) DEFAULT 'informational'
        )
        ''')

        # Create indexes for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_hash ON qa_pairs(query_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords ON qa_pairs USING GIN(query_keywords)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_type ON qa_pairs(query_type)')

        conn.commit()
        release_connection(conn)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        raise


def get_connection():
    """Get a database connection from the pool."""
    global connection_pool
    if connection_pool:
        return connection_pool.getconn()
    else:
        # Fallback to direct connection if pool not available
        return psycopg2.connect(DATABASE_URL)


def release_connection(conn):
    """Release a connection back to the pool."""
    global connection_pool
    if connection_pool:
        connection_pool.putconn(conn)
    else:
        conn.close()


def store_qa_pair(query, response, query_type='informational'):
    """
    Store a new Q&A pair in the database with improved metadata.
    Includes quality verification to prevent storing invalid data.
    
    Args:
        query (str): The user's question
        response (str): The answer/response
        query_type (str): The type of query (informational, person, location, etc.)
    """
    try:
        # STEP 1: Quality verification to prevent storing invalid data
        if not is_valid_qa_pair(query, response):
            logger.warning(f"Rejected low-quality Q&A pair. Query: {query[:50]}...")
            return False

        # Generate a hash for the query to help with exact matching
        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()

        # Extract meaningful keywords for better search
        clean_q = clean_text(query.lower())
        query_keywords = extract_keywords(clean_q)

        # Connect to database
        conn = get_connection()
        cursor = conn.cursor()

        # Check if exact query hash already exists
        cursor.execute(
            'SELECT id, use_count, response FROM qa_pairs WHERE query_hash = %s',
            (query_hash,)
        )

        exact_match = cursor.fetchone()

        if exact_match:
            # STEP 2: Quality check - is the new response better than existing one?
            qa_id, use_count, existing_response = exact_match

            # Only update if new response is higher quality or significantly different
            if compare_response_quality(existing_response, response):
                use_count += 1

                cursor.execute(
                    '''
                    UPDATE qa_pairs 
                    SET response = %s, 
                        updated_at = %s, 
                        use_count = %s,
                        query_type = %s
                    WHERE id = %s
                    ''',
                    (response, datetime.now().isoformat(), use_count, query_type, qa_id)
                )
                logger.debug(f"Updated existing Q&A pair with better response (used {use_count} times)")
            else:
                # Just increment usage count without changing response
                cursor.execute(
                    '''
                    UPDATE qa_pairs 
                    SET use_count = use_count + 1
                    WHERE id = %s
                    ''',
                    (qa_id,)
                )
                logger.debug(f"Kept existing response but incremented usage count")
        else:
            # Check for very similar query using keyword matching
            similar_query, confidence = find_similar_query(query, threshold=0.85)

            if similar_query and confidence > 0.85:
                # Get the existing response for comparison
                cursor.execute(
                    'SELECT response FROM qa_pairs WHERE query = %s',
                    (similar_query,)
                )
                similar_result = cursor.fetchone()
                existing_response = similar_result[0] if similar_result else None

                # STEP 3: Quality check for similar query update
                if existing_response and compare_response_quality(existing_response, response):
                    logger.debug(f"Found similar query with confidence {confidence:.2f}")
                    cursor.execute(
                        '''
                        UPDATE qa_pairs 
                        SET response = %s, 
                            updated_at = %s,
                            use_count = use_count + 1,
                            query_type = %s
                        WHERE query = %s
                        RETURNING use_count
                        ''',
                        (response, datetime.now().isoformat(), query_type, similar_query)
                    )
                    result = cursor.fetchone()
                    updated_count = result[0] if result else 1
                    logger.debug(f"Updated similar Q&A pair with better response (used {updated_count} times)")
                else:
                    # Just increment the usage count
                    cursor.execute(
                        '''
                        UPDATE qa_pairs 
                        SET use_count = use_count + 1
                        WHERE query = %s
                        ''',
                        (similar_query,)
                    )
                    logger.debug(f"Kept existing response for similar query but incremented usage count")
            else:
                # STEP 4: Final verification before adding completely new Q&A pair
                # Only store meaningful queries and responses
                if len(query.split()) >= 3 and len(response) >= 100:
                    cursor.execute(
                        '''
                        INSERT INTO qa_pairs 
                        (query, query_hash, query_keywords, response, created_at, updated_at, use_count, query_type) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ''',
                        (
                            query,
                            query_hash,
                            query_keywords,
                            response,
                            datetime.now().isoformat(),
                            datetime.now().isoformat(),
                            1,
                            query_type
                        )
                    )
                    logger.debug(f"Created new validated Q&A pair: {query[:50]}...")
                else:
                    logger.warning(f"Rejected too short query or response. Query: {query}")

        conn.commit()
        release_connection(conn)
        return True

    except Exception as e:
        logger.error(f"Error storing Q&A pair: {str(e)}", exc_info=True)
        return False


def is_valid_qa_pair(query, response):
    """
    Verify if a Q&A pair meets quality standards for storage.
    
    Args:
        query (str): The user's question
        response (str): The system's response
        
    Returns:
        bool: True if the pair meets quality standards, False otherwise
    """
    # Check 1: Minimum length requirements
    if len(query.strip()) < 5 or len(response.strip()) < 50:
        return False

    # Check 2: Query must contain actual question words or be a clear command
    question_indicators = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'can', 'is', 'are', 'will', 'should',
                           'did', 'does', 'do']
    command_indicators = ['tell', 'explain', 'describe', 'show', 'list', 'find', 'search', 'get', 'give']

    query_lower = query.lower()
    has_question_indicator = any(indicator in query_lower for indicator in question_indicators)
    has_command_indicator = any(indicator in query_lower for indicator in command_indicators)

    if not (has_question_indicator or has_command_indicator):
        return False

    # Check 3: Response must not be an error message
    error_indicators = [
        'error', 'sorry', 'couldn\'t find', 'no information', 'no results',
        'unable to', 'failed to', 'cannot', 'not available'
    ]

    response_lower = response.lower()
    if any(indicator in response_lower for indicator in error_indicators) and len(response) < 200:
        return False

    # Check 4: Response should contain substantial text content
    if response.count('.') < 3:  # At least 3 sentences
        return False

    # Check 5: Response content should be relevant to query
    # Extract keywords from both query and response
    query_words = set(extract_words(clean_text(query_lower)))
    response_words = set(extract_words(clean_text(response_lower)))

    # Calculate keyword overlap
    if len(query_words) > 0:
        overlap = len(query_words.intersection(response_words)) / len(query_words)
        if overlap < 0.2:  # At least 20% keyword overlap
            return False

    return True


def compare_response_quality(existing_response, new_response):
    """
    Compare the quality of two responses to determine if the new one is better.
    
    Args:
        existing_response (str): The current response in the database
        new_response (str): The new response to potentially store
        
    Returns:
        bool: True if the new response is better quality, False otherwise
    """
    # Only replace if new response is significantly different
    if len(new_response) < 50:
        return False

    # Strategy 1: Length comparison - longer responses tend to be more informative
    # But only if the difference is significant (>30%)
    len_diff_ratio = len(new_response) / max(1, len(existing_response))
    if len_diff_ratio > 1.3:
        return True
    elif len_diff_ratio < 0.7:
        return False

    # Strategy 2: Content diversity - more sentences or paragraphs usually mean more information
    existing_sentences = existing_response.count('.')
    new_sentences = new_response.count('.')

    if new_sentences > existing_sentences * 1.5:
        return True

    # Strategy 3: Structural richness - HTML formatting indicates better structured content
    existing_html_tags = len(re.findall(r'<[^>]+>', existing_response))
    new_html_tags = len(re.findall(r'<[^>]+>', new_response))

    if new_html_tags > existing_html_tags * 1.5:
        return True

    # Strategy 4: Include basic semantic analysis
    existing_words = set(extract_words(clean_text(existing_response.lower())))
    new_words = set(extract_words(clean_text(new_response.lower())))

    # If new response contains substantially more unique words
    if len(new_words) > len(existing_words) * 1.3:
        return True

    # By default, prefer keeping the existing response
    return False


def find_similar_query(query, threshold=0.7):
    """
    Find a semantically similar query in the database using improved matching.
    
    Args:
        query (str): The query to find matches for
        threshold (float): Similarity threshold (0-1)
        
    Returns:
        tuple: (response, confidence) if found, else (None, 0)
    """
    try:
        # Generate hash for exact matching
        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()

        # Clean query and extract keywords
        clean_q = clean_text(query.lower())
        query_keywords = extract_keywords(clean_q)

        if not query_keywords:
            return None, 0

        # Connect to database
        conn = get_connection()
        cursor = conn.cursor()

        # First check for exact hash match
        cursor.execute(
            'SELECT query, response FROM qa_pairs WHERE query_hash = %s ORDER BY use_count DESC LIMIT 1',
            (query_hash,)
        )
        exact_match = cursor.fetchone()

        if exact_match:
            db_query, db_response = exact_match
            release_connection(conn)
            logger.debug(f"Found exact hash match for query")
            return db_response, 1.0

        # Next, find records that share keywords
        keywords_param = '{' + ','.join(query_keywords) + '}'

        cursor.execute(
            '''
            SELECT query, response, query_keywords
            FROM qa_pairs
            WHERE query_keywords && %s
            ORDER BY use_count DESC, updated_at DESC
            LIMIT 50
            ''',
            (keywords_param,)
        )

        potential_matches = cursor.fetchall()
        release_connection(conn)

        if not potential_matches:
            return None, 0

        # Compare similarity using multiple metrics
        best_match = None
        best_score = 0

        # Get n-grams from the query for comparison
        query_bigrams = get_ngrams(clean_q, 2)
        query_trigrams = get_ngrams(clean_q, 3)
        query_keywords_set = set(query_keywords)

        for db_query, db_response, db_keywords in potential_matches:
            # Skip if no overlap in keywords at all
            db_keywords_set = set(db_keywords)
            keyword_overlap = len(query_keywords_set.intersection(db_keywords_set))

            if keyword_overlap == 0:
                continue

            # Calculate keyword similarity (Jaccard similarity)
            keyword_union = len(query_keywords_set.union(db_keywords_set))
            keyword_similarity = keyword_overlap / keyword_union if keyword_union > 0 else 0

            # Calculate n-gram similarity
            clean_db_query = clean_text(db_query.lower())
            db_bigrams = get_ngrams(clean_db_query, 2)
            db_trigrams = get_ngrams(clean_db_query, 3)

            # Calculate bigram overlap
            bigram_overlap = len(query_bigrams.intersection(db_bigrams))
            bigram_union = len(query_bigrams.union(db_bigrams))
            bigram_similarity = bigram_overlap / bigram_union if bigram_union > 0 else 0

            # Calculate trigram overlap
            trigram_overlap = len(query_trigrams.intersection(db_trigrams))
            trigram_union = len(query_trigrams.union(db_trigrams))
            trigram_similarity = trigram_overlap / trigram_union if trigram_union > 0 else 0

            # Calculate combined similarity score with weights
            # Give more weight to keywords (semantic) than to n-grams (syntactic)
            combined_score = (
                    0.5 * keyword_similarity +
                    0.3 * bigram_similarity +
                    0.2 * trigram_similarity
            )

            # Apply length penalty (larger difference in length = lower score)
            len_ratio = min(len(clean_q), len(clean_db_query)) / max(len(clean_q), len(clean_db_query))
            final_score = combined_score * len_ratio

            if final_score > best_score:
                best_score = final_score
                best_match = db_response

        # Return the best match if above threshold
        if best_score >= threshold:
            logger.debug(f"Found similar query with confidence {best_score:.2f}")
            return best_match, best_score
        else:
            logger.debug(f"No similar query found above threshold (best: {best_score:.2f})")
            return None, best_score

    except Exception as e:
        logger.error(f"Error finding similar query: {str(e)}", exc_info=True)
        return None, 0


def clean_text(text):
    """Clean and normalize text for comparison."""
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_keywords(text):
    """Extract meaningful keywords from text."""
    # Split by spaces and filter out stop words and very short words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                  'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like',
                  'from', 'of', 'as', 'what', 'when', 'where', 'who', 'why', 'how',
                  'can', 'could', 'would', 'should', 'may', 'might', 'must', 'need',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'should',
                  'this', 'that', 'these', 'those', 'them', 'they', 'their', 'we', 'us',
                  'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she',
                  'her', 'hers', 'it', 'its', 'be', 'been', 'being', 'am'}

    words = [word for word in text.split() if word not in stop_words and len(word) > 2]

    # Get most frequent words for better relevance
    if len(words) > 5:
        word_counts = Counter(words)
        keywords = [word for word, _ in word_counts.most_common(15)]
    else:
        keywords = words

    return keywords


def extract_words(text):
    """Extract all meaningful words from text."""
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                  'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like',
                  'from', 'of', 'as', 'what', 'when', 'where', 'who', 'why', 'how'}

    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return words


def get_ngrams(text, n):
    """Generate character n-grams from text."""
    return set(' '.join([text[i:i + n] for i in range(len(text) - n + 1)]))


def get_database_stats():
    """Get statistics about the QA database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get total count of QA pairs
        cursor.execute('SELECT COUNT(*) FROM qa_pairs')
        result = cursor.fetchone()
        total_count = result[0] if result else 0

        # Get most used QA pairs
        cursor.execute(
            '''
            SELECT query, use_count 
            FROM qa_pairs 
            ORDER BY use_count DESC 
            LIMIT 5
            '''
        )
        most_used = cursor.fetchall()

        # Get count by query type
        cursor.execute(
            '''
            SELECT query_type, COUNT(*) 
            FROM qa_pairs 
            GROUP BY query_type 
            ORDER BY COUNT(*) DESC
            '''
        )
        type_counts = cursor.fetchall()

        release_connection(conn)

        return {
            "total_qa_pairs": total_count,
            "most_used_queries": most_used,
            "query_type_distribution": type_counts
        }

    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}", exc_info=True)
        return {
            "total_qa_pairs": 0,
            "most_used_queries": [],
            "query_type_distribution": []
        }
