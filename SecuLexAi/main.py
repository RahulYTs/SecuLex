import os
import logging
from flask import Flask, request, render_template, jsonify, session
from SecuLexAi import database
from SecuLexAi import search
from SecuLexAi import model

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key_for_development")

# Ensure the database is initialized
database.init_db()


@app.route('/')
def index():
    """Render the main chat interface."""
    # Create a chat history in session if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []

    return render_template('index.html', chat_history=session['chat_history'])


@app.route('/learning')
def learning_stats():
    """Render the learning statistics page."""
    return render_template('stats.html')


@app.route('/ask', methods=['POST'])
def ask():
    """Process user query and return response."""
    try:
        data = request.json
        query = data['query'].strip() if data and 'query' in data else ''

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        logger.debug(f"Received query: {query}")

        # Identify query type for better matching and categorization
        query_type = identify_query_type(query)
        logger.debug(f"Query type identified as: {query_type}")

        # Step 1: Check if we have a similar query in our database
        db_response, confidence = database.find_similar_query(query)

        # If we found a similar query with high confidence, return that response
        if db_response and confidence >= 0.7:
            logger.debug(f"Found answer in database with confidence {confidence:.2f}")
            response = db_response
            source = "database"

            # Log usage statistics
            add_metadata = {
                'source': 'database',
                'confidence': f"{confidence:.2f}"
            }
        else:
            # Step 2: If not in database, search the web
            logger.debug("Searching the web for information...")
            search_results = search.search_web(query)

            # Step 3: Summarize the search results
            logger.debug("Summarizing search results...")
            response = model.summarize(query, search_results)
            source = "web"

            # Step 4: Store the new Q&A pair in the database with improved metadata
            logger.debug(f"Storing new Q&A pair in database with type: {query_type}...")
            database.store_qa_pair(query, response, query_type)

            # Log usage statistics
            add_metadata = {
                'source': 'web',
                'query_type': query_type
            }

        # Update chat history in session
        chat_entry = {'role': 'user', 'content': query}
        session['chat_history'] = session.get('chat_history', []) + [chat_entry]

        chat_entry = {'role': 'assistant', 'content': response, 'source': source}
        session['chat_history'] = session.get('chat_history', []) + [chat_entry]
        session.modified = True

        return jsonify({
            'response': response,
            'source': source,
            'metadata': add_metadata
        })

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def identify_query_type(query):
    """
    Identify the type of query to improve matching and categorization.
    
    Returns:
        str: The query type (person, location, time, reason, process, 
             comparison, recommendation, or informational)
    """
    query_topic = query.lower()

    # Question classification - detect question type
    if any(word in query_topic for word in ['who', 'person', 'people', 'someone']):
        return "person"
    elif any(word in query_topic for word in ['where', 'location', 'place', 'country', 'city']):
        return "location"
    elif any(word in query_topic for word in ['when', 'date', 'time', 'year', 'month', 'day']):
        return "time"
    elif any(word in query_topic for word in ['why', 'reason', 'cause', 'because']):
        return "reason"
    elif any(word in query_topic for word in ['how', 'process', 'steps', 'way', 'method', 'procedure']):
        return "process"
    elif any(word in query_topic for word in ['difference', 'compare', 'versus', 'vs', 'similarities']):
        return "comparison"
    elif any(word in query_topic for word in ['best', 'top', 'most', 'recommend', 'suggestion', 'should']):
        return "recommendation"
    else:
        return "informational"


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the chat history in the session."""
    session['chat_history'] = []
    return jsonify({'status': 'success'})


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the database and learning progress."""
    try:
        # Get database statistics
        stats = database.get_database_stats()

        return jsonify({
            'total_qa_pairs': stats['total_qa_pairs'],
            'most_used_queries': [
                {'query': q, 'count': c} for q, c in stats['most_used_queries']
            ],
            'query_types': [
                {'type': t, 'count': c} for t, c in stats['query_type_distribution']
            ]
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
