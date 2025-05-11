from SecuLexAi.main import app

# This file serves as an entry point for Gunicorn to load the Flask application

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)