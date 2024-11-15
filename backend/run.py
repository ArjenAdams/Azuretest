import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Haal de poort op uit omgevingsvariabelen, standaard naar 5000
    port = int(os.environ.get("PORT", 5000))

    # Start de app op host 0.0.0.0 en de opgegeven port
    app.run(host='0.0.0.0', port=port)