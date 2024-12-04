echo "Installing requirements..." >> /home/site/wwwroot/startup.log
pip install -r requirements.txt  # Installeer de dependencies
echo "Starting run.py" >> /home/site/wwwroot/startup.log
python run.py  # Start de Flask-app met de ontwikkelserver
