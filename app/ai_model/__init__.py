import os
import sys

# Een fix voor het importeren van modules tijdens het opstarten van de
# applicatie
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
