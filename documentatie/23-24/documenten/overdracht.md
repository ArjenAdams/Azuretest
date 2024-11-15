
## Inleiding
Met behulp van een interactief dashboard willen we het mogelijk maken dat gebruikers, artsen/patiënten, kunnen interacteren met een AI-model. We willen verschillende vormen van explainable AI (XAI) mogelijk maken en meten wat het effect van uitleg is op het vertrouwen van de gebruiker op het AI-model. 

In dit document wordt beknopt beschreven wat er gemaakt is in het semester 2024, hoe men er mee om gaat, en hoe de volgende groep het beste verder kan werken.


# AI en explanations

Bij dit document zitten twee html exports van Jupyter notebooks gevoegd: in 'HERNOEM DIT.html' is terug te vinden hoe het model getraind is; in 'overzicht-schermen.html' staat een overzicht van iedere explanation, en documentatie van over hoe het werkt. De html bestanden kunnen geopend worden met een browser naar keuze, de oorspronkelijke ipynb bestanden zijn ook terug te vinden in de Github repository.

## SAD-Model

Het model verwerkt in het dashboard is een XGBoost model en voorspelt sepsis associated delirium (SAD), getraind op MIMIC-IV ([Johnson et al., 2023](#1)). Het model is gebaseerd op [Yang et al. (2023)](#2), voor gedetailleerde informatie kan je het best die paper lezen. De gebruikte features zijn terug te vinden in [appendix A](#A), of in de eerder genoemde paper. 

### Gebruik

Na het model getraind te hebben, hebben we een pickle bestand gemaakt van het xgboost object ('xgb.pkl'), deze is terug te vinden in de repository. Van de pickle valt weer een xgboost object te maken door:

```model = pickle.load(open("<path>\xgb.pkl", "rb"))```

Voor je een voorspelling kan maken, vereisen XGBoost modellen dat de data in een DMatrix staat. Pandas dataframes en Numpy arrays vallen hierin om te zetten d.m.v. `xgb.DMatrix(<data>)`, let wel dat de kolommen in de juiste volgorde staan: 

```
['age', 'weight', 'gender', 'temperature', 'eart_rate', 'resp_rate', 'spo2', 'sbp', 'dbp', 'mbp', 'wbc', 'hemoglobin', 'platelet', 'bun', 'cr', 'glu', 'Na', 'Cl', 'K', 'Mg', 'Ca', 'P', 'inr', 'pt', 'ptt', 'bicarbonate', 'aniongap', 'gcs', 'vent', 'crrt', 'vaso', 'seda', 'sofa_score', 'ami', 'ckd', 'copd', 'hyperte', 'dm', 'sad', 'aki', 'stroke', 'AISAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE', 'unknown', 'CCU', 'CVICU', 'MICU', 'MICU/SICU', 'NICU', 'SICU', 'TSICU']
```


## Counterfactuals

Voor counterfactuals hadden we aanvankelijk DiCE gebruikt, maar vanwege incompatibiliteit met XGBoost en de erg hoge runtime hebben we er voor gekozen zelf algoritmes te schrijven om counterfactuals te genereren.

### KDTree

De KDTree counterfactual gebruikt simpelweg de scikit-learn KDTree om de record(s) in de gekoppelde dataset te vinden die het dichtst bij de oorspronkelijke record zitten, maar wel een andere voorspelling hebben. Details zijn te vinden in de docstrings.

### Genetic

Het genetisch algoritme genereerd fictieve record(s) die erg op de oorspronkelijke record lijken, maar wel een andere voorspelling hebben. Gedetailleerde informatie is te vinden in de docstring.

## SHAP en andere visualisaties.

Informatie hierover valt voornamelijk te vinden in 'overzicht-schermen.html'. In de repo is ook 'shap_html.ipynb' te vinden, hierin staat informatie over hoe de visualisaties omgezet worden naar html en/of png 

# Frontend en backend

a

a

a

a

a

a

a

a

a

a

a

a

a

a

a

a

a

a

# vervolgstappen

//iets over llms
//iets over enquetes ofzo peter schrijf jij maar wat leuks hier


# Bronnen

<a id="1"></a> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). *PhysioNet*. https://doi.org/10.13026/6mm1-ek67.

<a id="2"></a> Yang, Z., Hu, J., Hua, T., Zhang, J., Zhang, Z., & Yang, M. (2023). Development of a machine learning-based prediction model for sepsis-associated delirium in the intensive care unit. *Scientific Reports, 13*(1). https://doi.org/10.1038/s41598-023-38650-4

# Appendix

## <a id="A"></a> A

<table> <thead> <tr><th> </th><th> </th><th>Missing </th><th>Overall </th><th>NON-SAD </th><th>SAD </th></tr> </thead> <tbody> <tr><td>n </td><td> </td><td> </td><td>14620 </td><td>9230 </td><td>5390 </td></tr> <tr><td>age, mean (SD) </td><td> </td><td>0 </td><td>66.9 (15.9) </td><td>66.7 (15.8) </td><td>67.3 (16.1) </td></tr> <tr><td>weight, mean (SD) </td><td> </td><td>160 </td><td>83.1 (23.6) </td><td>83.1 (23.0) </td><td>83.2 (24.6) </td></tr> <tr><td>gender, n (%) </td><td>FEMALE </td><td>0 </td><td>8518 (58.3) </td><td>5416 (58.7) </td><td>3102 (57.6) </td></tr> <tr><td> </td><td>MALE </td><td> </td><td>6102 (41.7) </td><td>3814 (41.3) </td><td>2288 (42.4) </td></tr> <tr><td>race, n (%) </td><td>AISAN </td><td>0 </td><td>426 (2.9) </td><td>311 (3.4) </td><td>115 (2.1) </td></tr> <tr><td> </td><td>BLACK </td><td> </td><td>1266 (8.7) </td><td>766 (8.3) </td><td>500 (9.3) </td></tr> <tr><td> </td><td>HISPANIC </td><td> </td><td>557 (3.8) </td><td>360 (3.9) </td><td>197 (3.7) </td></tr> <tr><td> </td><td>OTHER </td><td> </td><td>642 (4.4) </td><td>417 (4.5) </td><td>225 (4.2) </td></tr> <tr><td> </td><td>WHITE </td><td> </td><td>9723 (66.5) </td><td>6372 (69.0) </td><td>3351 (62.2) </td></tr> <tr><td> </td><td>unknown </td><td> </td><td>2006 (13.7) </td><td>1004 (10.9) </td><td>1002 (18.6) </td></tr> <tr><td>first_careunit, n (%) </td><td>CCU </td><td>0 </td><td>1366 (9.3) </td><td>881 (9.5) </td><td>485 (9.0) </td></tr> <tr><td> </td><td>CVICU </td><td> </td><td>3461 (23.7) </td><td>2772 (30.0) </td><td>689 (12.8) </td></tr> <tr><td> </td><td>MICU </td><td> </td><td>3078 (21.1) </td><td>1601 (17.3) </td><td>1477 (27.4) </td></tr> <tr><td> </td><td>MICU/SICU</td><td> </td><td>2706 (18.5) </td><td>1780 (19.3) </td><td>926 (17.2) </td></tr> <tr><td> </td><td>NICU </td><td> </td><td>534 (3.7) </td><td>234 (2.5) </td><td>300 (5.6) </td></tr> <tr><td> </td><td>SICU </td><td> </td><td>1887 (12.9) </td><td>1112 (12.0) </td><td>775 (14.4) </td></tr> <tr><td> </td><td>TSICU </td><td> </td><td>1588 (10.9) </td><td>850 (9.2) </td><td>738 (13.7) </td></tr> <tr><td>temperature, mean (SD) </td><td> </td><td>48 </td><td>36.7 (0.8) </td><td>36.7 (0.8) </td><td>36.8 (0.9) </td></tr> <tr><td>heart_rate, mean (SD) </td><td> </td><td>1 </td><td>89.7 (20.3) </td><td>88.2 (19.6) </td><td>92.3 (21.1) </td></tr> <tr><td>resp_rate, mean (SD) </td><td> </td><td>24 </td><td>19.6 (6.0) </td><td>19.1 (5.9) </td><td>20.6 (6.1) </td></tr> <tr><td>spo2, mean (SD) </td><td> </td><td>5 </td><td>97.1 (4.0) </td><td>97.3 (3.7) </td><td>96.7 (4.3) </td></tr> <tr><td>sbp, mean (SD) </td><td> </td><td>6 </td><td>120.3 (23.9) </td><td>119.6 (23.1) </td><td>121.4 (25.1) </td></tr> <tr><td>dbp, mean (SD) </td><td> </td><td>15 </td><td>66.5 (17.7) </td><td>65.7 (16.8) </td><td>67.9 (19.0) </td></tr> <tr><td>mbp, mean (SD) </td><td> </td><td>14 </td><td>81.5 (17.8) </td><td>81.0 (17.0) </td><td>82.5 (19.0) </td></tr> <tr><td>wbc, mean (SD) </td><td> </td><td>115 </td><td>13.1 (8.1) </td><td>12.8 (7.8) </td><td>13.7 (8.4) </td></tr> <tr><td>hemoglobin, mean (SD) </td><td> </td><td>96 </td><td>10.3 (2.2) </td><td>10.3 (2.1) </td><td>10.5 (2.3) </td></tr> <tr><td>platelet, mean (SD) </td><td> </td><td>102 </td><td>191.5 (106.0)</td><td>190.7 (105.3)</td><td>192.9 (107.1)</td></tr> <tr><td>bun, mean (SD) </td><td> </td><td>60 </td><td>28.2 (22.9) </td><td>26.1 (21.0) </td><td>31.7 (25.6) </td></tr> <tr><td>cr, mean (SD) </td><td> </td><td>56 </td><td>1.5 (1.5) </td><td>1.4 (1.5) </td><td>1.6 (1.6) </td></tr> <tr><td>glu, mean (SD) </td><td> </td><td>66 </td><td>150.2 (74.4) </td><td>144.9 (66.5) </td><td>159.3 (85.5) </td></tr> <tr><td>Na, mean (SD) </td><td> </td><td>50 </td><td>137.4 (5.5) </td><td>136.9 (5.0) </td><td>138.2 (6.2) </td></tr> <tr><td>Cl, mean (SD) </td><td> </td><td>51 </td><td>103.8 (6.7) </td><td>103.8 (6.3) </td><td>103.9 (7.3) </td></tr> <tr><td>K, mean (SD) </td><td> </td><td>59 </td><td>4.3 (0.8) </td><td>4.3 (0.8) </td><td>4.3 (0.9) </td></tr> <tr><td>Mg, mean (SD) </td><td> </td><td>608 </td><td>2.0 (0.5) </td><td>2.0 (0.5) </td><td>2.0 (0.5) </td></tr> <tr><td>Ca, mean (SD) </td><td> </td><td>1399 </td><td>8.2 (0.9) </td><td>8.2 (0.8) </td><td>8.2 (0.9) </td></tr> <tr><td>P, mean (SD) </td><td> </td><td>1341 </td><td>3.8 (1.5) </td><td>3.7 (1.3) </td><td>4.0 (1.7) </td></tr> <tr><td>inr, mean (SD) </td><td> </td><td>1609 </td><td>1.5 (0.8) </td><td>1.5 (0.7) </td><td>1.6 (0.8) </td></tr> <tr><td>pt, mean (SD) </td><td> </td><td>1578 </td><td>17.0 (9.8) </td><td>16.7 (8.8) </td><td>17.5 (11.3) </td></tr> <tr><td>ptt, mean (SD) </td><td> </td><td>1658 </td><td>37.8 (22.4) </td><td>37.2 (21.4) </td><td>39.0 (24.0) </td></tr> <tr><td>bicarbonate, mean (SD) </td><td> </td><td>57 </td><td>22.2 (4.6) </td><td>22.5 (4.3) </td><td>21.7 (5.1) </td></tr> <tr><td>aniongap, mean (SD) </td><td> </td><td>64 </td><td>15.0 (4.6) </td><td>14.4 (4.2) </td><td>16.0 (4.9) </td></tr> <tr><td>gcs, mean (SD) </td><td> </td><td>2 </td><td>14.2 (2.4) </td><td>14.3 (2.4) </td><td>14.1 (2.3) </td></tr> <tr><td>vent, n (%) </td><td>FALSE </td><td>0 </td><td>8023 (54.9) </td><td>5974 (64.7) </td><td>2049 (38.0) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>6597 (45.1) </td><td>3256 (35.3) </td><td>3341 (62.0) </td></tr> <tr><td>crrt, n (%) </td><td>FALSE </td><td>0 </td><td>14364 (98.2) </td><td>9152 (99.2) </td><td>5212 (96.7) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>256 (1.8) </td><td>78 (0.8) </td><td>178 (3.3) </td></tr> <tr><td>vaso, n (%) </td><td>FALSE </td><td>0 </td><td>7482 (51.2) </td><td>4992 (54.1) </td><td>2490 (46.2) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>7138 (48.8) </td><td>4238 (45.9) </td><td>2900 (53.8) </td></tr> <tr><td>seda, n (%) </td><td>FALSE </td><td>0 </td><td>7962 (54.5) </td><td>4968 (53.8) </td><td>2994 (55.5) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>6658 (45.5) </td><td>4262 (46.2) </td><td>2396 (44.5) </td></tr> <tr><td>sofa_score, mean (SD) </td><td> </td><td>0 </td><td>3.6 (1.9) </td><td>3.4 (1.7) </td><td>3.9 (2.2) </td></tr> <tr><td>ami, n (%) </td><td>FALSE </td><td>0 </td><td>12976 (88.8) </td><td>8293 (89.8) </td><td>4683 (86.9) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>1644 (11.2) </td><td>937 (10.2) </td><td>707 (13.1) </td></tr> <tr><td>ckd, n (%) </td><td>FALSE </td><td>0 </td><td>11680 (79.9) </td><td>7417 (80.4) </td><td>4263 (79.1) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>2940 (20.1) </td><td>1813 (19.6) </td><td>1127 (20.9) </td></tr> <tr><td>copd, n (%) </td><td>FALSE </td><td>0 </td><td>14088 (96.4) </td><td>8944 (96.9) </td><td>5144 (95.4) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>532 (3.6) </td><td>286 (3.1) </td><td>246 (4.6) </td></tr> <tr><td>hyperte, n (%) </td><td>FALSE </td><td>0 </td><td>8322 (56.9) </td><td>5158 (55.9) </td><td>3164 (58.7) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>6298 (43.1) </td><td>4072 (44.1) </td><td>2226 (41.3) </td></tr> <tr><td>dm, n (%) </td><td>FALSE </td><td>0 </td><td>11962 (81.8) </td><td>7477 (81.0) </td><td>4485 (83.2) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>2658 (18.2) </td><td>1753 (19.0) </td><td>905 (16.8) </td></tr> <tr><td>sad, n (%) </td><td>NON-SAD </td><td>0 </td><td>9230 (63.1) </td><td>9230 (100.0) </td><td> </td></tr> <tr><td> </td><td>SAD </td><td> </td><td>5390 (36.9) </td><td> </td><td>5390 (100.0) </td></tr> <tr><td>aki, n (%) </td><td>FALSE </td><td>0 </td><td>6462 (44.2) </td><td>4541 (49.2) </td><td>1921 (35.6) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>8158 (55.8) </td><td>4689 (50.8) </td><td>3469 (64.4) </td></tr> <tr><td>stroke, n (%) </td><td>FALSE </td><td>0 </td><td>13479 (92.2) </td><td>8777 (95.1) </td><td>4702 (87.2) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>1141 (7.8) </td><td>453 (4.9) </td><td>688 (12.8) </td></tr> <tr><td> </td><td>TRUE </td><td> </td><td>1852 (12.7) </td><td>703 (7.6) </td><td>1149 (21.3) </td></tr> </tbody> </table>