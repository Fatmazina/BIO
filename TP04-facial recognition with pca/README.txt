TP04 - Reconnaissance Faciale par PCA
======================================

ÉTAPES :
--------
1. Mets 5 photos de ton visage dans le dossier :  dataset/moi/
   (nomme-les photo1.jpg, photo2.jpg, ... photo5.jpg)

2. Mets ta photo de test dans ce dossier et renomme-la :  face_test.jpg

3. Installe les dépendances :
   pip install opencv-python numpy matplotlib

4. Lance le script :
   python TP04_face_recognition_pca.py

RÉSULTATS :
-----------
- Console    : distance, identité prédite, décision MATCH / NO MATCH
- eigenfaces.png  : les vecteurs propres calculés
- tp04_result.png : image test avec rectangle et résultat
