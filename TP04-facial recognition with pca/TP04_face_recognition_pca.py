"""
TP04 - Reconnaissance Faciale par PCA (Eigenfaces) et Viola-Jones
==================================================================
Objectifs :
  1. Détection de visage par Viola-Jones (Cascades de Haar)
  2. Construction d'un modèle PCA (Eigenfaces) sur une base d'images
  3. Projection des visages dans le sous-espace PCA
  4. Comparaison par distance Euclidienne
  5. Prise de décision par seuillage

Structure du dossier d'entraînement attendue :
  dataset/
    person1/
        img1.jpg
        img2.jpg
        ...
    person2/
        img1.jpg
        ...
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


# ─────────────────────────────────────────────────────────────────
#  CLASSE PRINCIPALE : FaceRecognitionPCA
# ─────────────────────────────────────────────────────────────────

class FaceRecognitionPCA:

    def __init__(self, n_components: int = 30):
        """
        Initialise :
          - détecteur Viola-Jones
          - nombre de composantes principales (k)
          - variables internes (mean, eigenvectors, projections, labels)
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise FileNotFoundError("Impossible de charger haarcascade_frontalface_default.xml")

        self.n_components  = n_components
        self.mean          = None   # visage moyen
        self.eigenvectors  = None   # k vecteurs propres (eigenfaces)
        self.projections   = None   # projections PCA des images d'entraînement
        self.labels        = []     # label (nom de personne) pour chaque image
        self.label_names   = []     # liste des noms uniques

        print(f"[INFO] FaceRecognitionPCA initialisé (k={n_components})")

    # ─────────────────────────────────────────────────────────────
    #  1. Détection de visage
    # ─────────────────────────────────────────────────────────────

    def detect_face(self, image: np.ndarray):
        """
        Entrée  : image BGR
        Sortie  : visage détecté en niveaux de gris (100x100), coords
        Étapes  : conversion gris → detectMultiScale → plus grand visage → resize 100x100
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 3,
            minSize      = (20, 20)
        )

        if len(faces) == 0:
            return None, None

        largest  = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        face     = gray[y:y+h, x:x+w]
        face     = cv2.resize(face, (100, 100))
        return face, (x, y, w, h)

    # ─────────────────────────────────────────────────────────────
    #  2. Chargement du dataset
    # ─────────────────────────────────────────────────────────────

    def load_dataset(self, dataset_path: str):
        """
        Parcourt un dossier structuré par personne :
          dataset/
            person1/  img1.jpg ...
            person2/  img1.jpg ...

        Pour chaque image : détecte le visage, vectorise, stocke label.
        Retourne X (numpy array shape [N, 10000]), y (labels string)
        """
        X      = []
        y      = []
        failed = 0

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dossier introuvable : {dataset_path}")

        persons = sorted(os.listdir(dataset_path))
        self.label_names = []

        for person in persons:
            person_dir = os.path.join(dataset_path, person)
            if not os.path.isdir(person_dir):
                continue

            self.label_names.append(person)
            count = 0

            for filename in os.listdir(person_dir):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                img_path = os.path.join(person_dir, filename)
                img      = cv2.imread(img_path)
                if img is None:
                    continue

                face, _ = self.detect_face(img)
                if face is None:
                    failed += 1
                    continue

                X.append(face.flatten().astype(np.float64))
                y.append(person)
                count += 1

            print(f"[INFO]   {person} : {count} image(s) chargée(s)")

        print(f"[INFO] Dataset chargé : {len(X)} visages, {failed} échec(s) de détection")

        if len(X) == 0:
            raise ValueError("Aucun visage détecté dans le dataset !")

        return np.array(X), y

    # ─────────────────────────────────────────────────────────────
    #  3. Calcul PCA (Eigenfaces)
    # ─────────────────────────────────────────────────────────────

    def compute_pca(self, X: np.ndarray):
        """
        Étapes :
          1. Calcul de la moyenne
          2. Centrage (soustraction de la moyenne)
          3. Matrice de covariance
          4. Valeurs et vecteurs propres
          5. Tri décroissant par valeur propre
          6. Sélection des n_components premiers vecteurs propres

        Stocke : self.mean, self.eigenvectors
        """
        # 1. Visage moyen
        self.mean = np.mean(X, axis=0)

        # 2. Centrage
        X_centered = X - self.mean

        # 3. Matrice de covariance  (N x N au lieu de D x D pour l'efficacité)
        #    Astuce : cov = (1/N) * A * A^T  où A est la matrice centrée
        N   = X_centered.shape[0]
        cov = (1.0 / N) * (X_centered @ X_centered.T)

        # 4. Valeurs et vecteurs propres
        eigenvalues, eigenvectors_small = np.linalg.eigh(cov)

        # Reprojection dans l'espace original : u_i = A^T * v_i
        eigenvectors_full = X_centered.T @ eigenvectors_small

        # Normalisation des vecteurs propres
        for i in range(eigenvectors_full.shape[1]):
            norm = np.linalg.norm(eigenvectors_full[:, i])
            if norm > 0:
                eigenvectors_full[:, i] /= norm

        # 5. Tri décroissant
        idx            = np.argsort(eigenvalues)[::-1]
        eigenvalues    = eigenvalues[idx]
        eigenvectors_full = eigenvectors_full[:, idx]

        # 6. Sélection des k premiers
        k = min(self.n_components, eigenvectors_full.shape[1])
        self.eigenvectors = eigenvectors_full[:, :k]

        print(f"[INFO] PCA calculée : {k} composantes sélectionnées sur {N} images")

        # Variance expliquée
        total_var    = np.sum(eigenvalues)
        explained    = np.sum(eigenvalues[:k]) / total_var * 100 if total_var > 0 else 0
        print(f"[INFO] Variance expliquée par les {k} composantes : {explained:.1f}%")

    # ─────────────────────────────────────────────────────────────
    #  4. Projection dans l'espace PCA
    # ─────────────────────────────────────────────────────────────

    def project(self, face_vector: np.ndarray) -> np.ndarray:
        """
        Projette un vecteur de visage (aplati) dans l'espace PCA.
        Retourne le vecteur de coordonnées dans l'espace réduit.
        """
        centered = face_vector.astype(np.float64) - self.mean
        return self.eigenvectors.T @ centered

    def build_gallery(self, X: np.ndarray, y: list):
        """
        Projette toutes les images d'entraînement et stocke les projections.
        """
        self.projections = np.array([self.project(x) for x in X])
        self.labels      = y
        print(f"[INFO] Galerie construite : {len(self.projections)} projections")

    # ─────────────────────────────────────────────────────────────
    #  5. Reconnaissance
    # ─────────────────────────────────────────────────────────────

    def recognize(self, image_path: str, threshold: float = 3500.0):
        """
        Charge une image, détecte le visage, projette dans PCA,
        trouve le plus proche voisin dans la galerie.

        Retourne un dict :
          identity      : nom prédit
          min_distance  : distance minimale
          verdict       : MATCH / NO MATCH
          image         : image BGR originale
          face_coords   : (x, y, w, h)
        """
        if self.projections is None:
            raise RuntimeError("Modèle non entraîné. Appelez fit() d'abord.")

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERREUR] Impossible de lire : {image_path}")
            return None

        face, coords = self.detect_face(image)
        if face is None:
            print("[AVERTISSEMENT] Aucun visage détecté dans l'image de test.")
            return {
                "identity"    : "Inconnu",
                "min_distance": float("inf"),
                "verdict"     : "NO MATCH",
                "image"       : image,
                "face_coords" : None,
            }

        # Projection du visage test
        test_proj = self.project(face.flatten())

        # Distance Euclidienne avec toutes les projections de la galerie
        distances = np.linalg.norm(self.projections - test_proj, axis=1)

        min_idx      = np.argmin(distances)
        min_distance = distances[min_idx]
        identity     = self.labels[min_idx]
        verdict      = "MATCH" if min_distance < threshold else "NO MATCH"

        return {
            "identity"    : identity,
            "min_distance": round(float(min_distance), 2),
            "verdict"     : verdict,
            "image"       : image,
            "face_coords" : coords,
        }

    # ─────────────────────────────────────────────────────────────
    #  Méthode utilitaire : entraîner d'un coup
    # ─────────────────────────────────────────────────────────────

    def fit(self, dataset_path: str):
        """
        Charge le dataset, calcule la PCA, construit la galerie.
        Appel unique pour tout configurer.
        """
        print(f"\n[INFO] Chargement du dataset depuis '{dataset_path}'...")
        X, y = self.load_dataset(dataset_path)

        print("\n[INFO] Calcul de la PCA...")
        self.compute_pca(X)

        print("\n[INFO] Construction de la galerie...")
        self.build_gallery(X, y)

        print("\n[INFO] Modèle prêt.\n")


# ─────────────────────────────────────────────────────────────────
#  EXPÉRIMENTATIONS
# ─────────────────────────────────────────────────────────────────

def experiment_k_components(dataset_path: str, test_image_path: str,
                             threshold: float = 3500.0):
    """
    Expérimentation 1 : Étudier l'effet de k = 10, 20, 50
    """
    print("\n" + "=" * 60)
    print("  EXPÉRIMENTATION — Effet du nombre de composantes k")
    print("=" * 60)

    results = []
    for k in [10, 20, 50]:
        model = FaceRecognitionPCA(n_components=k)
        model.fit(dataset_path)
        result = model.recognize(test_image_path, threshold=threshold)
        if result:
            results.append((k, result["min_distance"], result["identity"], result["verdict"]))
            print(f"  k={k:>3} | distance={result['min_distance']:>10.2f} | "
                  f"identité={result['identity']:<15} | {result['verdict']}")

    print("=" * 60)
    return results


def experiment_threshold(dataset_path: str, test_images: list):
    """
    Expérimentation 2 : Tableau Distance vs Décision pour différents seuils
    """
    print("\n" + "=" * 60)
    print("  EXPÉRIMENTATION — Effet du seuil")
    print("=" * 60)

    model = FaceRecognitionPCA(n_components=30)
    model.fit(dataset_path)

    thresholds = [1000, 2000, 3000, 4000, 5000]
    print(f"  {'Image':<20} {'Distance':>10} | " +
          " | ".join([f"seuil={t}" for t in thresholds]))
    print("-" * 60)

    for img_path in test_images:
        result = model.recognize(img_path, threshold=99999)
        if result:
            dist = result["min_distance"]
            decisions = ["MATCH" if dist < t else "NO MATCH" for t in thresholds]
            name = os.path.basename(img_path)[:18]
            print(f"  {name:<20} {dist:>10.2f} | " + " | ".join(
                [f"{'OK' if d == 'MATCH' else 'X ':>8}" for d in decisions]))

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
#  AFFICHAGE
# ─────────────────────────────────────────────────────────────────

def display_eigenfaces(model: FaceRecognitionPCA, n_show: int = 8):
    """Affiche les n premières eigenfaces."""
    n = min(n_show, model.eigenvectors.shape[1])
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    for i in range(n):
        ef = model.eigenvectors[:, i].reshape(100, 100)
        ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-10) * 255
        axes[i].imshow(ef.astype(np.uint8), cmap="gray")
        axes[i].set_title(f"EF {i+1}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle("Eigenfaces (vecteurs propres)", fontsize=12)
    plt.tight_layout()
    plt.savefig("eigenfaces.png", dpi=150)
    plt.show()
    print("[INFO] Eigenfaces sauvegardées dans 'eigenfaces.png'")


def display_result(model: FaceRecognitionPCA, result: dict, threshold: float):
    """Affiche l'image test avec le rectangle, la distance et le verdict."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    img_rgb = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis("off")

    is_match = result["verdict"] == "MATCH"
    color    = "#1D9E75" if is_match else "#D85A30"
    label    = "MATCH" if is_match else "NO MATCH"

    if result["face_coords"]:
        x, y, w, h = result["face_coords"]
        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 10,
                f"{result['identity']}  |  d={result['min_distance']:.0f}",
                color=color, fontsize=11, fontweight="bold")

    ax.set_title(f"{label}  —  Identité : {result['identity']}\n"
                 f"Distance : {result['min_distance']:.2f}  |  Seuil : {threshold}",
                 fontsize=11, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig("tp04_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Résultat sauvegardé dans 'tp04_result.png'")


def print_report(result: dict, threshold: float):
    """Affiche le rapport dans la console."""
    print("\n" + "=" * 50)
    print("  TP04 — RAPPORT DE RECONNAISSANCE FACIALE PCA")
    print("=" * 50)
    print(f"  Identité prédite     : {result['identity']}")
    print(f"  Distance minimale    : {result['min_distance']}")
    print(f"  Seuil                : {threshold}")
    print("-" * 50)
    v      = result["verdict"]
    marker = "OK" if v == "MATCH" else "X"
    print(f"  Décision : [{marker}]  {v}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────
#  PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ─── À modifier selon votre structure de dossiers ───────────
    DATASET_PATH = "dataset"        # dossier contenant person1/, person2/, ...
    TEST_IMAGE   = "face_test.jpg"  # image à reconnaître
    N_COMPONENTS = 30               # nombre de composantes PCA
    THRESHOLD    = 3500.0           # seuil de distance (ajustable)
    # ────────────────────────────────────────────────────────────

    print(f"[DEBUG] Dossier actuel   : {os.getcwd()}")
    print(f"[DEBUG] Dataset existe   : {os.path.exists(DATASET_PATH)}")
    print(f"[DEBUG] Test existe      : {os.path.exists(TEST_IMAGE)}")

    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERREUR] Dossier dataset introuvable : '{DATASET_PATH}'")
        print("  Créez un dossier 'dataset/' avec des sous-dossiers par personne.")
        print("  Exemple : dataset/moi/  dataset/personne2/")
        exit(1)

    if not os.path.exists(TEST_IMAGE):
        print(f"\n[ERREUR] Image de test introuvable : '{TEST_IMAGE}'")
        exit(1)

    # 1. Création et entraînement du modèle
    model = FaceRecognitionPCA(n_components=N_COMPONENTS)
    model.fit(DATASET_PATH)

    # 2. Affichage des eigenfaces
    display_eigenfaces(model, n_show=8)

    # 3. Reconnaissance de l'image test
    result = model.recognize(TEST_IMAGE, threshold=THRESHOLD)
    if result is None:
        exit(1)

    # 4. Rapport console
    print_report(result, threshold=THRESHOLD)

    # 5. Affichage visuel
    display_result(model, result, threshold=THRESHOLD)

    # ── Expérimentations obligatoires ──
    print("\n[INFO] Lancement des expérimentations...")
    experiment_k_components(DATASET_PATH, TEST_IMAGE, threshold=THRESHOLD)
