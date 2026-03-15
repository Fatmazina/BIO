
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import euclidean


# ─────────────────────────────────────────────────────────────────
#  CLASSE PRINCIPALE : FaceVerificationSystem
# ─────────────────────────────────────────────────────────────────

class FaceVerificationSystem:

    def __init__(self, cascade_path: str = None):
        """
        Initialisation du détecteur de visage Viola-Jones.
        Si cascade_path n'est pas fourni, utilise le fichier OpenCV par défaut.
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise FileNotFoundError(
                f"Impossible de charger la cascade Haar : {cascade_path}\n"
                "Installez opencv-python ou fournissez le chemin correct."
            )

        self.reference_features = None   # histogramme LBP de référence
        self.reference_image    = None   # image originale de référence (pour affichage)
        self.reference_face     = None   # région du visage détectée (pour affichage)

        print("[INFO] FaceVerificationSystem initialisé.")

    # ─────────────────────────────────────────────────────────────
    #  1. Détection de visage (Viola-Jones)
    # ─────────────────────────────────────────────────────────────

    def detect_face(self, image: np.ndarray):
        """
        Détecte le visage le plus grand dans l'image.

        Paramètres
        ----------
        image : np.ndarray  Image BGR (lue avec cv2.imread)

        Retourne
        --------
        face_region : np.ndarray  Image recadrée sur le visage (niveaux de gris)
        coords      : tuple (x, y, w, h) coordonnées du rectangle
        None, None  si aucun visage détecté
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 5,
            minSize      = (30, 30)
        )

        if len(faces) == 0:
            print("[AVERTISSEMENT] Aucun visage détecté dans l'image.")
            return None, None

        # Conserver uniquement le plus grand visage (surface w × h maximale)
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        face_region = gray[y:y + h, x:x + w]
        return face_region, (x, y, w, h)

    # ─────────────────────────────────────────────────────────────
    #  2. Extraction LBP
    # ─────────────────────────────────────────────────────────────

    def _compute_lbp(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Calcule la carte LBP d'une image en niveaux de gris.
        Pour chaque pixel (hors bords) : compare avec ses 8 voisins immédiats.
          voisin ≥ centre → bit à 1, sinon → bit à 0.
        """
        rows, cols = gray_face.shape
        lbp_image  = np.zeros((rows, cols), dtype=np.uint8)

        # Décalages des 8 voisins (sens horaire depuis le voisin du haut-gauche)
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     ( 0,  1),
                     ( 1,  1), ( 1,  0), ( 1, -1),
                     ( 0, -1)]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = int(gray_face[i, j])
                code   = 0
                for bit, (di, dj) in enumerate(neighbors):
                    if int(gray_face[i + di, j + dj]) >= center:
                        code |= (1 << bit)
                lbp_image[i, j] = code

        return lbp_image

    def extract_lbp_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extrait le vecteur de caractéristiques LBP d'une image de visage.

        Étapes :
          1. Redimensionnement à 128×128 pixels
          2. Calcul de la carte LBP
          3. Histogramme normalisé (256 bins)

        Retourne
        --------
        histogram : np.ndarray de forme (256,), valeurs entre 0 et 1
        """
        # 1. Redimensionnement
        face_resized = cv2.resize(face_image, (128, 128))

        # 2. Carte LBP
        lbp_map = self._compute_lbp(face_resized)

        # 3. Histogramme normalisé (256 bins, valeurs 0-255)
        histogram, _ = np.histogram(lbp_map.ravel(), bins=256, range=(0, 256))
        histogram     = histogram.astype(np.float64)
        total         = histogram.sum()
        if total > 0:
            histogram /= total   # normalisation

        return histogram

    # ─────────────────────────────────────────────────────────────
    #  3. Enregistrement de la référence
    # ─────────────────────────────────────────────────────────────

    def setup_reference(self, image_path: str) -> bool:
        """
        Charge l'image de référence, détecte le visage,
        extrait et mémorise son histogramme LBP.

        Retourne True si réussi, False sinon.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERREUR] Impossible de lire l'image : {image_path}")
            return False

        face, coords = self.detect_face(image)
        if face is None:
            print("[ERREUR] Aucun visage trouvé dans l'image de référence.")
            return False

        self.reference_features = self.extract_lbp_features(face)
        self.reference_image    = image
        self.reference_face     = coords

        print(f"[INFO] Référence enregistrée depuis '{image_path}'  "
              f"(visage détecté : {coords})")
        return True

    # ─────────────────────────────────────────────────────────────
    #  4. Vérification
    # ─────────────────────────────────────────────────────────────

    def verify_face(self, image_path: str, threshold: float = 0.75):
        """
        Vérifie si le visage dans image_path correspond à la référence.

        Paramètres
        ----------
        image_path : str    Chemin vers l'image à tester
        threshold  : float  Seuil de similarité (défaut = 0.75)

        Retourne
        --------
        dict avec : similarity, distance, verdict, image, face_coords
        """
        if self.reference_features is None:
            raise RuntimeError("Aucune référence enregistrée. Appelez setup_reference() d'abord.")

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERREUR] Impossible de lire l'image : {image_path}")
            return None

        face, coords = self.detect_face(image)
        if face is None:
            return {
                "similarity" : 0.0,
                "distance"   : float("inf"),
                "verdict"    : "NO MATCH",
                "image"      : image,
                "face_coords": None,
            }

        test_features = self.extract_lbp_features(face)

        # Distance Euclidienne + conversion en similarité
        dist       = euclidean(self.reference_features, test_features)
        similarity = 1.0 - dist

        verdict = "MATCH" if similarity >= threshold else "NO MATCH"

        return {
            "similarity" : round(similarity, 4),
            "distance"   : round(dist, 4),
            "verdict"    : verdict,
            "image"      : image,
            "face_coords": coords,
        }


# ─────────────────────────────────────────────────────────────────
#  AFFICHAGE DES RÉSULTATS
# ─────────────────────────────────────────────────────────────────

def display_results(system: FaceVerificationSystem, result: dict,
                    ref_path: str, test_path: str):
    """
    Affiche côte à côte :
      - l'image de référence avec le rectangle de détection
      - l'image de test avec le rectangle et le statut (Match / No Match)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Image de référence ──
    ref_rgb = cv2.cvtColor(system.reference_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(ref_rgb)
    axes[0].set_title("Image de référence", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    if system.reference_face:
        x, y, w, h = system.reference_face
        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=2, edgecolor="#1D9E75", facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x, y - 8, "Référence", color="#1D9E75",
                     fontsize=10, fontweight="bold")

    # ── Image de test ──
    test_rgb = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB)
    axes[1].imshow(test_rgb)

    is_match  = result["verdict"] == "MATCH"
    color     = "#1D9E75" if is_match else "#D85A30"
    label     = f"✓ MATCH" if is_match else "✗ NO MATCH"
    sim_label = f"Similarité : {result['similarity']:.4f}  |  Seuil : 0.75"

    axes[1].set_title(f"Image de test — {label}", fontsize=13,
                      fontweight="bold", color=color)
    axes[1].axis("off")

    if result["face_coords"]:
        x, y, w, h = result["face_coords"]
        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=2, edgecolor=color, facecolor="none")
        axes[1].add_patch(rect)
        axes[1].text(x, y - 8, label, color=color,
                     fontsize=10, fontweight="bold")

    fig.suptitle(sim_label, fontsize=12, color="#444444", y=0.02)
    plt.tight_layout()
    plt.savefig("tp03_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Résultat sauvegardé dans 'tp03_result.png'")


def print_report(result: dict, threshold: float = 0.75):
    """Affiche le rapport dans la console."""
    print("\n" + "=" * 50)
    print("  TP03 — RAPPORT DE VÉRIFICATION FACIALE")
    print("=" * 50)
    print(f"  Distance Euclidienne : {result['distance']}")
    print(f"  Similarité           : {result['similarity']}")
    print(f"  Seuil                : {threshold}")
    print("-" * 50)
    v = result["verdict"]
    marker = "✓" if v == "MATCH" else "✗"
    print(f"  Décision : {marker}  {v}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────
#  PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ─── Modifier ces chemins avec vos vraies images ───
    REFERENCE_IMAGE = "face_reference.jpg"   # votre visage de référence
    TEST_IMAGE      = "face_test.jpg"        # visage à vérifier
    THRESHOLD       = 0.75                   # seuil expérimental (ajustable)
    # ──────────────────────────────────────────────────

    import os
    if not os.path.exists(REFERENCE_IMAGE) or not os.path.exists(TEST_IMAGE):
        print("[ERREUR] Fichiers images introuvables.")
        print(f"  Attendu : '{REFERENCE_IMAGE}' et '{TEST_IMAGE}' dans le même dossier.")
        exit(1)

    # 1. Initialisation du système
    system = FaceVerificationSystem()

    # 2. Enregistrement du visage de référence
    ok = system.setup_reference(REFERENCE_IMAGE)
    if not ok:
        exit(1)

    # 3. Vérification du visage de test
    result = system.verify_face(TEST_IMAGE, threshold=THRESHOLD)
    if result is None:
        exit(1)

    # 4. Affichage console
    print_report(result, threshold=THRESHOLD)

    # 5. Affichage visuel
    display_results(system, result, REFERENCE_IMAGE, TEST_IMAGE)
