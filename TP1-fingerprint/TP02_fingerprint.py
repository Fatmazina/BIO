import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.metrics import structural_similarity as compare_ssim


# ─────────────────────────────────────────────
#  UTILITAIRES COMMUNS
# ─────────────────────────────────────────────

def load_image_pil(path: str) -> Image.Image:
    """Charge une image depuis le disque."""
    return Image.open(path).convert("RGB")


def pil_to_gray_array(img: Image.Image) -> np.ndarray:
    """Convertit une image PIL en tableau numpy niveaux de gris (uint8)."""
    return np.array(img.convert("L"))


# ─────────────────────────────────────────────
#  PARTIE A — ORB (méthode locale)
# ─────────────────────────────────────────────

def method_orb(path1: str, path2: str, ratio_threshold: float = 0.15):
    """
    Méthode ORB : détection et correspondance de points clés.
    Décision : ACCEPTÉE si match_ratio > ratio_threshold (0.15)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV requis : pip install opencv-python")

    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Impossible de lire l'une des images.")

    # Détection ORB
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return {"method": "ORB", "matches": 0, "ratio": 0.0, "verdict": "REJETÉE",
                "detail": "Aucun descripteur détecté."}

    # Correspondance par distance de Hamming (BFMatcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Filtre de Lowe (ratio test)
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    total_kp = min(len(kp1), len(kp2))
    match_ratio = len(good_matches) / total_kp if total_kp > 0 else 0.0
    verdict = "ACCEPTÉE" if match_ratio > ratio_threshold else "REJETÉE"

    # Visualisation
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 5))
    plt.title(f"ORB — {len(good_matches)} correspondances · ratio={match_ratio:.3f} · {verdict}")
    plt.imshow(img_matches, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("orb_result.png", dpi=150)
    plt.show()

    return {
        "method": "ORB",
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "good_matches": len(good_matches),
        "match_ratio": round(match_ratio, 4),
        "threshold": ratio_threshold,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
#  PARTIE B — FFT (méthode globale fréquentielle)
# ─────────────────────────────────────────────

def method_fft(path1: str, path2: str, similarity_threshold: float = 0.85):
    """
    Méthode FFT : comparaison dans le domaine fréquentiel.
    Décision : ACCEPTÉE si cosine_similarity > similarity_threshold (0.85)
    """
    def preprocess_fft(path: str) -> np.ndarray:
        img = Image.open(path).convert("L").resize((300, 300))
        arr = np.array(img, dtype=np.float64)
        # Transformée de Fourier + magnitude
        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        # Normalisation log pour réduire la dynamique
        magnitude = np.log1p(magnitude)
        # Aplatir et normaliser
        flat = magnitude.flatten()
        norm = np.linalg.norm(flat)
        return flat / norm if norm > 0 else flat

    feat1 = preprocess_fft(path1)
    feat2 = preprocess_fft(path2)

    cosine_sim = float(np.dot(feat1, feat2))  # vecteurs déjà normalisés
    verdict = "ACCEPTÉE" if cosine_sim > similarity_threshold else "REJETÉE"

    # Visualisation des spectres
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, path, title in zip(axes, [path1, path2], ["Empreinte 1 — spectre FFT", "Empreinte 2 — spectre FFT"]):
        img = Image.open(path).convert("L").resize((300, 300))
        arr = np.array(img, dtype=np.float64)
        mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(arr))))
        ax.imshow(mag, cmap="inferno")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"FFT — similarité cosinus={cosine_sim:.4f} · {verdict}", fontsize=13)
    plt.tight_layout()
    plt.savefig("fft_result.png", dpi=150)
    plt.show()

    return {
        "method": "FFT",
        "cosine_similarity": round(cosine_sim, 4),
        "threshold": similarity_threshold,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
#  PARTIE C — Gabor (texture orientée)
# ─────────────────────────────────────────────

def method_gabor(path1: str, path2: str, distance_threshold: float = 50.0):
    """
    Méthode Gabor : extraction de caractéristiques de texture orientée.
    Orientations : 0°, 45°, 90°, 135° — fréquence = 0.3
    Décision : ACCEPTÉE si distance euclidienne < distance_threshold (50.0)
    """
    try:
        from skimage.filters import gabor
    except ImportError:
        raise ImportError("scikit-image requis : pip install scikit-image")

    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
    frequency = 0.3

    def extract_gabor_features(path: str) -> np.ndarray:
        img = Image.open(path).convert("L").resize((300, 300))
        arr = np.array(img, dtype=np.float64) / 255.0
        features = []
        for theta in orientations:
            filt_real, filt_imag = gabor(arr, frequency=frequency, theta=theta)
            magnitude = np.sqrt(filt_real ** 2 + filt_imag ** 2)
            features.append(magnitude.mean())
            features.append(magnitude.std())
        return np.array(features)

    feat1 = extract_gabor_features(path1)
    feat2 = extract_gabor_features(path2)

    distance = float(np.linalg.norm(feat1 - feat2))
    verdict = "ACCEPTÉE" if distance < distance_threshold else "REJETÉE"

    # Visualisation des réponses Gabor pour la 1ère image
    from skimage.filters import gabor as skgabor
    img_arr = np.array(Image.open(path1).convert("L").resize((300, 300)), dtype=np.float64) / 255.0
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, theta in enumerate(orientations):
        angle_deg = int(np.degrees(theta))
        r, im_ = skgabor(img_arr, frequency=frequency, theta=theta)
        mag = np.sqrt(r ** 2 + im_ ** 2)
        axes[0, i].imshow(r, cmap="gray")
        axes[0, i].set_title(f"Réel {angle_deg}°", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(mag, cmap="hot")
        axes[1, i].set_title(f"Magnitude {angle_deg}°", fontsize=9)
        axes[1, i].axis("off")
    fig.suptitle(f"Gabor — distance={distance:.2f} · seuil={distance_threshold} · {verdict}", fontsize=13)
    plt.tight_layout()
    plt.savefig("gabor_result.png", dpi=150)
    plt.show()

    return {
        "method": "Gabor",
        "features_img1": feat1.tolist(),
        "features_img2": feat2.tolist(),
        "euclidean_distance": round(distance, 4),
        "threshold": distance_threshold,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
#  PARTIE D — SSIM (similarité structurelle)
# ─────────────────────────────────────────────

def preprocess(image_path: str) -> np.ndarray:
    """
    Prétraitement obligatoire pour la méthode SSIM :
    1. Conversion en niveaux de gris
    2. Redimensionnement (300×300)
    3. Égalisation d'histogramme
    4. Binarisation (seuil = 128)
    5. Extraction des contours (FIND_EDGES)
    """
    img = Image.open(image_path).convert("L")          # 1. Niveaux de gris
    img = img.resize((300, 300))                        # 2. Redimensionnement
    img = ImageOps.equalize(img)                        # 3. Égalisation histogramme
    img = img.point(lambda p: 255 if p >= 128 else 0)  # 4. Binarisation seuil=128
    img = img.filter(ImageFilter.FIND_EDGES)            # 5. Extraction contours
    return np.array(img)


def method_ssim(path1: str, path2: str, ssim_threshold: float = 0.75):
    """
    Méthode SSIM : comparaison de la similarité structurelle.
    Décision : ACCEPTÉE si SSIM ≥ ssim_threshold (0.75)
    """
    img1 = preprocess(path1)
    img2 = preprocess(path2)

    similarity = compare_ssim(img1, img2, data_range=255)
    verdict = "ACCEPTÉE" if similarity >= ssim_threshold else "REJETÉE"

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Empreinte 1 — prétraitée", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("Empreinte 2 — prétraitée", fontsize=11)
    axes[1].axis("off")
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)
    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Différence absolue", fontsize=11)
    axes[2].axis("off")
    fig.suptitle(f"SSIM — score={similarity:.4f} · seuil={ssim_threshold} · {verdict}", fontsize=13)
    plt.tight_layout()
    plt.savefig("ssim_result.png", dpi=150)
    plt.show()

    return {
        "method": "SSIM",
        "ssim_score": round(similarity, 4),
        "threshold": ssim_threshold,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
#  RAPPORT FINAL
# ─────────────────────────────────────────────

def print_report(results: list):
    """Affiche un tableau récapitulatif de toutes les méthodes."""
    print("\n" + "=" * 55)
    print("  TP02 — RAPPORT DE RECONNAISSANCE D'EMPREINTE DIGITALE")
    print("=" * 55)
    print(f"  {'Méthode':<10} {'Score / Valeur':<25} {'Décision'}")
    print("-" * 55)

    accepted = 0
    for r in results:
        m = r["method"]
        if m == "ORB":
            score_str = f"ratio={r['match_ratio']:.3f}"
        elif m == "FFT":
            score_str = f"cosine={r['cosine_similarity']:.4f}"
        elif m == "Gabor":
            score_str = f"distance={r['euclidean_distance']:.2f}"
        else:
            score_str = f"SSIM={r['ssim_score']:.4f}"

        v = r["verdict"]
        if v == "ACCEPTÉE":
            accepted += 1
        marker = "✓" if v == "ACCEPTÉE" else "✗"
        print(f"  {m:<10} {score_str:<25} {marker} {v}")

    print("-" * 55)
    final = "IDENTITÉ CONFIRMÉE" if accepted >= 3 else "IDENTITÉ NON CONFIRMÉE"
    print(f"  Résultat global : {accepted}/4 méthodes acceptées")
    print(f"  >> {final}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ─── Modifier ces chemins avec vos vraies images ───
    IMAGE_1 = "fingerprint1.jpg"   # empreinte de référence
    IMAGE_2 = "fingerprint2.jpg"   # empreinte à tester
    # ──────────────────────────────────────────────────

    if not os.path.exists(IMAGE_1) or not os.path.exists(IMAGE_2):
        print("[ERREUR] Fichiers images introuvables.")
        print(f"  Attendu : '{IMAGE_1}' et '{IMAGE_2}' dans le même dossier.")
        exit(1)

    print(f"\nComparaison de '{IMAGE_1}' et '{IMAGE_2}'\n")

    results = []

    print("[A] ORB en cours...")
    results.append(method_orb(IMAGE_1, IMAGE_2))

    print("[B] FFT en cours...")
    results.append(method_fft(IMAGE_1, IMAGE_2))

    print("[C] Gabor en cours...")
    results.append(method_gabor(IMAGE_1, IMAGE_2))

    print("[D] SSIM en cours...")
    results.append(method_ssim(IMAGE_1, IMAGE_2))

    print_report(results)
