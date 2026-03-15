from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# Create results folder
# -------------------------------------------------
os.makedirs("results", exist_ok=True)

# -------------------------------------------------
# Load original image
# -------------------------------------------------
img = Image.open("image.jpg")

# Save original
img.save("results/image_originale.png")

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.title("Image Originale")
plt.axis("off")
plt.show()




# =================================================
# PARTIE 2 — RESIZE
# =================================================
img_resize = img.resize((300,300))
img_resize.save("results/image_redimensionnee.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_resize)
plt.title("Resize")
plt.axis("off")
plt.show()


# =================================================
# PARTIE 3 — BRIGHTNESS
# =================================================
enhancer = ImageEnhance.Brightness(img)
img_bright = enhancer.enhance(1.5)
img_bright.save("results/image_luminosite_augmente.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_bright)
plt.title("Brightness +")
plt.axis("off")
plt.show()


# =================================================
# PARTIE 4 — GRAYSCALE
# =================================================
img_gray = img.convert("L")
img_gray.save("results/image_gris.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Gray")
plt.show()


# =================================================
# PARTIE 5 — BINARIZATION
# =================================================
threshold = 128
img_binary = img_gray.point(lambda p: 255 if p > threshold else 0)
img_binary.save("results/image_binarisee.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Gray")

plt.subplot(1,2,2)
plt.imshow(img_binary, cmap="gray")
plt.axis("off")
plt.title("Binary")
plt.show()


# =================================================
# PARTIE 6 — EDGE DETECTION
# =================================================
img_edges = img_gray.filter(ImageFilter.FIND_EDGES)
img_edges.save("results/image_contours.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Gray")

plt.subplot(1,2,2)
plt.imshow(img_edges, cmap="gray")
plt.axis("off")
plt.title("Edges")
plt.show()


# =================================================
# PARTIE 7 — GAUSSIAN BLUR
# =================================================
img_blur = img.filter(ImageFilter.GaussianBlur(radius=2))
img_blur.save("results/image_flou_gaussien.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img_blur)
plt.axis("off")
plt.title("Gaussian Blur")
plt.show()


# =================================================
# HISTOGRAM COMPARISON (Original vs Gray)
# =================================================

# Histogram of original image (RGB)
hist_color = img.histogram()

# Histogram of grayscale image
hist_gray = img_gray.histogram()

plt.figure(figsize=(10,5))

# ----- Original RGB histogram -----
plt.subplot(1,2,1)

plt.plot(hist_color[0:256], label="Red")
plt.plot(hist_color[256:512], label="Green")
plt.plot(hist_color[512:768], label="Blue")

plt.title("Histogram - Image Originale")
plt.xlabel("Intensity")
plt.ylabel("Pixels")
plt.legend()


# ----- Grayscale histogram -----
plt.subplot(1,2,2)

plt.plot(hist_gray)
plt.title("Histogram - Image Grise")
plt.xlabel("Intensity")
plt.ylabel("Pixels")

plt.show()



# =================================================
# PARTIE 9 — HISTOGRAM EQUALIZATION
# =================================================
img_eq = ImageOps.equalize(img_gray)
img_eq.save("results/image_egalisee.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Before")

plt.subplot(1,2,2)
plt.imshow(img_eq, cmap="gray")
plt.axis("off")
plt.title("After Equalization")
plt.show()
