import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from scipy.ndimage import maximum_filter
def convolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    k_height, k_width = kernel.shape
    img_height, img_width = image.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image)
    for y in range(img_height):
        for x in range(img_width):
            roi = padded_image[y:y + k_height, x:x + k_width]
            output[y, x] = np.sum(roi * kernel)
    return output
def gaussian_kernel(size=5, sigma=1.4):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)
def create_log_kernel(size=9, sigma=2.0):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    term1 = -1 / (np.pi * sigma**4)
    term2 = 1 - (xx**2 + yy**2) / (2 * sigma**2)
    term3 = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = term1 * term2 * term3
    return kernel
def main():
    print("--- Starting Image Processing Pipeline ---")
    print("Loading and preparing image...")
    image_rgb = data.chelsea().astype(np.float32)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_gray, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')
    plt.show()
    print("Executing Stage 1: Edge Detection (from scratch)...")
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    grad_x = convolve2d(image_gray, sobel_x_kernel)
    grad_y = convolve2d(image_gray, sobel_y_kernel)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_magnitude = (edge_magnitude / np.max(edge_magnitude) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(grad_x, cmap='gray'); axes[0].set_title('Sobel Gx'); axes[0].axis('off')
    axes[1].imshow(grad_y, cmap='gray'); axes[1].set_title('Sobel Gy'); axes[1].axis('off')
    axes[2].imshow(edge_magnitude, cmap='gray'); axes[2].set_title('NumPy Edge Magnitude'); axes[2].axis('off')
    plt.suptitle("Stage 1: Edge Detection Results", fontsize=16)
    plt.show()
    print("Executing Stage 2: Feature Enhancement (from scratch)...")
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian_img = convolve2d(image_gray, laplacian_kernel)
    sharpened_image = np.clip(image_gray - laplacian_img, 0, 255)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(image_gray, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(sharpened_image, cmap='gray'); axes[1].set_title('NumPy Sharpened'); axes[1].axis('off')
    plt.suptitle("Stage 2: Sharpening Results", fontsize=16)
    plt.show()
    print("Executing Stage 3a: Corner Detection (from scratch)...")
    gauss_kernel = gaussian_kernel(5, 1.4)
    S_x2 = convolve2d(grad_x**2, gauss_kernel)
    S_y2 = convolve2d(grad_y**2, gauss_kernel)
    S_xy = convolve2d(grad_x * grad_y, gauss_kernel)
    
    k = 0.04
    det_M = (S_x2 * S_y2) - (S_xy**2)
    trace_M = S_x2 + S_y2
    harris_response = det_M - k * (trace_M**2)
    
    corner_img = np.copy(image_rgb)
    corners = np.argwhere(harris_response > 0.01 * np.max(harris_response))
    for y, x in corners:
        cv2.circle(corner_img, (x, y), 3, (255, 0, 0), 1)

    print("Executing Stage 3b: Blob Detection (from scratch)...")
    log_kernel = create_log_kernel(size=15, sigma=4.0)
    log_image = convolve2d(image_gray, log_kernel)
    local_maxima = maximum_filter(log_image, size=15)
    maxima_points = (log_image == local_maxima) & (log_image > log_image.mean() * 5)
    blobs = np.argwhere(maxima_points)
    
    blob_img = np.copy(image_rgb)
    blob_radii = 4.0 * np.sqrt(2)
    for y, x in blobs:
        cv2.circle(blob_img, (x, y), int(blob_radii), (0, 255, 0), 2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(corner_img.astype(np.uint8)); axes[0].set_title('NumPy Harris Corners'); axes[0].axis('off')
    axes[1].imshow(blob_img.astype(np.uint8)); axes[1].set_title('NumPy LoG Blobs'); axes[1].axis('off')
    plt.suptitle("Stage 3: Interest Point Detection", fontsize=16)
    plt.show()
    print("Executing Final Stage: Comparison with OpenCV built-ins...")
    cv_edges = cv2.Canny(image_gray.astype(np.uint8), 100, 200)    
    cv_harris_response = cv2.cornerHarris(image_gray, 2, 3, 0.04)
    cv_corner_img = np.copy(image_rgb)
    cv_corner_img[cv_harris_response > 0.01 * cv_harris_response.max()] = [255, 0, 0]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].imshow(edge_magnitude, cmap='gray'); axes[0, 0].set_title('Our NumPy Edges (Sobel)')
    axes[0, 1].imshow(cv_edges, cmap='gray'); axes[0, 1].set_title('OpenCV Edges (Canny)')
    axes[1, 0].imshow(corner_img.astype(np.uint8)); axes[1, 0].set_title('Our NumPy Corners')
    axes[1, 1].imshow(cv_corner_img.astype(np.uint8)); axes[1, 1].set_title('OpenCV Harris Corners')    
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    plt.suptitle("Final Comparison: NumPy vs. OpenCV", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()    
    print("--- Pipeline Finished ---")
if __name__ == "__main__":
    main()