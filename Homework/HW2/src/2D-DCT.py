import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# Implement 2D-DCT
def dct_2d(image):
    M, N = image.shape
    c = np.array([1 / np.sqrt(2) if i == 0 else 1 for i in range(max(M, N))])
    x = np.arange(M).reshape(-1, 1)
    y = np.arange(N).reshape(1, -1)

    dct = np.zeros((M, N))
    for u in tqdm(range(M), desc='2D-DCT'):
        for v in range(N):
            cos_x = np.cos((2 * x + 1) * u * np.pi / (2 * M))
            cos_y = np.cos((2 * y + 1) * v * np.pi / (2 * N))
            dct[u, v] = (2 / N) * c[u] * c[v] * np.sum(image * cos_x * cos_y)

    return dct


# Implement 2D-IDCT
def idct_2d(dct):
    M, N = dct.shape
    c = np.array([1 / np.sqrt(2) if i == 0 else 1 for i in range(max(M, N))])
    u = np.arange(M).reshape(-1, 1)
    v = np.arange(N).reshape(1, -1)

    idct = np.zeros((M, N))
    for x in tqdm(range(M), desc='2D-IDCT'):
        for y in range(N):
            cos_u = np.cos((2 * x + 1) * u * np.pi / (2 * M))
            cos_v = np.cos((2 * y + 1) * v * np.pi / (2 * N))
            idct[x, y] = (2 / N) * np.sum(c[u] * c[v] * dct * cos_u * cos_v)

    return np.clip(idct, 0, 255)


# Implement PSNR
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr_value


if __name__ == '__main__':
    print("START!")
    
    # 1. Load the image and convert to grayscale
    print("1. Load the image and convert to grayscale...")
    image = cv2.imread('image/lena.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (256, 256))
    print("image.shape:", image.shape)
    
    if image is None:
        raise ValueError(
            "Image not found. Make sure 'lena.png' is in the current directory.")
    cv2.imwrite('image/lena_original.png', image)
        
    # 2. Apply one 2D-DCT
    print("2. Apply one 2D-DCT...")
    start_time = time.time()
    dct_coefficients_2d = dct_2d(image)
    end_time = time.time()
    print(f'One 2D-DCT runtime: {end_time - start_time:.4f} seconds')
    
    # 3. Visualize the 2D-DCT coefficients in the log domain
    print("3. Visualize the 2D-DCT coefficients in the log domain...")
    log_dct_2d = np.log(np.abs(dct_coefficients_2d) + 1)
    plt.figure(figsize=(5, 5))
    plt.imshow(log_dct_2d, cmap='gray')
    plt.title('2D-DCT Coefficients')
    plt.savefig('image/DCT_coefficients_2D.png')
    # plt.show()
    
    # 4. Apply one 2D-IDCT
    print("4. Apply one 2D-IDCT...")
    reconstructed_image_2d = idct_2d(dct_coefficients_2d)
    cv2.imwrite('image/lena_reconstructed_2D.png', reconstructed_image_2d)
    
    # 5. Calculate PSNR (2D)
    print("5. Calculate PSNR (2D)...")
    psnr_value = psnr(image, reconstructed_image_2d)
    print(f'PSNR between original and reconstructed image (2D): {psnr_value:.2f} dB')
    
    print("END!")
