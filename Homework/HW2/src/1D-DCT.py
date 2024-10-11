import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# Implement fast 2D-DCT using two 1D-DCT
def dct_1d(vector):
    N = len(vector)
    u = np.arange(N).reshape(-1, 1)
    x = np.arange(N).reshape(1, -1)
    c_u = np.array([1 / np.sqrt(2) if i == 0 else 1 for i in range(N)])
    cos_x = np.cos((2 * x + 1) * u * np.pi / (2 * N))
    dct = c_u * np.sqrt(2 / N) * np.sum(vector * cos_x, axis=1)
    
    return dct


def two_dct_1d(image):
    M, N = image.shape
    dct_rows = np.zeros((M, N))

    # Apply 1D-DCT on rows
    for i in tqdm(range(M), desc='1D-DCT on rows'):
        dct_rows[i, :] = dct_1d(image[i, :])

    # Apply 1D-DCT on columns
    dct = np.zeros((M, N))
    for j in tqdm(range(N), desc='1D-DCT on columns'):
        dct[:, j] = dct_1d(dct_rows[:, j])

    return dct


# Implement fast 2D-IDCT using two 1D-IDCT
def idct_1d(vector):
    N = len(vector)
    x = np.arange(N).reshape(-1, 1)
    u = np.arange(N).reshape(1, -1)
    c_u = np.array([1 / np.sqrt(2) if i == 0 else 1 for i in range(N)])
    cos_x = np.cos((2 * x + 1) * u * np.pi / (2 * N))
    idct = np.sqrt(2 / N) * np.sum(c_u * vector * cos_x, axis=1)
    
    return idct


def two_idct_1d(dct):
    M, N = dct.shape
    idct_columns = np.zeros((M, N))

    # Apply 1D-IDCT on columns
    for j in tqdm(range(N), desc='1D-IDCT on columns'):
        idct_columns[:, j] = idct_1d(dct[:, j])

    # Apply 1D-IDCT on rows
    idct = np.zeros((M, N))
    for i in tqdm(range(M), desc='1D-IDCT on rows'):
        idct[i, :] = idct_1d(idct_columns[i, :])

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

    # 2. Apply two 1D-DCT
    print("2. Apply two 1D-DCT...")
    start_time = time.time()
    dct_coefficients_1d = two_dct_1d(image)
    end_time = time.time()
    print(f'Two 1D-DCT runtime: {end_time - start_time:.4f} seconds')

    # 3. Visualize the 1D-DCT coefficients in the log domain
    log_dct_1d = np.log(np.abs(dct_coefficients_1d) + 1)
    plt.figure(figsize=(5, 5))
    plt.imshow(log_dct_1d, cmap='gray')
    plt.title('1D-DCT Coefficients')
    plt.savefig('image/DCT_coefficients_1D.png')
    # plt.show()

    # 4. Apply two 1D-IDCT
    print("4. Apply two 1D-IDCT...")
    reconstructed_image_1d = two_idct_1d(dct_coefficients_1d)
    cv2.imwrite('image/lena_reconstructed_1D.png', reconstructed_image_1d)

    # 5. Calculate PSNR (1D)
    print("5. Calculate PSNR (1D)...")
    psnr_value = psnr(image, reconstructed_image_1d)
    print(f'PSNR between original and reconstructed image (1D): {psnr_value:.2f} dB')

    print("END!")
