import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# Implement 2D-DCT
def dct_2d(image):
    M, N = image.shape
    dct = np.zeros((M, N))
    
    for u in tqdm(range(M), desc='2D-DCT'):
        print("u:", u)
        for v in range(N):
            sum_val = 0
            for x in range(M):
                for y in range(N):
                    sum_val += image[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * M)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
            
            c_u = 1 / np.sqrt(2) if u == 0 else 1
            c_v = 1 / np.sqrt(2) if v == 0 else 1
            dct[u, v] = (2 / N) * c_u * c_v * sum_val
    
    return dct


# Implement 2D-IDCT
def idct_2d(dct):
    M, N = dct.shape
    idct = np.zeros((M, N))
    
    for x in tqdm(range(M), desc='2D-IDCT'):
        for y in range(N):
            sum_val = 0
            for u in range(M):
                for v in range(N):
                    c_u = 1 / np.sqrt(2) if u == 0 else 1
                    c_v = 1 / np.sqrt(2) if v == 0 else 1
                    sum_val += c_u * c_v * dct[u, v] * np.cos((2 * x + 1) * u * np.pi / (2 * M)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
            
            idct[x, y] = (2 / N) * sum_val
    
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
    image = cv2.resize(image, (256, 256))
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
    
    # 6. Validate the output using OpenCV's DCT function
    print("6. Validate the output using OpenCV's DCT function...")
    start_time = time.time()
    dct_coefficients_opencv = cv2.dct(np.float32(image))
    end_time = time.time()
    print(f"OpenCV's DCT runtime: {end_time - start_time:.4f} seconds")
    
    log_dct_opencv = np.log(np.abs(dct_coefficients_opencv) + 1)
    plt.imshow(log_dct_opencv, cmap='gray')
    plt.title('DCT Coefficients (OpenCV)')
    plt.savefig('image/DCT_coefficients_OpenCV.png')
    # plt.show()
    
    reconstructed_image_opencv = cv2.idct(np.float32(dct_coefficients_opencv))
    cv2.imwrite('lena_reconstructed_OpenCV.png', reconstructed_image_opencv)
    
    print("END!")
