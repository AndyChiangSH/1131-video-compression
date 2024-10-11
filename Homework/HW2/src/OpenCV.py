import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


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

    # 2. Apply OpenCV's DCT
    print("2. Apply OpenCV's DCT...")
    start_time = time.time()
    dct_coefficients_opencv = cv2.dct(np.float32(image))
    end_time = time.time()
    print(f"OpenCV's DCT runtime: {end_time - start_time:.4f} seconds")
    
    log_dct_opencv = np.log(np.abs(dct_coefficients_opencv) + 1)
    plt.figure(figsize=(5, 5))
    plt.imshow(log_dct_opencv, cmap='gray')
    plt.title("OpenCV's DCT Coefficients")
    plt.savefig('image/DCT_coefficients_OpenCV.png')
    # plt.show()
    
    # 3. Apply OpenCV's IDCT
    print("3. Apply OpenCV's IDCT...")
    reconstructed_image_opencv = cv2.idct(np.float32(dct_coefficients_opencv))
    cv2.imwrite('image/lena_reconstructed_OpenCV.png', reconstructed_image_opencv)
    
    psnr_value = psnr(image, reconstructed_image_opencv)
    print(f'PSNR between original and reconstructed image (OpenCV): {psnr_value:.2f} dB')

    print("END!")
