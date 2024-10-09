import cv2


if __name__ == '__main__':
    print("START!")

    # 1. Load the image and convert to grayscale
    print("1. Load the image and convert to grayscale...")
    image = cv2.imread('image/lena_original.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (256, 256))
    print("image.shape:", image.shape)

    # 2. Load the reconstructed image and convert to grayscale
    print("2. Load the reconstructed image and convert to grayscale...")
    image_reconstructed = cv2.imread('image/lena_reconstructed_2D.png', cv2.IMREAD_GRAYSCALE)
    # image_reconstructed = cv2.resize(image_reconstructed, (256, 256))
    print("image_reconstructed.shape:", image_reconstructed.shape)

    # 3. Calculate the PSNR between the original and reconstructed image
    print("3. Calculate the PSNR between the original and reconstructed image...")
    psnr_value = cv2.PSNR(image, image_reconstructed)
    print(f'PSNR between original and reconstructed image: {psnr_value:.2f} dB')

    print("END!")
