import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import time


# Implement custom DCT for an 8x8 block
def dct_2d(block):
    """Apply 2D DCT to an 8x8 block."""
    M, N = block.shape
    c = np.array([1 / np.sqrt(2) if i == 0 else 1 for i in range(max(M, N))])
    x = np.arange(M).reshape(-1, 1)
    y = np.arange(N).reshape(1, -1)

    dct = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            cos_x = np.cos((2 * x + 1) * u * np.pi / (2 * M))
            cos_y = np.cos((2 * y + 1) * v * np.pi / (2 * N))
            dct[u, v] = (2 / N) * c[u] * c[v] * np.sum(block * cos_x * cos_y)

    return dct


# Quantize using quantization tables
def quantize(block, q_table):
    """Quantize an 8x8 DCT-transformed block."""
    
    return np.round(block / q_table).astype(np.int32)


# Zigzag scan
def zigzag_scan(block):
    """Perform a zigzag scan on an 8x8 block."""
    flat_block = np.zeros(64, dtype=np.int32)
    for i in range(64):
        index = np.unravel_index(zigzag_indices[i], (8, 8))
        flat_block[i] = block[index]
        
    return flat_block


# Run Length Encoding
def run_length_encode(block):
    """Run-Length Encode a 1D block after zigzag scan."""
    zigzag_array = zigzag_scan(block)
    rle = []
    zero_count = 0

    for i in range(len(zigzag_array)):
        if zigzag_array[i] == 0:
            zero_count += 1
        else:
            if zero_count != 0:
                rle.append((0, zero_count))
                zero_count = 0
            rle.append((zigzag_array[i], 1))

    if zero_count > 0:
        rle.append((0, zero_count))

    return rle


# Run Length Decoding
def run_length_decode(rle):
    """Run-Length Decode a list into an 8x8 block using zigzag scan."""
    flat_block = np.zeros(64, dtype=np.int32)
    position = 0

    for value, count in rle:
        for _ in range(count):
            flat_block[position] = value
            position += 1

    # Use zigzag order to reconstruct the 8x8 block
    block = np.zeros((8, 8), dtype=np.int32)
    for i in range(64):
        index = np.unravel_index(zigzag_indices[i], (8, 8))
        block[index] = flat_block[i]

    return block


# Inverse Quantize
def inverse_quantize(block, q_table):
    """Inverse quantize the DCT coefficients."""
    
    return np.multiply(block, q_table)


# Implement custom IDCT for an 8x8 block
def idct_2d(block):
    """Apply 2D IDCT to an 8x8 block."""
    M, N = block.shape
    c = np.array([1 / np.sqrt(2) if i == 0 else 1 for i in range(max(M, N))])
    u = np.arange(M).reshape(-1, 1)
    v = np.arange(N).reshape(1, -1)

    idct = np.zeros((M, N))
    for x in range(M):
        for y in range(N):
            cos_u = np.cos((2 * x + 1) * u * np.pi / (2 * M))
            cos_v = np.cos((2 * y + 1) * v * np.pi / (2 * N))
            idct[x, y] = (2 / N) * np.sum(c[u] * c[v] * block * cos_u * cos_v)

    return np.clip(idct, 0, 255)


# Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# Function to process the image with given quantization table
def process_image(q_table, image):
    # Check if the image dimensions are suitable for 8x8 blocks
    height, width = image.shape
    assert height % 8 == 0 and width % 8 == 0, "Image dimensions should be divisible by 8"

    # Divide image into 8x8 blocks
    print(">> Divide image into 8x8 blocks...")
    blocks = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            blocks.append(block)

    start_time = time.time()

    encoded_blocks = []
    reconstructed_image = np.zeros((height, width), dtype=np.uint8)

    # Apply DCT, Quantize, and Encode using the selected quantization table
    print(">> Apply DCT, Quantize, and Encode using the selected quantization table...")
    for block in blocks:
        dct_block = dct_2d(block)
        quantized_block = quantize(dct_block, q_table)
        encoded_blocks.append(run_length_encode(quantized_block))

    # Decode blocks and reconstruct the image
    print(">> Decode blocks and reconstruct the image...")
    block_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Decode and apply IDCT
            decoded_rle = run_length_decode(encoded_blocks[block_idx])
            dequantized_block = inverse_quantize(decoded_rle.reshape((8, 8)), q_table)
            reconstructed_block = idct_2d(dequantized_block)
            reconstructed_image[i:i+8, j:j+8] = reconstructed_block

            block_idx += 1

    end_time = time.time()
    running_time = end_time - start_time

    # Calculate PSNR
    print(">> Calculate PSNR...")
    psnr_value = calculate_psnr(image, reconstructed_image)

    return reconstructed_image, encoded_blocks, running_time, psnr_value


# Main function
if __name__ == '__main__':
    print("START!")
    
    # Load the image
    print("> Load the image...")
    image = cv2.imread('image/lena.png', cv2.IMREAD_GRAYSCALE)

    # Zigzag order mapping (indices to access blocks in zigzag order)
    zigzag_indices = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]).flatten()

    # Quantization tables
    quantization_table_1 = np.array([
        [10, 7, 6, 10, 14, 24, 31, 37],
        [7, 7, 8, 11, 16, 35, 36, 33],
        [8, 8, 10, 14, 24, 34, 41, 34],
        [8, 10, 13, 17, 31, 52, 48, 37],
        [11, 13, 22, 34, 41, 65, 62, 46],
        [14, 21, 33, 38, 49, 62, 68, 55],
        [29, 38, 47, 52, 62, 73, 72, 61],
        [43, 55, 57, 59, 67, 60, 62, 59]
    ])

    quantization_table_2 = np.array([
        [10, 11, 14, 28, 59, 59, 59, 59],
        [11, 13, 16, 40, 59, 59, 59, 59],
        [14, 16, 34, 59, 59, 59, 59, 59],
        [28, 40, 59, 59, 59, 59, 59, 59],
        [59, 59, 59, 59, 59, 59, 59, 59],
        [59, 59, 59, 59, 59, 59, 59, 59],
        [59, 59, 59, 59, 59, 59, 59, 59],
        [59, 59, 59, 59, 59, 59, 59, 59]
    ])
    
    quantization_table_jpeg = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Process image using both quantization tables
    print("> Processing with Quantization Table 1...")
    reconstructed_image_1, encoded_blocks_1, running_time_1, psnr_1 = process_image(quantization_table_1, image)
    with open('encoded_image/quantization_table_1.pkl', 'wb') as f:
        pickle.dump(encoded_blocks_1, f)
    encoded_size_1 = os.path.getsize('encoded_image/quantization_table_1.pkl')

    print("> Processing with Quantization Table 2...")
    reconstructed_image_2, encoded_blocks_2, running_time_2, psnr_2 = process_image(quantization_table_2, image)
    with open('encoded_image/quantization_table_2.pkl', 'wb') as f:
        pickle.dump(encoded_blocks_2, f)
    encoded_size_2 = os.path.getsize('encoded_image/quantization_table_2.pkl')

    print("> Processing with Quantization Table JPEG...")
    reconstructed_image_jpeg, encoded_blocks_jpeg, running_time_jpeg, psnr_jpeg = process_image(quantization_table_jpeg, image)
    with open('encoded_image/quantization_table_jpeg.pkl', 'wb') as f:
        pickle.dump(encoded_blocks_jpeg, f)
    encoded_size_jpeg = os.path.getsize('encoded_image/quantization_table_jpeg.pkl')

    # Print comparison metrics
    print(f"Quantization Table 1: Encoded Size = {encoded_size_1} bytes, Running Time = {running_time_1:.2f} seconds, PSNR = {psnr_1:.2f} dB")
    print(f"Quantization Table 2: Encoded Size = {encoded_size_2} bytes, Running Time = {running_time_2:.2f} seconds, PSNR = {psnr_2:.2f} dB")
    print(f"Quantization Table JPEG: Encoded Size = {encoded_size_jpeg} bytes, Running Time = {running_time_jpeg:.2f} seconds, PSNR = {psnr_jpeg:.2f} dB")

    # Save the decoded image
    print("> Save the decoded image...")
    cv2.imwrite("decoded_image/original.png", image)
    cv2.imwrite("decoded_image/quantization_table_1.png", reconstructed_image_1)
    cv2.imwrite("decoded_image/quantization_table_2.png", reconstructed_image_2)
    cv2.imwrite("decoded_image/quantization_table_jpeg.png", reconstructed_image_jpeg)
    
    # Plot original vs reconstructed images for visual analysis
    print("> Plot original vs reconstructed images for visual analysis...")
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 4, 2)
    plt.title('Reconstructed Image (1)')
    plt.imshow(reconstructed_image_1, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title('Reconstructed Image (2)')
    plt.imshow(reconstructed_image_2, cmap='gray')
    
    plt.subplot(1, 4, 4)
    plt.title('Reconstructed Image (JEPG)')
    plt.imshow(reconstructed_image_jpeg, cmap='gray')

    plt.savefig("decoded_image/comparison.png")
    plt.show()
    
    print("END!")
