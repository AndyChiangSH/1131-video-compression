import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def three_step_search(img1, img2, block_size, search_range):
    height, width = img1.shape
    mv_x = np.zeros((height // block_size, width // block_size))
    mv_y = np.zeros((height // block_size, width // block_size))
    reconstructed_frame = np.zeros_like(img1)
    residual = np.zeros_like(img1)

    initial_step_size = search_range // 2

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img1[i:i + block_size, j:j + block_size]
            min_cost = float('inf')
            best_match = (0, 0)
            center_x, center_y = i, j
            step_size = initial_step_size

            while step_size > 0:
                for x in range(-step_size, step_size + 1, step_size):
                    for y in range(-step_size, step_size + 1, step_size):
                        ref_x = center_x + x
                        ref_y = center_y + y
                        if ref_x < 0 or ref_y < 0 or ref_x + block_size > height or ref_y + block_size > width:
                            continue
                        candidate_block = img2[ref_x:ref_x +
                                               block_size, ref_y:ref_y + block_size]
                        cost = np.sum((block - candidate_block) ** 2)

                        if cost < min_cost:
                            min_cost = cost
                            best_match = (ref_x - i, ref_y - j)
                            center_x, center_y = ref_x, ref_y

                step_size //= 2

            mv_x[i // block_size, j // block_size] = best_match[0]
            mv_y[i // block_size, j // block_size] = best_match[1]
            ref_x, ref_y = i + best_match[0], j + best_match[1]
            reconstructed_frame[i:i + block_size, j:j +
                                block_size] = img2[ref_x:ref_x + block_size, ref_y:ref_y + block_size]
            residual[i:i + block_size, j:j + block_size] = block - \
                reconstructed_frame[i:i + block_size, j:j + block_size]

    return mv_x, mv_y, reconstructed_frame, residual


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(255 ** 2 / mse)
    
    return psnr


if __name__ == '__main__':
    print("START!")

    # Read images
    print("> Read images...")
    img1 = cv2.imread('image/one_gray.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image/two_gray.png', cv2.IMREAD_GRAYSCALE)
    
    # Set parameters
    print("> Set parameters...")
    block_size = 8
    search_range = 32   # [8, 16, 32]
    file_name = f"three_step_search_{search_range}"
    
    # Compute motion vectors and reconstructed frame using three-step search
    print("> Compute motion vectors and reconstructed frame using three-step search...")
    start_time = time.time()
    mv_x, mv_y, reconstructed_frame, residual = three_step_search(
        img1, img2, block_size, search_range)
    runtime = time.time() - start_time
    psnr = calculate_psnr(img1, reconstructed_frame)
    print(f'Three-Step Search with Search Range {search_range}: PSNR = {psnr:.2f}, Runtime = {runtime:.2f} sec')
    
    # Save to log
    print("> Save to log...")
    with open(f'log/{file_name}.txt', 'w') as f:
        f.write(f'Three-Step Search with Search Range {search_range}: PSNR = {psnr:.2f}, Runtime = {runtime:.2f} sec\n')

    # Save output frames and residuals
    print("> Save reconstructed and residual image...")
    cv2.imwrite(f'reconstructed_frame/{file_name}.png', reconstructed_frame)
    cv2.imwrite(f'residual/{file_name}.png', residual)

    # Plot motion vectors (optional)
    print("> Plot motion vectors (optional)...")
    plt.quiver(mv_y, mv_x)
    plt.gca().invert_yaxis()
    plt.title('Motion Vectors')
    plt.savefig(f'motion_vector/{file_name}.png')
    
    print("END!")
