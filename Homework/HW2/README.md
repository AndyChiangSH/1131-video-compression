# Homework #2 – 2D-DCT

> Author: 312553024 江尚軒  
> Date: 2024/10/09

## Requirement

- 2D-DCT
    - Implement 2D-DCT to transform “lena.png” to DCT coefficients (visualize in log domain).
        - Convert the input image to grayscale first.
        - Visualize the coefficients in the log domain. Feel free to scale and clip the coefficients for visualization.
    - Implement 2D-IDCT to reconstruct the image.
    - Evaluate the PSNR.
- Two 1D-DCT
    - Implement a fast algorithm by two 1D-DCT to transform “lena.png” to DCT coefficients.
- Compare the runtime between 2D-DCT and two 1D-DCT.
- Do **not** use any functions for DCT and IDCT, e.g., cv2.dct
    - Although, you can still use these functions to validate your output.
- Deadline: 2024/10/14 1:19 PM
- Upload to E3 with required files:
    - **VC_HW2_[student_id].pdf**: Report PDF
    - **VC_HW2_[student_id].zip**: Zipped source code (C/C++/Python/MATLAB) and a **README** file

## How to run?

1. Move into this folder
    
    ```bash
    cd ./VC_HW2_312553024/
    ```
    
2. Create this conda environment
    
    ```bash
    conda env create -f environment.yml
    ```
    
3. Activate this conda environment
    
    ```bash
    conda activate 1131-video-compression-HW2
    ```
    
4. Run this code with 2D-DCT & IDCT
    
    ```bash
    python src/2D-DCT.py
    ```
    
5. Run this code with 1D-DCT & IDCT
    
    ```bash
    python src/1D-DCT.py
    ```
    
6. Run this code with OpenCV's DCT & IDCT
    
    ```bash
    python src/OpenCV.py
    ```
    
7. The DCT coefficients in the log domain will be saved in the `image/` folder
8. The reconstructed images will also be saved in the `image/` folder