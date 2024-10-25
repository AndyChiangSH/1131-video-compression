# Homework #3 – Motion Estimation & Compensation

> Author: 312553024 江尚軒  
> Date: 2024/10/25

## Requirement

- Motion estimation (ME)
    - Block size: 8x8
    - Search range: [+-8]
    - Full search block matching algorithm
    - Integer precision
- Motion compensation (MC)
    - Save the reconstructed frame (after MC) and the residual
- Search range:
    - Compare the results (in PSNR and runtime) with different search ranges ([+-8], [+-16], [+-32]).
- Three-step search
    - Compare the results (in PSNR and runtime) with the Full search algorithm.
- Deadline: 2023/10/28 1:19 PM
- Upload to E3 with required files :
    - **VC_HW3_[student_id].pdf**: Report PDF
    - **VC_HW3_[student_id].zip**: Zipped source code (C/C++/Python/MATLAB) and a **README** file

## How to run?

1. Move into this folder
    
    ```bash
    cd ./VC_HW3_312553024/
    ```
    
2. Create this conda environment
    
    ```bash
    conda env create -f environment.yml
    ```
    
3. Activate this conda environment
    
    ```bash
    conda activate 1131-video-compression-HW3
    ```
    
4. For full search block matching, please run this code
    
    ```bash
    python src/full_search.py
    ```
    
5. For three-step search, please run this code
    
    ```bash
    python src/three_step_search.py
    ```
    
6. The reconstructed frame will be saved in the `reconstructed_frame/` folder
7. The residual will also be saved in the `residual/` folder
8. The motion vector will also be saved in the `motion_vector/` folder
9. The PSNR and runtime will also be saved in the `log/` folder