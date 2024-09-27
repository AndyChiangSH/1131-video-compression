from PIL import Image
import numpy as np


def normalize_channel(channel):
    """
    Normalize the channel to ensure pixel values are within [0, 255].
    """
    channel = np.clip(channel, 0, 255)
    
    return channel.astype(np.uint8)


def save_grayscale_image(channel, filename):
    """
    Save a single channel as a grayscale image.
    """
    img = Image.fromarray(channel, 'L')
    img.save(filename)
    
    print(f"Saved {filename}")


def main():
    # Load the image
    try:
        img = Image.open('./image/lena.png').convert('RGB')
    except FileNotFoundError:
        print("Error: 'lena.png' can not found.")
        return

    # Convert image to NumPy array
    img_np = np.array(img)

    # Separate the RGB channels
    R = img_np[:, :, 0].astype(float)
    G = img_np[:, :, 1].astype(float)
    B = img_np[:, :, 2].astype(float)

    R = normalize_channel(R)
    G = normalize_channel(G)
    B = normalize_channel(B)

    # Convert from RGB to YUV
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.331 * G + 0.5 * B + 128
    V = 0.5 * R - 0.419 * G - 0.081 * B + 128
    
    Y = normalize_channel(Y)
    U = normalize_channel(U)
    V = normalize_channel(V)

    # Convert from RGB to YCbCr
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    
    Cb = normalize_channel(Cb)
    Cr = normalize_channel(Cr)

    # Save RGB channels
    save_grayscale_image(R, './image/lena_R.png')
    save_grayscale_image(G, './image/lena_G.png')
    save_grayscale_image(B, './image/lena_B.png')

    # Save YUV channels
    save_grayscale_image(Y, './image/lena_Y.png')
    save_grayscale_image(U, './image/lena_U.png')
    save_grayscale_image(V, './image/lena_V.png')

    # Save YCbCr channels
    save_grayscale_image(Cb, './image/lena_Cb.png')
    save_grayscale_image(Cr, './image/lena_Cr.png')


if __name__ == "__main__":
    print("Start color transform...")
    
    main()
    
    print("End color transform...")
