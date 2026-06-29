"""
Error Control Coding — BSC Simulation with Linear Block Codes
=========================================================
Schemes compared: Unencoded  |  Repetition (3,1)  |  Hamming (7,4)
Quality metric  : PSNR (Peak Signal-to-Noise Ratio) in dB

Author: Shikhar Srivastava
Description:
    Simulates grayscale image transmission over a Binary Symmetric Channel (BSC).
    Implements and compares two Linear Block Codes (LBC) — Repetition (3,1) and
    Hamming (7,4) — measuring how well each scheme recovers the image from noise.
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

# ─────────────────────────────────────────────
# 2. DOWNLOAD TEST IMAGE (run once in Colab)
# ─────────────────────────────────────────────
# !wget https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png -O test.png


# ─────────────────────────────────────────────
# 3. UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    Higher PSNR = better quality recovery.
    Returns 100 dB if images are identical (MSE = 0).
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def bsc_channel(bits, p):
    """
    Simulate a Binary Symmetric Channel (BSC).

    Parameters:
        bits (np.ndarray): Input binary bit array (dtype uint8).
        p (float): Crossover/error probability (0 <= p <= 1).

    Returns:
        np.ndarray: Received bits after channel noise is applied.

    How it works:
        Each bit is independently flipped with probability p.
        XOR with a noise mask (0 = no flip, 1 = flip).
    """
    noise = np.random.choice([0, 1], size=len(bits), p=[1 - p, p]).astype(np.uint8)
    return np.bitwise_xor(bits, noise)


# ─────────────────────────────────────────────
# 4. REPETITION CODE (3,1)
# ─────────────────────────────────────────────

def repetition_encode(bits):
    """
    Repetition (3,1) Encoder.
    Each data bit is repeated 3 times.
    Code Rate R = 1/3 (low efficiency, high redundancy).

    Example: bit '1' → '1 1 1', bit '0' → '0 0 0'
    """
    return np.repeat(bits, 3)


def repetition_decode(bits):
    """
    Repetition (3,1) Decoder using Majority Logic.
    For each group of 3 received bits, the majority vote wins.

    Example: received '1 0 1' → decoded as '1'  (2 ones vs 1 zero)
    """
    groups = bits.reshape(-1, 3)
    majority = (np.sum(groups, axis=1) > 1.5).astype(np.uint8)
    return majority


# ─────────────────────────────────────────────
# 5. HAMMING CODE (7,4)
# ─────────────────────────────────────────────

def hamming_74_encode(bits):
    """
    Hamming (7,4) Systematic Encoder.
    Encodes 4 data bits into 7-bit codewords using the Generator Matrix G.
    Code Rate R = 4/7 (~0.571) — more efficient than repetition.

    Generator Matrix G (4×7):
        - First 4 columns = Identity (systematic part = original data bits)
        - Last 3 columns  = Parity bits

    Encoding: codeword = data_bits × G  (mod 2)
    """
    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    # Pad bits to be divisible by 4
    pad_len = (4 - len(bits) % 4) % 4
    padded = np.append(bits, np.zeros(pad_len, dtype=np.uint8))

    reshaped = padded.reshape(-1, 4)
    codewords = np.dot(reshaped, G) % 2  # Matrix multiply + mod 2
    return codewords.flatten().astype(np.uint8)


def hamming_74_decode(bits):
    """
    Hamming (7,4) Syndrome Decoder.
    Detects and corrects single-bit errors using the Parity Check Matrix H.

    Parity Check Matrix H (3×7):
        - H × codeword^T = syndrome (mod 2)
        - Syndrome = 000 → no error
        - Syndrome ≠ 000 → identifies the error bit position

    Steps:
        1. Compute syndrome S = received_bits × H^T  (mod 2)
        2. Match syndrome against columns of H to find error position
        3. Flip the identified bit to correct the error
        4. Return only the first 4 bits (the data bits, systematic form)
    """
    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ])
    # Pad bits to be divisible by 7
    pad_len = (7 - len(bits) % 7) % 7
    padded = np.append(bits, np.zeros(pad_len, dtype=np.uint8))

    reshaped = padded.reshape(-1, 7)
    syndromes = np.dot(reshaped, H.T) % 2  # Syndrome calculation

    corrected = reshaped.copy()
    H_columns = H.T.tolist()  # Each column of H = syndrome for that bit position

    for i, s in enumerate(syndromes):
        if np.any(s):  # Non-zero syndrome means there's an error
            try:
                err_pos = H_columns.index(s.tolist())
                corrected[i, err_pos] ^= 1  # Flip the erroneous bit
            except ValueError:
                pass  # Multi-bit error: uncorrectable, skip

    return corrected[:, :4].flatten().astype(np.uint8)  # Return data bits only


# ─────────────────────────────────────────────
# 6. MAIN SIMULATION
# ─────────────────────────────────────────────

def run_advanced_simulation(img_path, p=0.03):
    """
    Run the full BSC channel simulation and compare coding schemes.

    Parameters:
        img_path (str): Path to the input image file.
        p (float): BSC crossover probability (default = 3%).

    Pipeline:
        1. Load and preprocess image (grayscale, 128×128)
        2. Convert image to binary bitstream
        3. Simulate transmission over BSC — unencoded
        4. Encode with Repetition (3,1), transmit, decode
        5. Encode with Hamming (7,4), transmit, decode
        6. Compute PSNR for each and visualize results
    """

    # ── Step 1: Load Image ──────────────────────────────────────────
    img = Image.open(img_path).convert('L').resize((128, 128))
    orig_arr = np.array(img, dtype=np.uint8)
    print(f"Image loaded: {orig_arr.shape}, dtype={orig_arr.dtype}")

    # ── Step 2: Convert to Binary Bitstream ────────────────────────
    # np.unpackbits: converts each uint8 pixel to 8 bits
    # Total bits = 128 × 128 × 8 = 131,072 bits
    orig_bits = np.unpackbits(orig_arr)
    print(f"Total bits in original image: {len(orig_bits)}")

    # ── Step 3: Unencoded Transmission ─────────────────────────────
    # Send raw bits directly through the noisy channel
    unencoded_noisy = bsc_channel(orig_bits, p)
    unencoded_arr = np.packbits(unencoded_noisy).reshape(orig_arr.shape)

    # ── Step 4: Repetition (3,1) ───────────────────────────────────
    rep_encoded = repetition_encode(orig_bits)          # 131,072 → 393,216 bits
    rep_noisy   = bsc_channel(rep_encoded, p)           # Add noise
    rep_decoded = repetition_decode(rep_noisy)          # Majority vote
    rep_arr     = np.packbits(rep_decoded[:len(orig_bits)]).reshape(orig_arr.shape)

    # ── Step 5: Hamming (7,4) ──────────────────────────────────────
    ham_encoded = hamming_74_encode(orig_bits)          # 131,072 → ~229,376 bits
    ham_noisy   = bsc_channel(ham_encoded, p)           # Add noise
    ham_decoded = hamming_74_decode(ham_noisy)          # Syndrome correction
    ham_arr     = np.packbits(ham_decoded[:len(orig_bits)]).reshape(orig_arr.shape)

    # ── Step 6: Compute PSNR ───────────────────────────────────────
    psnr_unencoded = calculate_psnr(orig_arr, unencoded_arr)
    psnr_rep       = calculate_psnr(orig_arr, rep_arr)
    psnr_ham       = calculate_psnr(orig_arr, ham_arr)

    print("\n── PSNR Results ──────────────────────────")
    print(f"  Unencoded      : {psnr_unencoded:.2f} dB")
    print(f"  Repetition(3,1): {psnr_rep:.2f} dB  (Rate R = 1/3)")
    print(f"  Hamming(7,4)   : {psnr_ham:.2f} dB  (Rate R = 4/7)")
    print("──────────────────────────────────────────\n")

    # ── Step 7: Visualize ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Error Control Coding — BSC Simulation  |  Error Probability p = {p}",
                 fontsize=14, fontweight='bold')

    images_and_titles = [
        (orig_arr,       "Original"),
        (unencoded_arr,  f"Unencoded\nPSNR: {psnr_unencoded:.1f} dB"),
        (rep_arr,        f"Repetition (3,1)\nPSNR: {psnr_rep:.1f} dB"),
        (ham_arr,        f"Hamming (7,4)\nPSNR: {psnr_ham:.1f} dB"),
    ]

    for ax, (im, title) in zip(axes, images_and_titles):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("error_control_coding_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as error_control_coding_result.png")


# ─────────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_advanced_simulation('test.png', p=0.03)
