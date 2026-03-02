import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.signal import convolve2d

# Read image
img = cv2.imread("Raw.jpg") 
if img is None:
    raise FileNotFoundError("Could not read image.")

# Pre-processing
scale_percent = 10
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.show()
# print(img_rgb.shape)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.axis("off")
plt.savefig("gray.png", dpi=300, bbox_inches="tight")

# Convolution
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]], dtype=np.float32)
result = convolve2d(img_gray, kernel, mode='same', boundary='symm')

# Show and save
plt.figure()
plt.imshow(result, cmap='gray')
plt.axis("off")
plt.savefig("result_of_convolution.png", dpi=300, bbox_inches="tight")
# plt.show()

def reflect_idx(i: int, n: int) -> int:
    """Mirror index for scipy.signal.convolve2d(..., boundary='symm')."""
    if i < 0:
        return -i - 1
    if i >= n:
        return 2 * n - i - 1
    return i


def build_laplacian_matrix_symm(h, w):
    # Build matrix
    rows = []
    cols = []
    data = []

    offsets = [
        (0, 0, 4.0),
        (0, -1, -1.0),  # left
        (0, 1, -1.0),   # right
        (-1, 0, -1.0),  # up
        (1, 0, -1.0),   # down
    ]

    for r in range(h):
        for c in range(w):
            row_idx = r * w + c
            for dr, dc, val in offsets:
                rr = reflect_idx(r + dr, h)
                cc = reflect_idx(c + dc, w)
                col_idx = rr * w + cc
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(val)

    n = h * w
    return coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()

# Source
# https://github.com/hanyoseob/matlab-CG?tab=readme-ov-file
def conjgrad(A, b, x, tol=1e-10, max_iters=500):
    """Conjugate gradient solver matching the provided MATLAB routine."""
    if max_iters is None:
        max_iters = len(b)

    r = b - A @ x
    p = r.copy()
    rsold = float(r @ r)

    for i in range(max_iters):
        Ap = A @ p
        denom = float(p @ Ap)
        if abs(denom) < 1e-20:
            print(f"CG stopped early at iter {i}: denominator near zero.")
            break

        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = float(r @ r)

        if np.sqrt(rsnew) < tol:
            print(f"CG converged at iter {i + 1}, residual={np.sqrt(rsnew):.3e}")
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew
        if i % 100 == 0 or i == max_iters - 1:
            print(i)

    return x

# Matrix form y = A x (x is flattened grayscale image)
H, W = img_gray.shape
A = build_laplacian_matrix_symm(H, W)
x_vec = img_gray.astype(np.float32).reshape(-1)
y_vec = A @ x_vec
y_from_A = y_vec.reshape(H, W)

plt.figure()
plt.imshow(y_from_A, cmap='gray')
plt.axis("off")
plt.savefig("result_from_matrix.png", dpi=300, bbox_inches="tight")

# Inverse (solve A x = y) using conjugate gradient
b_vec = result.astype(np.float32).reshape(-1)
x0 = np.zeros_like(b_vec, dtype=np.float32)
x_cg = conjgrad(A, b_vec, x0, tol=1e-8, max_iters=20000)

# Laplacian has constant-offset ambiguity; align means for comparison/visualization.
x_cg += float(np.mean(img_gray) - np.mean(x_cg))
x_cg_img = x_cg.reshape(H, W)

plt.figure()
plt.imshow(np.clip(x_cg_img, 0, 255), cmap='gray')
plt.axis("off")
plt.savefig("reconstructed_cg.png", dpi=300, bbox_inches="tight")

rec_mae = np.mean(np.abs(x_cg_img - img_gray.astype(np.float32)))
res_norm = np.linalg.norm((A @ x_cg) - b_vec)
print(f"CG reconstruction MAE (mean-aligned): {rec_mae:.6f}")
print(f"CG residual norm ||Ax-b||: {res_norm:.6e}")
