import argparse
import cv2
import numpy as np


def load_frames(path1: str, path2: str):
    """Load two frames from disk and return them as BGR images."""
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)
    if im1 is None:
        raise FileNotFoundError(f"Could not load image: '{path1}'")
    if im2 is None:
        raise FileNotFoundError(f"Could not load image: '{path2}'")
    return im1, im2


def extract_luminance(image):
    """Convert a BGR image to YCrCb and return the Y (luminance) channel."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0]


def select_roi(image):
    """Let the user select a rectangular ROI by clicking two corners.

    Returns (x1, y1, x2, y2) with x1 <= x2 and y1 <= y2.
    """
    points = []
    display = image.copy()
    window_name = "Select Pedestrian (2 Clicks)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 3, (0, 0, 255), -1)
            print(f"Click {len(points)}: (x={x}, y={y})")

            if len(points) == 2:
                x1 = min(points[0][0], points[1][0])
                y1 = min(points[0][1], points[1][1])
                x2 = max(points[0][0], points[1][0])
                y2 = max(points[0][1], points[1][1])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                print("Rectangle selected! Press any key to continue...")

            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 2:
        raise ValueError("You must click two corners to define the rectangle.")

    x1 = min(points[0][0], points[1][0])
    y1 = min(points[0][1], points[1][1])
    x2 = max(points[0][0], points[1][0])
    y2 = max(points[0][1], points[1][1])
    return x1, y1, x2, y2


def compute_search_window(roi, frame_shape, dx: int, dy: int):
    """Expand the ROI by (dx, dy) pixels, clamped to frame boundaries.

    Parameters
    ----------
    roi : tuple (x1, y1, x2, y2)
    frame_shape : (height, width) of the frame
    dx, dy : search range in pixels

    Returns (sx1, sy1, sx2, sy2).
    """
    x1, y1, x2, y2 = roi
    img_h, img_w = frame_shape

    sx1 = max(0, x1 - dx)
    sy1 = max(0, y1 - dy)
    sx2 = min(img_w, x2 + dx)
    sy2 = min(img_h, y2 + dy)
    return sx1, sy1, sx2, sy2


def block_match_sad(ref_block, search_frame, search_bounds, block_size):
    """Find the best-matching block inside search_bounds using SAD.

    Parameters
    ----------
    ref_block : 2-D ndarray – the reference luminance block from frame 1.
    search_frame : 2-D ndarray – full luminance channel of frame 2.
    search_bounds : tuple (sx1, sy1, sx2, sy2).
    block_size : tuple (width, height) of the reference block.

    Returns (best_x, best_y, min_sad).
    """
    sx1, sy1, sx2, sy2 = search_bounds
    width, height = block_size
    ref_int = ref_block.astype(np.int32)

    min_sad = float("inf")
    best_x, best_y = sx1, sy1

    for cy in range(sy1, sy2 - height + 1):
        for cx in range(sx1, sx2 - width + 1):
            cand_block = search_frame[cy : cy + height, cx : cx + width].astype(np.int32)
            sad = np.sum(np.abs(ref_int - cand_block))

            if sad < min_sad:
                min_sad = sad
                best_x = cx
                best_y = cy

    return best_x, best_y, min_sad

def compute_luminance_difference(ref_block, best_block):
    """Compute signed/absolute luminance differences and compression metrics.

    Returns a dictionary containing the difference matrices and statistics.
    """
    ref_int = ref_block.astype(np.int32)
    best_int = best_block.astype(np.int32)

    # 1. Signed residual: range [-255, +255] (exact difference for compression)
    diff_signed = best_int - ref_int

    # 2. Absolute difference: range [0, 255] (black = 0 error, bright = high error)
    diff_abs = np.abs(diff_signed).astype(np.uint8)

    # 3. Shifted signed residual for visualization: 0 error -> 128 (mid-gray)
    diff_shifted = np.clip(diff_signed + 128, 0, 255).astype(np.uint8)

    # 4. Summary statistics
    mad = float(np.mean(diff_abs))
    mse = float(np.mean(diff_signed ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))

    return {
        "diff_signed": diff_signed,
        "diff_abs": diff_abs,
        "diff_shifted": diff_shifted,
        "mean_signed": float(np.mean(diff_signed)),
        "min_diff": int(np.min(diff_signed)),
        "max_diff": int(np.max(diff_signed)),
        "mad": mad,
        "mse": mse,
        "psnr": psnr,
    }


def show_search_window(im2, roi, search_bounds):
    """Display frame 2 with the original ROI (red) and search window (green)."""
    x1, y1, x2, y2 = roi
    sx1, sy1, sx2, sy2 = search_bounds

    display = im2.copy()
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

    cv2.imshow("Frame 2 - Search Window (Green)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_results(im2, roi, search_bounds, best_pos, block_size, ref_block, best_block, diff_data):
    """Display the final matching results and luminance difference visualizations."""
    x1, y1, x2, y2 = roi
    sx1, sy1, sx2, sy2 = search_bounds
    best_x, best_y = best_pos
    width, height = block_size

    result = im2.copy()
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 255, 0), 1)
    cv2.rectangle(result, (best_x, best_y), (best_x + width, best_y + height), (255, 0, 0), 2)
    sep = np.full((height, 4), 255, dtype=np.uint8)
    comparison = np.hstack([
        ref_block,
        sep,
        best_block,
        sep,
        diff_data["diff_abs"],
        sep,
        diff_data["diff_shifted"],
    ])

    cv2.imshow("Frame 2 - Best Match (Blue) inside Search Window (Green)", result)
    cv2.imshow("Reference Block (Y1)", ref_block)
    cv2.imshow("Best Matched Block (Y2)", best_block)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Block-matching motion estimation between two consecutive frames."
    )
    parser.add_argument("--frame1", default="frame1.png", help="Path to the first frame (default: frame1.png)")
    parser.add_argument("--frame2", default="frame2.png", help="Path to the second frame (default: frame2.png)")
    parser.add_argument("--dx", type=int, default=50, help="Horizontal search range in pixels (default: 30)")
    parser.add_argument("--dy", type=int, default=20, help="Vertical search range in pixels (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load frames ---
    im1, im2 = load_frames(args.frame1, args.frame2)
    Y1 = extract_luminance(im1)
    Y2 = extract_luminance(im2)

    # --- Select ROI on frame 1 ---
    x1, y1, x2, y2 = select_roi(im1)
    width = x2 - x1
    height = y2 - y1
    ref_block = Y1[y1:y2, x1:x2]
    roi = (x1, y1, x2, y2)

    print(f"\nBounding Box: Top-Left=({x1}, {y1}), Bottom-Right=({x2}, {y2})")
    print(f"Block Size: width={width}px, height={height}px")
    print(f"Reference luminance block shape: {ref_block.shape}")

    # --- Compute search window ---
    search_bounds = compute_search_window(roi, Y2.shape, args.dx, args.dy)
    sx1, sy1, sx2, sy2 = search_bounds

    print(f"\nSearch Window: Top-Left=({sx1}, {sy1}), Bottom-Right=({sx2}, {sy2})")
    print(f"Search Window Size: width={sx2 - sx1}px, height={sy2 - sy1}px")

    show_search_window(im2, roi, search_bounds)

    # --- Block matching ---
    best_x, best_y, min_sad = block_match_sad(ref_block, Y2, search_bounds, (width, height))
    mad = min_sad / (width * height)
    mv_x = best_x - x1
    mv_y = best_y - y1
    best_block = Y2[best_y : best_y + height, best_x : best_x + width]

    diff_data = compute_luminance_difference(ref_block, best_block)

    print("\n--- Luminance Difference Results ---")
    print(f"Signed Diff Range: [{diff_data['min_diff']}, {diff_data['max_diff']}] | Mean: {diff_data['mean_signed']:.2f}")
    print(f"MAD: {diff_data['mad']:.2f} | MSE: {diff_data['mse']:.2f} | PSNR: {diff_data['psnr']:.2f} dB")

    reconstructed = ref_block.astype(np.int32) + diff_data["diff_signed"]
    assert np.array_equal(reconstructed, best_block.astype(np.int32)), "Reconstruction check failed!"
    print("Reconstruction check (Y1_ref + residual == Y2_best): PASSED")

    print("\n--- Block Matching Results ---")
    print(f"Original Position (Frame 1): ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"Best Match Position (Frame 2): ({best_x}, {best_y}) -> ({best_x + width}, {best_y + height})")
    print(f"Motion Vector (mv_x, mv_y): ({mv_x}, {mv_y}) pixels")
    print(f"Minimum SAD: {min_sad} | MAD: {mad:.2f} (avg luminance diff/pixel)")

    # --- Show results ---
    show_results(im2, roi, search_bounds, (best_x, best_y), (width, height), ref_block, best_block, diff_data)


if __name__ == "__main__":
    main()