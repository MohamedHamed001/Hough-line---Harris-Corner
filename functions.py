import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
def gaussian_kernel(size, sigma):
    """ Generate a Gaussian kernel matrix. """
    kernel_range = range(-int(size/2), int(size/2) + 1)
    x, y = np.meshgrid(kernel_range, kernel_range)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_gaussian_blur(image, kernel):
    """ Apply Gaussian blur to an image using a kernel. """
    if len(image.shape) == 3:
        # For color images, apply the blur to each channel
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[:, :, c] = apply_gaussian_blur_single_channel(image[:, :, c], kernel)
        return Image.fromarray(blurred)
    else:
        # For a single channel image (grayscale)
        return apply_gaussian_blur_single_channel(image, kernel)

def apply_gaussian_blur_single_channel(array, kernel):
    """ Apply Gaussian blur to a single channel using a kernel. """
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    array_padded = np.pad(array, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    blurred = np.zeros_like(array)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            blurred[i, j] = np.sum(array_padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    
    return blurred


def sobel_filters(img):
    """Apply Sobel filters to an image to find gradients."""
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])
    
    img_array = np.array(img)
    height, width = img_array.shape
    grad_mag = np.zeros_like(img_array)
    grad_dir = np.zeros_like(img_array, dtype=float)

    # Iterate over each pixel excluding the border
    for i in range(1, height-1):
        for j in range(1, width-1):
            region = img_array[i-1:i+2, j-1:j+2]
            px = np.sum(Gx * region)
            py = np.sum(Gy * region)
            
            grad_mag[i, j] = np.sqrt(px**2 + py**2)
            grad_dir[i, j] = np.arctan2(py, px)

    return grad_mag, grad_dir

def non_maximum_suppression(grad_mag, grad_dir):
    """Apply non-maximum suppression to thin the edges."""
    M, N = grad_mag.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = grad_dir * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                # Angle quantization
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = grad_mag[i, j+1]
                    r = grad_mag[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = grad_mag[i+1, j-1]
                    r = grad_mag[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = grad_mag[i+1, j]
                    r = grad_mag[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = grad_mag[i-1, j-1]
                    r = grad_mag[i+1, j+1]

                # Non-maximum suppression
                if (grad_mag[i,j] >= q) and (grad_mag[i,j] >= r):
                    Z[i,j] = grad_mag[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z
def double_threshold(nms_image, low_threshold, high_threshold):
    # Create binary images for strong and weak edges
    strong_edges = np.zeros_like(nms_image, dtype=np.uint8)
    weak_edges = np.zeros_like(nms_image, dtype=np.uint8)
    
    # Strong edges
    strong_edges[nms_image >= high_threshold] = 255
    
    # Weak edges
    weak_edges[(nms_image < high_threshold) & (nms_image >= low_threshold)] = 255
    
    return strong_edges, weak_edges
def hysteresis(strong_edges, weak_edges):
    height, width = strong_edges.shape

        # Initialize the final edges image
    final_edges = np.zeros((height, width), dtype=np.uint8)
        
        # Check if the weak edge pixels are connected to strong edge pixels
    for i in range(1, height-1):
        for j in range(1, width-1):
            if weak_edges[i, j] != 0:
                # Check if one of the neighbors is a strong edge
                if ((strong_edges[i+1, j-1] == 255) or (strong_edges[i+1, j] == 255) or
                    (strong_edges[i+1, j+1] == 255) or (strong_edges[i, j-1] == 255) or
                    (strong_edges[i, j+1] == 255) or (strong_edges[i-1, j-1] == 255) or
                    (strong_edges[i-1, j] == 255) or (strong_edges[i-1, j+1] == 255)):
                    final_edges[i, j] = 255
                else:
                    final_edges[i, j] = 0
            # Strong edges are always part of the final edge map
            elif strong_edges[i, j] == 255:
                final_edges[i, j] = 255

    return final_edges
def apply_canny(image, sigma, kernel_size, low_threshold, high_threshold):
    if image.ndim == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    # Step 2: Apply Gaussian blur
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_img = apply_gaussian_blur(image_gray, kernel)
    
    # Step 3: Compute gradients using Sobel filters
    gradient_magnitude, gradient_direction = sobel_filters(blurred_img)
    
    # Step 4: Apply Non-maximum Suppression
    nms_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # Step 5: Apply Double Thresholding
    strong_edges, weak_edges = double_threshold(nms_image, low_threshold, high_threshold)
    
    # Step 6: Apply Hysteresis
    final_edges = hysteresis(strong_edges, weak_edges)

    return final_edges

def draw_lines(image, lines):
    if lines is not None:
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 100 * (-b))
            y1 = int(y0 + 100 * (a))
            x2 = int(x0 - 100 * (-b))
            y2 = int(y0 - 100 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image



def hough_transform_visual(image, resolution, threshold_ratio=0.2):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection (using Canny)
    edges = apply_canny(gray, sigma=1, kernel_size=5, low_threshold=50, high_threshold=150)
    
    # Image dimensions
    height, width = edges.shape
    
    # Maximum distance (diagonal length of the image)
    max_dist = int(np.sqrt(height**2 + width**2))
    
    # Define the number of rows in the accumulator
    num_rho = int(resolution * max_dist)
    theta_max = 180
    
    # Rho and theta ranges
    rhos = np.linspace(-max_dist, max_dist, num_rho)
    thetas = np.deg2rad(np.arange(theta_max))
    
    # Create the accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # Voting process
    edge_points = np.argwhere(edges != 0)
    for y, x in edge_points:
        for idx, theta in enumerate(thetas):
            rho = int((x * np.cos(theta) + y * np.sin(theta)) + max_dist)
            rho_idx = int(rho * (num_rho - 1) / (2 * max_dist))
            accumulator[rho_idx, idx] += 1
    max_votes = accumulator.max()
    threshold = int(threshold_ratio * max_votes)   
    # Detect lines
    lines = []
    for rho_idx, theta_idx in np.argwhere(accumulator > threshold):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))
    
    return draw_lines(image, lines)



def harris_corner_visual(image, k=0.05, window_size=5, sigma=1, threshold_ratio=0.2):
    """Detect corners using Harris Corner Detection algorithm and mark them with red dots."""
    # Convert to grayscale if necessary
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian filter to smooth the grayscale image
    gray_blurred = gaussian_filter(gray, sigma=sigma)

    # Gradient calculations using Sobel operator on the grayscale image
    sobel_x = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Products of derivatives
    Ixx = sobel_x**2
    Iyy = sobel_y**2
    Ixy = sobel_x * sobel_y

    # Sum of products of derivatives, applying the Gaussian filter again
    Sxx = gaussian_filter(Ixx, sigma=sigma)
    Syy = gaussian_filter(Iyy, sigma=sigma)
    Sxy = gaussian_filter(Ixy, sigma=sigma)

    # Harris corner response
    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    R = det - k * (trace**2)

    # Thresholding
    threshold = threshold_ratio * R.max()
    corners = (R > threshold).astype(int)

    # Non-maximum suppression using a specified window
    footprint = np.ones((window_size, window_size))
    local_max = maximum_filter(R, footprint=footprint) == R
    corners = corners & local_max

    # Marking corners on the original color image with red dots
    corner_image = np.copy(image)  # Work on the original image
    y_coords, x_coords = np.where(corners == 1)
    for y, x in zip(y_coords, x_coords):
        cv2.circle(corner_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Draw red dots

    return corner_image





