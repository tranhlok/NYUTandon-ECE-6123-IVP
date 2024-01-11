'''
Image and Video Processing, Section A

Computer Assignment 6 
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm    # Used to display a progress bar while running for-loops

################################################ TODO ###############################################
# Read in two frames that are several frames apart.
# For example, frame100 and frame110
# Read in grayscale mode

#Load a frame from the sequence
img1 = cv2.imread("/content/drive/MyDrive/colab/CA6/Football/frame166.jpg", 0)
img1 = img1.astype('float')

# Load another frame that is 10 frames after the above frame
img2 = cv2.imread("/content/drive/MyDrive/colab/CA6/Football/frame176.jpg", 0)
img2 = img2.astype('float')

### Plot the two Frames
plt.figure(figsize=(16, 10))

plt.subplot(1, 2, 1)
plt.imshow(img1, cmap = 'gray')
plt.title('Frame 166')

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap = 'gray')
plt.title('Frame 176')

###################################  TODO  ###############################
# Define a function to calculate the MSE with the error block as the input
def mse(error):
  return np.mean(error**2)

################################################ TODO ###############################################
# Define EBMA() which takes as input the template(target block), image, template location(x0, y0) and search range
# Return the matching block and the motion vector
def EBMA(template,img,x0,y0,range_x,range_y):
    # get the number of rows and columns of the image
    rows, cols = img.shape
    # get the number of rows and columns of the template
    b_rows, b_cols = template.shape
    # initialize maximum error, motion vector and matchblock
    min_mse = float('inf')
    xm = 0
    ym = 0
    matchblock = np.zeros((b_rows, b_cols))
    # loop over the searching range to find the matchblock with the smallest error.
    for i in range(max(1,x0-range_x),min(rows-b_rows,x0+range_x)):
        for j in range(max(1,y0-range_y),min(cols-b_cols,y0+range_y)):
            candidate = img[i:i+b_rows, j:j+b_cols]
            error = template - candidate
            mse_error = mse(error)
            if mse_error < min_mse:
                # update motion vector, matchblock and max_error if the error of the new block is smaller
                xm = i
                ym = j
                matchblock = candidate
                min_mse = mse_error
    return xm, ym, matchblock

################################################ TODO ###############################################
# define quantization function to quantize the dct coefficients
# recall the quantization function: Q(f)=floor( (f-mean+Q/2)/Q) *Q+mean
# Assume the mean of the dct coefficients is 0
def quant(dct_coef, q):
    dctimg_quant = np.floor((dct_coef + q / 2) / q) * q
    return dctimg_quant

################################################ TODO ###############################################
# define searching range for EBMA
range_x = 24
range_y = 24
# get the row and column size of the images.
rows, cols = img1.shape[0], img1.shape[1]
# define the block size
N = 8

# Pad the right and bottom sides of image 2, so that the image dimensions (minus the first row/col) is a multiple of N.
img2_pad = np.pad(img2, [[0, N - (rows - 1) % N], [0, N - (cols - 1) % N]], mode='edge')

################################################ TODO ###############################################
# initialize the predicted image as zeros with the same size as img2_pad
pred_img_pad = np.zeros_like(img2_pad)
# Assume the first row & col are already reconstructed, copy them directly from img2
pred_img_pad[0, :] = img2_pad[0, :]
pred_img_pad[:, 0] = img2_pad[:, 0]
# Initialize an array for the error image, which we will be reusing for the next part
err_img_pad = np.zeros_like(img2_pad)

################################################ TODO ###############################################
# Loop through all blocks and for each block find mode that has minimum error
for x0 in tqdm(np.arange(1, (rows - 1), N)):
    for y0 in np.arange(1, (cols - 1), N):
        # get the current block
        patch = img2_pad[x0:x0 + N, y0:y0 + N]
        min_MSE = 255 ** 2

        # mode 0 Vertical
        pred_block = np.zeros((N, N))
        # Vertical prediction to fill pred_block
        pred_block[:, 0] = patch[:, 0]
        # get the error block between the predicted block and the current block
        err_block = patch - pred_block
        # calculate the mse of the error block
        current_mse = mse(err_block)
        # update the predicted block and error block if the mse is smaller
        if current_mse < min_MSE:
            min_pred_block = pred_block.copy()
            min_err_block = err_block.copy()
            min_MSE = current_mse

        # mode 1 Horizontal
        pred_block = np.zeros((N, N))
        # Horizontal prediction to fill pred_block
        pred_block[0, :] = patch[0, :]
        err_block = patch - pred_block
        current_mse = mse(err_block)
        if current_mse < min_MSE:
            min_pred_block = pred_block.copy()
            min_err_block = err_block.copy()
            min_MSE = current_mse

        # mode 2: DC
        pred_block = np.full((N, N), np.mean(patch))
        err_block = patch - pred_block
        current_mse = mse(err_block)
        if current_mse < min_MSE:
            min_pred_block = pred_block.copy()
            min_err_block = err_block.copy()
            min_MSE = current_mse

        # inter-prediction
        # perform EBMA to the current block to find the best match in img1
        xm, ym, pred_block = EBMA(patch, img1, x0, y0, range_x, range_y)
        err_block = patch - pred_block
        current_mse = mse(err_block)
        if current_mse < min_MSE:
            min_pred_block = pred_block.copy()
            min_err_block = err_block.copy()
            min_MSE = current_mse

        ## Put the min_pred_block and min_err_block in the correct position in the output images
        pred_img_pad[x0:x0 + N, y0:y0 + N] = min_pred_block
        err_img_pad[x0:x0 + N, y0:y0 + N] = min_err_block

# Remove padding
pred_img = pred_img_pad[0:rows, 0:cols]
err_img = err_img_pad[0:rows, 0:cols]

################################################ TODO ###############################################
# plot the original image, predicted image, error image

### Plot the two Frames
plt.figure(figsize=(16, 10))

plt.subplot(1, 3, 2)
plt.imshow(pred_img, cmap = 'gray')
plt.title('Predicted Image')

plt.subplot(1, 3, 3)
plt.imshow(err_img, cmap = 'gray')
plt.title('Error Image')

plt.subplot(1, 3, 1)
plt.imshow(img2, cmap = 'gray')
plt.title('Original Image - Frame 176')

################################################ TODO ###############################################
# QUANTIZE WITH DIFFERENT STEP SIZE: 4, 16, 32, 64, 128

Q_list = [4, 16, 32, 64, 128]
# Lists to hold reconstructed image, non-zero counts, psnr
Rec_img =[]
Non_zero = []
PSNR = []
dct_block_quant = np.zeros(img2.shape)
err_block_rec = np.zeros(img2.shape)

for q in Q_list:
    non_zero = 0
    rec_img_pad = np.zeros(img2_pad.shape)

    # Assume first row & col are already reconstructed, copy them directly from img2
    rec_img_pad[0, 0:cols] = img2[0, :]
    rec_img_pad[0:rows, 0] = img2[:, 0]

    for x0 in range(1, int(rows/N)):
        for y0 in range(1, int(cols/N)):
            # Extract current error block from the error image
            err_block = err_img[x0*N:(x0+1)*N, y0*N:(y0+1)*N]

            # Perform DCT on the current error block
            dct_block = cv2.dct(err_block.astype(float))

            # Quantize the coefficients
            dct_block_quant = quant(dct_block, q)

            # Count number of non-zero in this block, update non_zero
            non_zero += np.count_nonzero(dct_block_quant)

            # IDCT to the quantized DCT block
            err_block_rec = cv2.idct(dct_block_quant.astype(float))

            # Reconstruct the block
            rec_img_pad[x0*N:(x0+1)*N, y0*N:(y0+1)*N] = pred_img[x0*N:(x0+1)*N, y0*N:(y0+1)*N] + err_block_rec

    # Remove padding
    rec_img = rec_img_pad[0:rows, 0:cols]

    # Calculate PSNR, Append items to lists
    error_b = img2 - rec_img
    mse1 = mse(error_b)
    psnr = 20 * np.log10(255 / np.sqrt(mse1))
    PSNR.append(psnr)
    Non_zero.append(non_zero)
    # Clip rec_img to (0,255) and change back to uint8
    rec_img = np.clip(rec_img, 0, 255).astype('uint8')
    Rec_img.append(rec_img)

################################################ TODO ###############################################
# Plot the PSNR vs. Nonzero curve, each Reconstructed image with different quantization stepsprint(Non_zero, PSNR)
plt.plot(Non_zero, PSNR)

plt.title('PSNR vs. Nonzero')
plt.xlabel('Nonzero Coefficients')
plt.ylabel('PSNR (dB)')
plt.show()

## reconstructed imags with different quantization steps
num_quantizations = len(Q_list)

fig, axes = plt.subplots(num_quantizations, 1, figsize=(10, 6 * num_quantizations))

for i, q in enumerate(Q_list):
    axes[i].imshow(Rec_img[i], cmap='gray')
    axes[i].set_title(f'Reconstructed Image (Q={q})')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

