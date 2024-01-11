import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve as convolveim
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def gauss(size, sigma):
    """
    This function will generate 3 filters given the size of the filter and sigma of Gaussian:
    1: gaussian filter;
    2: derivative of gaussian filters in x and y direction.
    """
    # define the x range
    x_ax = np.arange(0,size) - size/2 + 0.5

    # make 1D gaussian filter
    gauss = np.exp(-(x_ax**2)/(2*sigma**2))

    # Compose 2D Gaussian filter from 1D, using the   separability property of 2D Gaussian
    gauss2 = np.outer(gauss,gauss)

    # Normalize the filter so that all coefficients sum to 1
    gauss1 = gauss2 / np.sum(gauss2)

    # Create derivatives of gaussian
    gauss1_dx = np.matrix(np.zeros((np.shape(gauss1)), dtype="float32"))
    gauss1_dy = np.matrix(np.zeros((np.shape(gauss1)), dtype="float32"))
    for j in range(0, len(x_ax)):
        # derivative filter in x
        gauss1_dx[:, j] = (gauss1[:, j] * (-x_ax[j])/(sigma*sigma)).reshape(size,1)

        # difference filter in y
        gauss1_dy[j, :] = (gauss1[j, :] * (-x_ax[j])/(sigma*sigma)).reshape(1,size)

    return gauss1,gauss1_dx, gauss1_dy
# Visualize the filters you created to make sure you are working with the correct filters
gauss1, gauss1_dx, gauss1_dy = gauss(5,1)
plt.figure()
plt.subplot(1,3,1)
plt.imshow(gauss1)
plt.subplot(1,3,2)
plt.imshow(gauss1_dx)
plt.subplot(1,3,3)
plt.imshow(gauss1_dy)

def harris(Ix, Iy , input_image,N):
    """
    The input to this function are the gradient images in x and y directions, the original image and N
    The function will output two arrays/lists x and y which are the N points with largest harris values,
    and an image of the harris values.
    """
    l, m = np.shape(input_image)
    ################################################ TODO ###############################################
    #Forming 3 images
    # (b) Compute three images Ix^2 , Iy^2 , Ix* Iy.
    #Ix square
    Ix2 = Ix**2
    #Iy square
    Iy2 = Iy**2
    #Ix*Iy
    Ixy = Ix*Iy

    # Smooth image Ix2, Iy2, Ixy  with Gaussian filter with sigma=2, size=7.
    # Get the gauss filter for smoothing (reuse what you have)
    gauss_smooth, _, _ = gauss(7,2)
    Ix2_smooth = convolveim(Ix2, gauss_smooth, mode='nearest')
    ################################################ TODO ###############################################
    # CONVOLVE as shown above
    Iy2_smooth = convolveim(Iy2, gauss_smooth, mode='constant')
    Ixy_smooth = convolveim(Ixy, gauss_smooth, mode='constant')
    # By doing this, Ix2_smooth, Iy2_smooth, Ixy_smooth are the three values needed to calculate
    # the A matrix for each pixel.

    ################################################ TODO ###############################################
    # Write code segment to find N harris points in the image
    # Refer to the page 17 of slides on features for the equation
    det = Ix2_smooth * Iy2_smooth - Ixy_smooth**2
    trace = Ix2_smooth + Iy2_smooth
    H = det - 0.04 * trace**2

    # Save a copy of the original harris values before detecting local max
    H0 = H

    # Detect local maximum over 7x7 windows
    local_max_win = 7
    a = int(np.floor(local_max_win / 2))
    H = np.pad(H, ((a, a), (a, a)), 'constant')

    # Initialize a mask to be all ones. The mask corresponds to the local maximum in H
    H_max = np.ones(H.shape)

    for i in range(a, l + a):
        for j in range(a, m + a):
            # Take a WxW patch centered at point (i,j), check if the center point is larger than all other points
            # in this patch. If it is NOT a local max, set H_max[i,j] = 0
            patch = H[i - a:i + a + 1, j - a:j + a + 1]
            if H[i, j] < np.max(patch):
                H_max[i, j] = 0

    # Multiply the mask with H, points that are not local max will become zero
    H = H_max * H
    H = H[a:-a, a:-a]

    # Find the largest N points' coordinates
    # Hint: use np.argsort() and np.unravel_index() to sort H and get the index in sorted order
    indices = np.unravel_index(np.argsort(H, axis=None)[::-1][:N], H.shape)
    x, y = indices

    # x, y should be arrays/lists of x and y coordinates of the Harris points.
    return x, y, H0

##### IMPORTANT: Convert your image to float once you load the image. ######
input_image = cv2.imread('/content/drive/MyDrive/colab/CA5/9.png',0).astype('float')

img = cv2.normalize(input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

################################################ TODO ###############################################
# Generating the gaussian filter

# a)  Generate gradient images of Ix and Iy using filters corresponding to derivative of Gaussian functions
# of a chosen **scale σ and window size w** (let us use w=4σ+1). You can use the convolve function from scipy.

sigma = 1
size =  int(4*sigma + 1)
# Function call to gauss
gauss_filt, gauss_filt_dx, gauss_filt_dy = gauss(size, sigma)


################################################ TODO ###############################################
# Convolving the filter with the image
# Convolve image with dx filter
Ix = convolveim(input_image,gauss_filt_dx,mode ='nearest')
# Convolve image with dy filter
Iy = convolveim(input_image,gauss_filt_dy,mode ='nearest')

x,y,H0 = harris(Ix, Iy ,input_image,100)
################################################ TODO ###############################################
# Plot: Ix, Iy, Original image with harris point labeled in red dots, H0 harris value image
# Hint: you may use "plt.plot(y,x, 'ro')"   # Note: x is vertical and y is horizontal in our above definition
                                            # But when plotting the point, the definition is reversed
plt.figure(figsize=(16, 10))

plt.subplot(1, 4, 1)
plt.imshow(Ix, cmap='gray')
plt.title('Ix')

plt.subplot(1, 4, 2)
plt.imshow(Iy, cmap='gray')
plt.title('Iy')

plt.subplot(1, 4, 3)
plt.imshow(img, cmap='gray')
plt.plot(y, x, 'ro', markersize = 2)
plt.title('Original Image with Harris Points')

plt.subplot(1, 4, 4)
plt.imshow(H0, cmap='gray')
plt.title('Harris Value Image')

plt.show()

def histo(theta4,mag4):
    """
    theta4: an array of quantized orientations, with values 0,1,2...7.
    mag4: an array of the same size with magnitudes
    """
    temp = np.zeros((1,8),dtype='float32')
    ################################################ TODO ###############################################
    # write code segment to add the magnitudes of all vectors in same orientations
    for i in np.arange(8):
        temp[0][i] = np.sum (mag4[np.where(theta4==i)])

    # temp should be a 1x8 vector, where each value corresponds to an orientation and
    # contains the sum of all gradient magnitude, oriented in that orientation
    return temp

def descriptor(theta16, mag16):
    """
    Given a 16x16 patch of theta and magnitude, generate a (1x128) descriptor
    """
    filt, _, _ = gauss(16, 8)
    mag16_filt = mag16 * filt

    # array to store the descriptor. Note that in the end descriptor should have size (1, 128)
    desp = np.array([])

    # find location of maximum theta
    histo16 = histo(theta16, mag16_filt)
    maxloc_theta16 = np.argmax(histo16)

    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            # create histogram of orientations on 4x4 patches in the neighborhood of the Harris points
            sub_theta = theta16[i:i+4, j:j+4]
            sub_mag = mag16_filt[i:i+4, j:j+4]
            sub_histo = histo(sub_theta, sub_mag)

            # shift histogram so that dominant orientation becomes first quantized orientation
            num_shifts = (8 - maxloc_theta16) % 8
            sub_histo = np.roll(sub_histo, num_shifts)

            # update descriptor with orientation magnitude sums for each subregion of size 4x4
            desp = np.concatenate((desp, sub_histo), axis=None)

    # normalize descriptor, clip descriptor, normalize descriptor again
    desp = desp / np.linalg.norm(desp)
    desp = np.clip(desp, 0, 0.2)
    desp = desp / np.linalg.norm(desp)

    desp = np.matrix(desp)
    return desp

def part_B(input_image):

    # Normalize the image
    img = cv2.normalize(input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Generate derivative of Gaussian filters, using sigma=1, filter window size=4*sigma+1
    sigma = 1
    size = 4*sigma+1
    _, filt_dx, filt_dy = gauss(size, sigma)

    ################################################ TODO ###############################################
    # Image convolved with filt_dx and filt_dy
    img_x = convolveim(img, filt_dx)
    img_y = convolveim(img, filt_dy)

    # Calculate magnitude and theta, then quantize theta.
    mag = np.sqrt(img_x ** 2 + img_y ** 2)
    theta = np.arctan2(img_y, img_x)
    theta = (theta/(2*np.pi))*360
    theta = theta*(theta>=0) + (360+theta)*(theta < 0)

    ################################################ TODO ###############################################
    # Quantize theta to 0,1,2,... 7, see instructions above
    q = 45
    N = 8
    theta_q = ((theta + q/2) // q) % N


    ################################################ TODO ###############################################
    # Call harris function to find 100 feature points
    x,y, H0 = harris(img_x, img_y ,input_image,100)

    # Pad 15 rows and columns. You will need this extra border to get a patch centered at the feature point
    #    when the feature points lie on the original border of the image.
    theta_q = cv2.copyMakeBorder(theta_q.astype('uint8'), 7,8,7,8, cv2.BORDER_REFLECT)
    ################################################ TODO ###############################################
    mag = cv2.copyMakeBorder(mag, 7,8,7,8, cv2.BORDER_REFLECT, 0)
    final_descriptor = np.zeros((1,128))

    for i in range (0, len(x)):
        # Since you have already added 15 rows and columns, now the new coordinates of the feature points are (x+8, y+8).
        # Then the patch should be [x[i]:x[i]+16,y[i]:y[i]16]
        # Your patch should be centered at the feature point.
        theta_temp = theta_q[x[i]:x[i]+16,y[i]:y[i]+16]
        # similarly, take a 16x16 patch of mag around the point
        mag_temp = mag[x[i]:x[i]+16,y[i]:y[i]+16]
        # function call to descriptors
        temp2 = descriptor(theta_temp, mag_temp)
        final_descriptor = np.vstack((final_descriptor,temp2))

    # Initially, final descriptor has a row of zeros. We are deleting that extra row here.
    final_descriptor = np.delete(final_descriptor,0,0)
    final_descriptor = np.nan_to_num(final_descriptor)
    final_descriptor = np.array(final_descriptor)

    # Combine x,y to form an array of size (Npoints,2) each row correspond to (x,y)
    # You could use np.hstack() or np.vstack()
    final_points = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))

    return final_descriptor,final_points

input_image = cv2.imread('/content/drive/MyDrive/colab/CA5/9.png',0).astype('float') # input image


# Visualization the results. Plot the feature point similiar to Part1 and plot SIFT features as bar
final_descriptor , final_points = part_B(input_image)

for i in range(0,10):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(input_image,cmap='gray')
    ax1.autoscale(False)
    ax1.plot(final_points[i][1],final_points[i][0], 'ro', markersize =2)
    ax2.bar(np.arange(1,129),final_descriptor[i,:])
    plt.show()

img1 = cv2.imread('/content/drive/MyDrive/colab/CA5/left.jpg',0).astype('float') # read left image
img_1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

descriptor_1, keypoints_1 = part_B(img_1)

img2 = cv2.imread('/content/drive/MyDrive/colab/CA5/right.jpg',0).astype('float') # read right image
img_2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

descriptor_2, keypoints_2 = part_B(img_2)

# Display detected points in the two images
plt.figure(figsize = (10,10))
plt.imshow(img_1,'gray')
plt.plot(keypoints_1[:,1],keypoints_1[:,0],'ro',ms=3)
plt.figure(figsize = (10,10))
plt.imshow(img_2,'gray')
plt.plot(keypoints_2[:,1],keypoints_2[:,0],'ro',ms=3)

# write function to find corresponding points in image
def points_matching(kp1, descriptor1, kp2, descriptor2, threshold):
    matched_loca = list()  # list of all corresponding points pairs. Point pairs can be stored as tuples
   ################################################ TODO ###############################################
    # Find matching points between img1 and img2 using the algorithm described in the above
    # For distance measuring, you may use np.linalg.norm()
    # You could implement it as nested loop for simplicity.
    for i in range(len(kp1)):
        distances = [np.linalg.norm(descriptor1[i] - descriptor2[j]) for j in range(len(kp2))]
        min_index = np.argmin(distances)
        distances.sort()
        ratio = distances[0] / distances[1]

        if ratio < threshold:
            matched_loca.append((kp1[i], kp2[min_index]))

    return matched_loca

# Test different thresholds for the matching
for r in [0.95, 0.8, 0.65, 0.5]:
    matched_loca = points_matching(keypoints_1, descriptor_1, keypoints_2, descriptor_2, r)
    final_image = np.concatenate((img_1, img_2), axis=1)
    print('threshold: ', r)
    print('number of corresponding points found:', len(matched_loca))

    ################################################ TODO ###############################################
    # Write code segment to draw lines joining corresponding points
    # Use cv2.line() to draw the line on final_image
    # Remember the x,y coordinate in numpy and OpenCV is opposite and you need to add image width for pt2
    for pt1, pt2 in matched_loca:
        pt1 = (int(pt1[1]), int(pt1[0]))
        pt2 = (int(pt2[1] + img_1.shape[1]), int(pt2[0]))
        cv2.line(final_image, pt1, pt2, (0, 255, 0), 1)

    plt.figure(figsize=(15, 15))
    plt.imshow(final_image, cmap='gray')
    plt.show()

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status, lw=1):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (255, 255, 255), lw)
    # return the visualization
    return vis

img1 = cv2.imread('/content/drive/MyDrive/colab/CA5/left.jpg',0) # read left image
img2 = cv2.imread('/content/drive/MyDrive/colab/CA5/right.jpg',0) # read right iamge

# Depending on your OpenCV version, you could set up SIFT differently
sift = cv2.SIFT_create()
# sift = cv2.xfeatures2d.SIFT_create()

kp1 = sift.detect(img1, None)
kp2 = sift.detect(img2, None)

# Visualize the keypoints
img1_kps = cv2.drawKeypoints(img1,kp1,None,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kps = cv2.drawKeypoints(img2,kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(15,15))
plt.subplot(121)
plt.imshow(img1_kps)
plt.subplot(122)
plt.imshow(img2_kps)
plt.title('images with keypoints')
plt.show()

################################################ TODO ###############################################
# Use sift.compute to generate sift descriptors/features
(kp1, features1) = sift.compute(img1,kp1)
(kp2, features2) = sift.compute(img2,kp2)


kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])

matcher = cv2.DescriptorMatcher_create("BruteForce")
################################################ TODO ###############################################
# Use knnMatch function in matcher to find corresonding features
# For robustness of the matching results, we'd like to find 2 best matches (i.e. k=2 for knnMatch)
# and return their matching distances
rawMatches = matcher.knnMatch(features1, features2, 2)
matches = []

# Now we validate if the matching is reliable by checking if the best maching distance is less than
# the second matching by a threshold, for example, 20% of the 2nd best maching distance
for m in rawMatches:
    ################################################ TODO ###############################################
    # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
    # Test the distance between points. use m[0].distance and m[1].distance
    if len(m) == 2 and m[0].distance < 0.8 * m[1].distance :
        matches.append((m[0].trainIdx, m[0].queryIdx))

ptsA = np.float32([kp1[i] for (_,i) in matches])
ptsB = np.float32([kp2[i] for (i,_) in matches])

################################################ TODO ###############################################
### Similar to what we did in part C
### Create an image img_match that shows the matching results by drawing lines between corresponding points.
img_match = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
img_match[:img1.shape[0], :img1.shape[1]] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img_match[:img2.shape[0], img1.shape[1]:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for p1, p2 in zip(ptsA, ptsB):
    (x1, y1) = p1
    (x2, y2) = p2
    x2 += img1.shape[1]  # Shift x-coordinate for the second image
    cv2.line(img_match, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

plt.figure(figsize=(15,15))
plt.imshow(img_match,'gray')
plt.title('Matching points (Before RANSAC)')
plt.show()

(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)

img_ransac = drawMatches(img1,img2,kp1,kp2,matches,status)
plt.figure(figsize=(15,15))
plt.imshow(img_ransac,'gray')
plt.title('Matching points (After RANSAC)')
plt.show()

# The given template does not work for. I have tried to overlay the wrapping but the pictures are not aligned. For this reason,
# i tried to align the corners of two image using H as following

(x,y) = img2.shape
corners1 = np.float32([[0, 0], [0, x], [y, x], [y, 0]]).reshape(-1, 1, 2)
# Transform corners of the first image using the homography
corners2 = cv2.perspectiveTransform(corners1, H)
# Combine corners of both images
combined_corners = np.concatenate((corners1, corners2), axis=0)
# Calculate bounding box of the stitched image
[x_min, y_min] = np.int32(combined_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(combined_corners.max(axis=0).ravel() + 0.5)
# Create a translation matrix
H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
# Warp the first image using the combined homography and translation
result = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
# Overlay the second image onto the stitched image
result[-y_min:x + -y_min, -x_min:y + -x_min] = img2

plt.figure(figsize=(15,15))
plt.imshow(result,'gray')
plt.title('Stitched image')

matched_idx = status.nonzero()[0]

def visualize_match(x):
    """
    This function visualize the matches interactively
    You could change the visualization of the matching keypoints by toggling a bar
    Need to have the matches and status ready
        matches: coarse matching results obtained from knnMatch
        status: the refined matching results provided by cv2.findHomography,
                the positive match determined by RANSAC is marked with 1,
                while the negative match is marked with 0
    """
    idx = matched_idx[x]
    img_ransac = drawMatches(img1,img2,kp1,kp2,matches[idx:idx+1],status[idx:idx+1], lw=2)

    plt.figure(figsize=(25,25))
    plt.imshow(img_ransac,'gray')
    plt.title('Matching points (After RANSAC)')
    plt.show()

interact(visualize_match, x=widgets.IntSlider(min=0, max=len(matched_idx)-1, step=1, value=100));

