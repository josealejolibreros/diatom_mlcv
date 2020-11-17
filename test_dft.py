import cv2
import numpy as np
from matplotlib import pyplot as plt
from descriptors import fast, edge_histogram, lbp, hog

def get_vector_descriptors_all_image_patches(file):
    img = cv2.imread(file, 0)
    print(file)
    img = cv2.resize(img, (512, 512))
    windowsize_r = 64
    windowsize_c = 64
    descriptors_global = []

    for r in range(0, img.shape[0] - 1, windowsize_r):
        for c in range(0, img.shape[1] - 1, windowsize_c):
            window = img[r:r + windowsize_r, c:c + windowsize_c]
            hist_standard_deviation = get_stdev_histogram_highpass_masked_image(window)

            #After tests, hist highpass dft filtered image > 500  coresponds
            #to a border
            if hist_standard_deviation > 0:
                descriptors_vector_one_patch = image_descriptors(window)
                descriptors_global.append(descriptors_vector_one_patch)
                del descriptors_vector_one_patch

    return descriptors_global


def image_descriptors(img):
    # _, descriptors = cv2.SIFT().detectAndCompute(img, None)

    grey_levels = 256

    '''
    fv_fast = fast.get_features(img)  # .flatten()

    if len(fv_fast.shape) > 0:
        # To adjust same columns as FAST (64 columns) (histogram of n FAST keypoints - dimension = 255)
        hist_fast = np.histogram(fv_fast, bins=grey_levels)[0]

    else:

        hist_fast = np.zeros(256)
    # Edge histograms (dimension = 80)
    fv_edge_hist = np.hstack(edge_histogram.get_features(img))
    '''
    hist_lbp = lbp.get_features(img)
    fv_hog = hog.get_features(img,orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True)

    # HOG + LBP (with current settings: 128 + 26)
    descriptors = np.hstack([fv_hog, hist_lbp])



    #print(file)
    return descriptors


def get_stdev_histogram_highpass_masked_image(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # To generate magnitude spectrum of the Discrete Fourier Transform -Only tests
    '''
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    '''

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0

    # apply mask and inverse DFT
    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


    #To generate magnitude spectrum mask -not necessary -Only tests
    '''
    magnitude_spectrum_mask = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    magnitude_spectrum_mask.astype(int)
    magnitude_spectrum_mask[magnitude_spectrum_mask < 1] = 0
    magnitude_spectrum_mask = np.nan_to_num(magnitude_spectrum_mask)
    '''

    img_back_rescaled = ((img_back - img_back.min()) * (1 / (img_back.max() - img_back.min()) * 255)).astype('uint8')

    hist, bin_edges = np.histogram(img_back_rescaled)
    hist_standard_deviation = hist.std()



    '''
    plt.subplot(141), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(magnitude_spectrum_mask, cmap='gray')
    plt.title('Masked spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(img_back, cmap='gray')
    plt.title(str(hist.std())), plt.xticks([]), plt.yticks([])

    plt.show()



    plt.hist(new_arr, bins='auto')
    '''

    #plt.show()

    return hist_standard_deviation


if __name__ == '__main__':
    model_folder = '.'
    img_name = 'descriptors/e-1.tif'
    get_vector_descriptors_all_image_patches(img_name)