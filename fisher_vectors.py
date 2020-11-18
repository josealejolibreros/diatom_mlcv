import sys, glob, argparse, os
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from descriptors import fast, edge_histogram, lbp, hog, zernike, grey_co_ocurrence_properties, pysift
import pickle
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
from descriptors.contour_properties import Contour

def pca(X):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    #assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca

def svd(X):
    # Data matrix X, X doesn't need to be 0-centered
    n, m = X.shape
    # Compute full SVD
    U, Sigma, Vh = np.linalg.svd(X,
      full_matrices=False, # It's not necessary to compute the full matrix of U or V
      compute_uv=True)
    # Transform X with SVD components
    X_svd = np.dot(U, np.diag(Sigma))
    return X_svd

def dictionary(descriptors, N):
    #em = cv2.EM(N)
    #em.train(descriptors)

    em = cv2.ml.EM_create()
    em.setClustersNumber(N)
    em.trainEM(descriptors)
    covs = em.getCovs()  # this was fixed only 2 weeks ago, so you might need an update
    
    #covs_inv = np.linalg.pinv(covs)
    #covs = np.linalg.inv(covs_inv)
    #print(covs)

    return np.float32(em.getMeans()), \
           np.float32(covs), np.float32(em.getWeights())[0]


def get_vector_descriptors_all_image_patches(file):
    #print(file)
    img = cv2.imread(file, 0)
    #print(file)
    #contour_global(img)
    img = cv2.resize(img, (512, 512))


    #c = plt.imshow(img)
    #plt.title('matplotlib.pyplot.imshow() function Example',
    #          fontweight="bold")
    #plt.show()


    windowsize_r = 64
    windowsize_c = 64
    descriptors_all_the_image = np.empty([0, 0])

    for r in range(0, img.shape[0] - 1, windowsize_r):
        for c in range(0, img.shape[1] - 1, windowsize_c):
            window = img[r:r + windowsize_r, c:c + windowsize_c]
            hist_standard_deviation = get_stdev_histogram_highpass_masked_image(window)

            #After tests, hist highpass dft filtered image > 450  coresponds
            #to a border
            if hist_standard_deviation > 450:
                descriptors_vector_one_patch = image_descriptors(window)
                if descriptors_all_the_image.shape == (0,0):
                    descriptors_all_the_image = descriptors_vector_one_patch
                else:
                    descriptors_all_the_image = np.vstack([descriptors_all_the_image, descriptors_vector_one_patch])
                del descriptors_vector_one_patch

    #print(descriptors_global)
    return descriptors_all_the_image


def contour_global(im):
    imgray = im
    thresh = cv2.adaptiveThreshold(imgray, 255, 0, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    k = 1000
    for cnt in contours:

        # first shows the original image
        im2 = im.copy()
        c = Contour(imgray, cnt)
        print
        c.leftmost, c.rightmost
        cv2.putText(im2, 'original image', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        #cv2.imshow('image', im2)
        if cv2.waitKey(k) == 27:
            break

        im2 = im.copy()

        # Now shows original contours, approximated contours, convex hull
        cv2.drawContours(im2, [cnt], 0, (0, 255, 0), 4)
        string1 = 'green : original contour'
        cv2.putText(im2, string1, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.imshow('image', im2)
        if cv2.waitKey(k) == 27:
            break

        approx = c.approx
        cv2.drawContours(im2, [approx], 0, (255, 0, 0), 2)
        string2 = 'blue : approximated contours'
        cv2.putText(im2, string2, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.imshow('image', im2)
        if cv2.waitKey(k) == 27:
            break

        hull = c.convex_hull
        cv2.drawContours(im2, [hull], 0, (0, 0, 255), 2)
        string3 = 'red : convex hull'
        cv2.putText(im2, string3, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.imshow('image', im2)
        if cv2.waitKey(k) == 27:
            break

        im2 = im.copy()

        # Now mark centroid and bounding box on image
        (cx, cy) = c.centroid
        cv2.circle(im2, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.putText(im2, 'green : centroid', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))

        (x, y, w, h) = c.bounding_box
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.putText(im2, 'red : bounding rectangle', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))

        (center, axis, angle) = c.ellipse
        cx, cy = int(center[0]), int(center[1])
        ax1, ax2 = int(axis[0]), int(axis[1])
        orientation = int(angle)
        cv2.ellipse(im2, (cx, cy), (ax1, ax2), orientation, 0, 360, (255, 255, 255), 3)
        cv2.putText(im2, 'white : fitting ellipse', (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

        cv2.circle(im2, c.leftmost, 5, (0, 255, 0), -1)
        cv2.circle(im2, c.rightmost, 5, (0, 255, 0))
        cv2.circle(im2, c.topmost, 5, (0, 0, 255), -1)
        cv2.circle(im2, c.bottommost, 5, (0, 0, 255))
        cv2.imshow('image', im2)
        if cv2.waitKey(k) == 27:
            break

        # Now shows the filled image, convex image, and distance image
        filledimage = c.filledImage
        cv2.putText(filledimage, 'filledImage', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, 255)
        cv2.imshow('image', filledimage)
        if cv2.waitKey(k) == 27:
            break

        conveximage = c.convexImage
        cv2.putText(conveximage, 'convexImage', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, 255)
        cv2.imshow('image', conveximage)
        if cv2.waitKey(k) == 27:
            break

        distance_image = c.distance_image()
        cv2.imshow('image', distance_image)
        cv2.putText(distance_image, 'distance_image', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
        if cv2.waitKey(k) == 27:
            break

def image_descriptors(img):


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
    fv_hog = hog.get_features(img,orientations=8, pixels_per_cell=(64, 64),
                                cells_per_block=(1, 1), visualize=True)
    greycoprops_description_properties = grey_co_ocurrence_properties.get_features(img)


    # HOG + LBP (with current settings: 128 + 26)
    descriptors = np.concatenate([fv_hog, hist_lbp,greycoprops_description_properties])


    return descriptors


def get_stdev_histogram_highpass_masked_image(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

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


    img_back_rescaled = ((img_back - img_back.min()) * (1 / (img_back.max() - img_back.min()) * 255)).astype('uint8')

    hist, bin_edges = np.histogram(img_back_rescaled)
    hist_standard_deviation = hist.std()

    return hist_standard_deviation












def folder_descriptors(folder, k, kmeans):
    files = glob.glob(folder + "/*.tif")
    print("Calculating descriptos. Number of images is", len(files))
    return np.concatenate([get_LocalsWithoutFisher(file, k, kmeans) for file in files])


def likelihood_moment(x, ytk, moment):
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk


def likelihood_statistics(samples, means, covs, weights):
    gaussians, s0, s1, s2 = {}, {}, {}, {}
    samples_2 = zip(range(0, len(samples)), samples)

    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]
    for index, x in samples_2:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
    del index,x
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        #for index, x in samples:
            #MODIFICADA LA SIGUIENTE LINEA
        idx=0
        for x in samples:

            probabilities = np.multiply(gaussians[idx], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)
            idx=idx+1

    return s0, s1, s2


def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])


def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])


def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32(
        [(s2[k] - 2 * means[k] * s1[k] + (means[k] * means[k] - sigma[k]) * s0[k]) / (np.sqrt(2 * w[k]) * sigma[k]) for
         k in range(0, len(w))])


def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))


def fisher_vector(samples, means, covs, w):
    s0, s1, s2 = likelihood_statistics(samples, means, covs, w)
    T = np.array(samples).shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    if len(a.shape) == 2:
        new_a = np.concatenate(a)
    else:
        new_a = a
    if len(b.shape) == 2:
        new_b = np.concatenate(b)
    else:
        new_b = b
    if len(c.shape) == 2:
        new_c = np.concatenate(c)
    else:
        new_c = c
    fv = np.concatenate([new_a, new_b, new_c])
    import math
    x = fv.min()
    if math.isnan(x):
        #print("es nan")
        fv = np.nan_to_num(fv)
    else:
        fv = normalize(fv)
    return fv


def generate_gmm(input_folder, N, k, kmeans):
    words = np.concatenate([folder_descriptors(folder, k, kmeans) for folder in glob.glob(input_folder + '/*')])
    print("Training GMM of size", N)
    means, covs, weights = dictionary(words, N)
    # Throw away gaussians with weights that are too small:
    th = 1.0 / N
    means = np.float32([m for k, m in zip(range(0, len(weights)), means) if weights[k] > th])
    covs = np.float32([m for k, m in zip(range(0, len(weights)), covs) if weights[k] > th])
    weights = np.float32([m for k, m in zip(range(0, len(weights)), weights) if weights[k] > th])

    np.save("means.gmm", means)
    np.save("covs.gmm", covs)
    np.save("weights.gmm", weights)
    return means, covs, weights

def generate_bof(input_folder):
    classes = len(glob.glob(input_folder + '/*'))
    total_samples = 0
    for _, dirnames, filenames in os.walk(input_folder):
        # ^ this idiom means "we won't be using this value"
        total_samples += len(filenames)
    dico = []
    for folder in glob.glob(input_folder + '/*'):
        if len(dico)>0:
            dico = np.append(dico,folder_sift_description(folder),axis=0)
        else:
            dico = folder_sift_description(folder)

    #dico = np.concatenate([folder_sift_description(folder) for folder in glob.glob(input_folder + '/*')])


    #step2
    k = classes * 10
    print(k)


    batch_size = total_samples * 3
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)
    return k, kmeans

def folder_sift_description(folder):
    files = glob.glob(folder + "/*.tif")
    print("Calculating descriptos. Number of images is", len(files))
    folder_dico = []
    for file in files:
        kp, des = extract_sift_features(file)
        for d in des:
            folder_dico.append(d)

    return folder_dico

def extract_sift_features(file):
    img = cv2.imread(file,0)
    img = cv2.resize(img,(256,256))
    '''
    kp, descriptors = pysift.computeKeypointsAndDescriptors(img,
                                                            sigma=0.2,
                                                            num_intervals=3,
                                                            assumed_blur=0.5,
                                                            image_border_width=5)
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)

    if (len(kp) == 0):
        descriptors = np.zeros((1,128))


    return kp, descriptors

def predict_sift_descriptors_histogram(kp,descriptors,k,kmeans):
    histogram_sift_words = np.zeros(k)
    nkp = np.size(kp)


    for d in descriptors:
        idx = kmeans.predict([d])
        try:
            histogram_sift_words[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly
        except:
            foo=0
    return histogram_sift_words

def get_global_features(file, k, kmeans):
    # print(file)
    img = cv2.imread(file, 0)
    # print(file)
    # contour_global(img)
    img = cv2.resize(img, (512, 512))
    zernike_moments = zernike.get_features(img, radius=21)

    kp, des = extract_sift_features(file)
    histogram_sift_words = predict_sift_descriptors_histogram(kp,des,k,kmeans)
    global_features = np.append(zernike_moments, histogram_sift_words)

    return global_features

def get_fisherLocals_plus_globals(file, k, kmeans):
    fv_local_features = fisher_vector(get_vector_descriptors_all_image_patches(file), *gmm)
    global_features = get_global_features(file, k, kmeans)
    fisherLocals_plus_globals = np.append(fv_local_features, global_features)
    return fisherLocals_plus_globals

def get_LocalsWithoutFisher(file, k, kmeans):
    local_features = get_vector_descriptors_all_image_patches(file)
    return local_features

def get_fisher_vectors_plus_globals_from_folder(folder, k, kmeans):
    files = glob.glob(folder + "/*.tif")
    return np.float32([get_fisherLocals_plus_globals(file, k, kmeans) for file in files])


def fisher_features_plus_global_features(folder, k, kmeans):
    folders = glob.glob(folder + "/*")
    features = {f: get_fisher_vectors_plus_globals_from_folder(f, k, kmeans) for f in folders}
    return features


def train(gmm, features):
    X = np.concatenate(list(features.values()))
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(features)), features.values())])

    #clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, Y)
    return clf


def success_rate(classifier, features):
    print("Applying the classifier...")
    X = np.concatenate(list(features.values()))
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(features)), features.values())])
    res = float(sum([a == b for a, b in zip(classifier.predict(X), Y)])) / len(Y)
    return res


def load_gmm(folder=""):
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    #return map(lambda file: np.load(file), map(lambda s: folder + "/", files))
    return tuple(map(lambda file: np.load(file), files))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dir", help="Directory with images", default='.')
    parser.add_argument("-g", "--loadgmm", help="Load Gmm dictionary", action='store_true', default=False)
    parser.add_argument('-n', "--number", help="Number of words in dictionary", default=5, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    model_folder = '.'
    working_folder = '/home/ubuntu/Escritorio/training'
    working_folder_test = '/home/ubuntu/Escritorio/test'



    #for N in [8, 2, 5, 16, 10, 12, 7, 6, 4, 3, 9, 15]:
    for N in [3,5,9]:
        print("N: "+str(N))
        #gmm = load_gmm(working_folder)
        #Descomentar aqui si no ha sido g
        # enerado gmm

        print("Entrando a generar BoF para almacenar SIFT")
        k, kmeans = generate_bof(working_folder)
        kmeans.verbose = False


        print("ENTRANDO A GENERAR GMM")
        gmm = generate_gmm(working_folder, N, k, kmeans)


        #gmm = load_gmm(working_folder) if args.loadgmm else generate_gmm(working_folder, 5)
        gmm = load_gmm(model_folder)
        print("GMM GENERADO")


        print("ENTRANDO A GENERAR FISHER FEATURES TRAINING")
        fisher_features_plus_globals_training = fisher_features_plus_global_features(working_folder, k, kmeans)
        print("FISHER FEATURES TRAINING GENERADO")


        # TBD, split the features into training and validation


        classifier = train(gmm, fisher_features_plus_globals_training)

        rate = success_rate(classifier, fisher_features_plus_globals_training)
        print("Success rate - validation -  is", rate)




        with open('classifier.model', 'wb') as model_file:
            pickle.dump(classifier, model_file)
            print("Classification model saved")


        with open('classifier.model', 'rb') as model_file:
            classifier = pickle.load(model_file)
            print("Classification model loaded")


        print("Generando fisher vectors test")
        fisher_features_test = fisher_features_plus_global_features(working_folder_test, k, kmeans)
        rate_test = success_rate(classifier, fisher_features_test)
        print("Success rate - test -  is", rate_test)
