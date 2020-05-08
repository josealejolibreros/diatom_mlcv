import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from descriptors import fast, edge_histogram, lbp, hog
import pickle


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
    print(covs)

    return np.float32(em.getMeans()), \
           np.float32(covs), np.float32(em.getWeights())[0]


def get_vector_descriptors_all_image_patches(file):
    img = cv2.imread(file, 0)
    print(file)
    img = cv2.resize(img, (512, 512))
    windowsize_r = 64
    windowsize_c = 64
    descriptors_global = []
    # Crop out the window and calculate the histogram
    #for r in range(0,  img.shape[0] - windowsize_r, windowsize_r):
    for r in range(0, img.shape[0] - 1, windowsize_r):
        #for c in range(0, img.shape[1] - windowsize_c, windowsize_c):
        for c in range(0, img.shape[1] - 1, windowsize_c):
            window = img[r:r + windowsize_r, c:c + windowsize_c]
            res = fourier_transform(window)
            descriptors_vector_one_patch = image_descriptors(window)
            descriptors_global.append(descriptors_vector_one_patch)
            del descriptors_vector_one_patch
    #descriptors_global = np.float32(pca(descriptors))

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


def fourier_transform(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()














def folder_descriptors(folder):
    files = glob.glob(folder + "/*.tif")
    print("Calculating descriptos. Number of images is", len(files))
    return np.concatenate([get_vector_descriptors_all_image_patches(file) for file in files])


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
        print("es nan")
        fv = np.nan_to_num(fv)
    else:
        fv = normalize(fv)
    return fv


def generate_gmm(input_folder, N):
    words = np.concatenate([folder_descriptors(folder) for folder in glob.glob(input_folder + '/*')])
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


def get_fisher_vectors_from_folder(folder, gmm):
    files = glob.glob(folder + "/*.tif")
    return np.float32([fisher_vector(get_vector_descriptors_all_image_patches(file), *gmm) for file in files])


def fisher_features(folder, gmm):
    folders = glob.glob(folder + "/*")
    features = {f: get_fisher_vectors_from_folder(f, gmm) for f in folders}
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

    print("ENTRANDO A GENERAR GMM")

    #gmm = load_gmm(working_folder)
    #Descomentar aqui si no ha sido generado gmm

    #gmm = generate_gmm(working_folder, args.number)


    #gmm = load_gmm(working_folder) if args.loadgmm else generate_gmm(working_folder, 5)
    gmm = load_gmm(model_folder)
    print("GMM GENERADO")

    '''
    print("ENTRANDO A GENERAR FISHER FEATURES TRAINING")
    fisher_features_training = fisher_features(working_folder, gmm)
    print("FISHER FEATURES TRAINING GENERADO")


    # TBD, split the features into training and validation

    
    classifier = train(gmm, fisher_features_training)

    rate = success_rate(classifier, fisher_features_training)
    print("Success rate - validation -  is", rate)

    

    config_dictionary = {'remote_hostname': 'google.com', 'remote_port': 80}

    with open('classifier.model', 'wb') as model_file:
        pickle.dump(classifier, model_file)
        print("Classification model saved")
    '''

    with open('classifier.model', 'rb') as model_file:
        classifier = pickle.load(model_file)
        print("Classification model loaded")


    print("Generando fisher vectors test")
    fisher_features_test = fisher_features(working_folder_test, gmm)
    rate_test = success_rate(classifier, fisher_features_test)
    print("Success rate - validation -  is", rate_test)
    foo =0