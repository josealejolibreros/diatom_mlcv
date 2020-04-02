import skimage.feature


def get_features(image,orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True):
    descriptor, _ = skimage.feature.hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block, visualize=visualize)
    return descriptor