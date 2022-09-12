"""
This file implements BSIF (http://www.ee.oulu.fi/~jkannala/bsif/bsif.html) histogram extraction.
"""

import os
from typing import List

import bsif
import cv2
import h5py
import numpy as np


def filter_index_generator(
    filter_size: int, num_filters: int, row: int, col: int, current_filter: int
) -> int:
    """Converts a filter index to a matrix index."""
    # C++ and python use row-major order, so the last dimension is contiguous
    # in doubt, refer to https://en.wikipedia.org/wiki/Row-_and_column-major_order#Column-major_order
    return current_filter + num_filters * (col + filter_size * row)


def generate_histogram(
    loaded_image: np.ndarray, filter_size: int, num_filters: int
) -> np.ndarray:
    """Extracts a histogram for the BSIF response to an image."""

    # initialize matrix of ones (to hold BSIF results)
    iris_code_img: np.ndarray = np.ones(
        (loaded_image.shape[0], loaded_image.shape[1]), np.int64
    )

    # create wrapping border around the image
    #   e.g. if filter size is 3x3, we want a border of 1 to account for edges
    border = int(filter_size) // 2

    imgWrap = cv2.copyMakeBorder(
        loaded_image, border, border, border, border, cv2.BORDER_WRAP
    )

    # load hard coded filters
    loaded_filter = bsif.load(filter_size, num_filters)

    # initialize current filter
    filter_kernel = np.empty((filter_size, filter_size), np.float64)

    # loop through filters, starting with the last one
    for current_filter in range(num_filters - 1, -1, -1):

        # Load current filter (need to do it this way due to the storage of the filter - Matlab file)
        for row in range(filter_size):
            for col in range(filter_size):
                filter_kernel[row, col] = loaded_filter[
                    filter_index_generator(
                        filter_size=filter_size,
                        num_filters=num_filters,
                        row=row,
                        col=col,
                        current_filter=current_filter,
                    )
                ]

        # convolves the original image with the kernel (filter)
        #   using default kernel anchor (center point is anchor)
        filtered_img = cv2.filter2D(
            imgWrap,
            ddepth=cv2.CV_64F,
            kernel=filter_kernel,
            delta=0,
            borderType=cv2.BORDER_CONSTANT,
        )

        # Convert any positive values in the matrix to 2^(i-1) as in the Matlab program
        for row in range(loaded_image.shape[0]):
            for col in range(loaded_image.shape[1]):
                # need to ignore the added border for statements loop over
                #   the ORIGINAL image size so the end
                #   border (bottom and left) will be ignored
                if filtered_img[(row + border), (col + border)] > 0.001:
                    # add the binary amount corresponding to the filter number
                    iris_code_img[row, col] += 2 ** (num_filters - 1 - current_filter)

    # create the histogram
    bins = 2**num_filters  # e.g. 256 bins for 8 filters (1-256)

    hist, _ = np.histogram(iris_code_img, bins=bins)

    return hist


def extract_and_store(
    image_location: str,
    file_names: List[str],
    out_location: str,
    out_filename: str,
    filter_size: int,
    segmentation_type: str = "bg",
    num_filters: int = 8,
):
    """Extracts features for a list of files and saves them as an HDF5 file."""

    valid_seg_types = ["bg", "wi"]
    assert (
        segmentation_type in valid_seg_types
    ), "Segmentation type must be one of: {}".format(valid_seg_types)

    # Create file name
    filename = os.path.join(
        out_location,
        "_".join(
            [
                out_filename,
                segmentation_type,
                str(num_filters) + "filters",
                str(filter_size) + "x" + str(filter_size),
            ]
        )
        + ".hdf5",
    )

    downsample = False
    if filter_size % 2 == 0:
        downsample = True
        filter_size = filter_size // 2

    # Create output file
    histogram_file = h5py.File(filename, "w")

    try:

        # Loop through images
        for image in file_names:

            # load the image (flags = 0 to load in B+W)
            src = cv2.imread(filename=(os.path.join(image_location, image)), flags=0)

            # crop image based on segmentation ('wi' whole image will not change the src)
            if segmentation_type == "bg":
                src = src[125:375, 195:445]

            if downsample:
                # blur an image and downsample it:
                #   size is defined as x,y for this function not row,col
                src = cv2.pyrDown(
                    src=src, dstsize=(src.shape[1] // 2, src.shape[0] // 2)
                )

            # get the histogram
            histogram = generate_histogram(src, filter_size, num_filters)

            # convert histograms to numpy array writing to file
            histogram_file[image] = np.asarray(histogram)

    finally:
        # close HDF5 file
        histogram_file.close()


def main():
    """Entrypoint for the script."""
    for file_name in ["04261d2234.tiff"]:
        img_src = cv2.imread((file_name), flags=0)
        hist = np.asarray(
            generate_histogram(loaded_image=img_src, filter_size=3, num_filters=5)
        )
        print("{}: {}".format(file_name, hist))


if __name__ == "__main__":
    main()
