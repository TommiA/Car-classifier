# Image classifier
Classifier for a bunch of images based on the Tensorflow and Google SSD mobilenet neural network

Python and Tensorflow based classifier for the images downloaded e.g. by the Nettiauto_scraper. Requires Tensorflow and supporting models available at https://github.com/tensorflow/models

Produces folder "sorted" for the classified images in the respective folders named after the class and "unsorted" for unknown classifications. Sorts images to classes of the highest probapility over 0.5.
