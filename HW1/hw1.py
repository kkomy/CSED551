import numpy as np
import cv2

def Gaussian(i, j, kernel_sigma):
    return 1/(2*np.pi*kernel_sigma**2)*np.exp(-(i**2+j**2)/(2*kernel_sigma**2))

def Gaussian_1D(i, kernel_sigma):
    return 1/(np.sqrt(2*np.pi)*kernel_sigma)*np.exp(-(i**2)/(2*kernel_sigma**2))

def filter_1D(kernel_size, kernel_sigma):
    vector = np.zeros((1, kernel_size))
    for i in range(kernel_size):
            vector[0][i] = Gaussian_1D(-(kernel_size+1)/2 + i, kernel_sigma)
    
    vector = vector/np.sum(vector)
    return vector

def filter_2D(kernel_size, kernel_sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            if(i<=(kernel_size+1)/2 and j<=(kernel_size+1)/2):
                kernel[i][j] = Gaussian(-(kernel_size+1)/2 + i, -(kernel_size+1)/2 + j, kernel_sigma)
    
    kernel = kernel/np.sum(kernel)
    return kernel


def filterGaussian(image, kernel_size, kernel_sigma, border_type, separable):
    loaded = cv2.imread(image, cv2.IMREAD_COLOR)
    length = int((kernel_size+1)/2)
    padded_image = cv2.copyMakeBorder(loaded, length, length, length, length, border_type)
    height, width, channels = loaded.shape
    new_image = np.zeros((height, width, channels))

    if separable:
        #1D Gaussian kernel
        row = filter_1D(kernel_size, kernel_sigma)
        column = np.transpose(row)
        for k in range(channels):
            for i in range(height):
                for j in range(width):
                    new_image[i][j][k] = np.sum(column*row*padded_image[i:i+kernel_size, j:j+kernel_size, k])
        name = 'filtered_' + image
        cv2.imwrite(name, new_image)
    else:
        #2D Gaussian kernel
        result = filter_2D(kernel_size, kernel_sigma)
        for k in range(channels):
            for i in range(height):
                for j in range(width):
                    new_image[i][j][k] = np.sum(result*padded_image[i:i+kernel_size, j:j+kernel_size, k])
        name = 'filtered_' + image
        cv2.imwrite(name, new_image)
    
    return new_image

for i in range(1, 4):
    color_image = 'color' + str(i) + '.jpg'
    gray_image = 'gray' + str(i) + '.jpg'
    filterGaussian(color_image, 5, 3.0, cv2.BORDER_CONSTANT, False)
    filterGaussian(gray_image, 5, 3.0, cv2.BORDER_CONSTANT, False)


'''HISTOGRAM EQUALIZATION'''

def GrayHistogramEqualization(image):
    loaded = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    height, width = loaded.shape
    new_image = np.zeros((height, width))

    histogram = np.zeros(256)
    cumulative = np.zeros(256)

    for i in range(height):
        for j in range(width):
            histogram[loaded[i][j]] += 1
    
    for i in range(1, 256):
        cumulative[i] = cumulative[i-1] + histogram[i]
    
    for i in range(height):
        for j in range(width):
            new_image[i][j] = 255*cumulative[loaded[i][j]]/(height*width)
    
    name = 'equalized_' + image
    cv2.imwrite(name, new_image)

    return new_image

for i in range(1, 4):
    gray_image = 'gray' + str(i) + '.jpg'
    new_image = GrayHistogramEqualization(gray_image)
    

def ColorHistogramEqualization(image):
    loaded = cv2.imread(image, cv2.IMREAD_COLOR)
    height, width, channels = loaded.shape
    new_image = np.zeros((height, width, channels))

    histogram = np.zeros((3, 256))
    cumulative = np.zeros((3, 256))

    for k in range(channels):
        for i in range(height):
            for j in range(width):
                histogram[k][loaded[i][j][k]] += 1
    
        for i in range(1, 256):
            cumulative[k][i] = cumulative[k][i-1] + histogram[k][i]
    
    for k in range(channels):
        for i in range(height):
            for j in range(width):
                new_image[i][j][k] = 255*cumulative[k][loaded[i][j][k]]/(height*width)
    
    name = 'equalized_' + image
    cv2.imwrite(name, new_image)

    return new_image

for i in range(1, 4):
    gray_image = 'color' + str(i) + '.jpg'
    new_image = ColorHistogramEqualization(gray_image)
