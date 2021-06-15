import cv2

# Import the specified png image and convert it to a boolean matrix
def import_image_as_matrix(file_name: str):
	# Read file
	im = cv2.imread(file_name)

	# To Grayscale
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# To Black & White
	im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]

	# Return 2d array
	return im

def test():
	im = import_image_as_matrix('../data/test_420x400.png')
	cv2.imwrite("../output/binary_input.png", im)


if __name__ == "__main__":
	test()