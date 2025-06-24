import pydicom
import matplotlib.pyplot as plt

# Load the DICOM file
dcm = pydicom.dcmread('sample_images/0a0c32c9e08cc2ea76a71649de56be6d/0a67f9edb4915467ac16a565955898d3.dcm')

# Get the pixel array
img = dcm.pixel_array

# Display the image
plt.imshow(img, cmap=plt.cm.gray)
plt.title('DICOM Image')
plt.axis('off')
plt.show()