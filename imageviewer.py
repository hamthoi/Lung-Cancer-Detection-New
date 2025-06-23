import pydicom
import matplotlib.pyplot as plt

# Load the DICOM file
dcm = pydicom.dcmread('sample_images/00edff4f51a893d80dae2d42a7f45ad1/0c3ddf408c349ffc072004c92a97e903.dcm')

# Get the pixel array
img = dcm.pixel_array

# Display the image
plt.imshow(img, cmap=plt.cm.gray)
plt.title('DICOM Image')
plt.axis('off')
plt.show()