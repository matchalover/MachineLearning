The dataset chosen for this project is a set of satellite image data found on Kaggle (https://www.kaggle.com/crawford/deepsat-sat4). The images in the dataset are originally from a National Agriculture Imagery Program (NAIP) dataset and the original SAT-4 and SAT-6 datasets can be
found here: http://csc.lsu.edu/~saikat/deepsat/.

The dataset has a total of 500,000 images. Each image is 28x28 pixels and has 4 color bands (red, green,
blue and near infrared), which results in 3,316 attributes per image (28x28x4). The dataset is already
broken down into training and testing datasets in an 80%/20% split. The training dataset has 400,000
images and the testing dataset has 100,000 images.
Each 28x28 Images in the subset were divided in four classes, and one-hot coded follows:
 barren land: [1,0,0,0]
 trees: [0,1,0,0]
 grassland: [0,0,1,0]
 other: [0,0,0,1]

Subsetting
The 400,000 image CSV dataset was too large for a laptop that has 8 GB memory to load in MATLAB. Thus, training models using the full dataset would be either very time-consuming or impossible. Any
gains in model performance from using the full dataset would be outweighed by the processing time required. Therefore, a random 100,000 image subset was taken from the initial 400,000 image
training dataset and a random 25,000 image test dataset was taken from the 100,000 image test dataset using the “randperm” function in MATLAB. This way scaled the data to a more
manageable size given the processing constraints. Unless otherwise specified, all results were obtained using these subsets of the initial training and test datasets.
