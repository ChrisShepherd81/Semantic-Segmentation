# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Results

### Graph
![graph diagram](./results/graph.png "Graph diagram")

### Cross entropy loss
![Cross entropy loss](./results/cross_entropy_loss.png "Cross entropy loss diagram")

### Example result pictures
|Good             			|  Bad						|
| ------------------------- | ------------------------- |
| ![Result Picture1](./results/img/um_000032.png "Example result picture 1")  |  ![Result Picture7](./results/img/um_000093.png "Example result picture 7")    |
| ![Result Picture2](./results/img/umm_000041.png "Example result picture 2") |  ![Result Picture8](./results/img/um_000074.png "Example result picture 8")    |
| ![Result Picture3](./results/img/umm_000093.png "Example result picture 3") |  ![Result Picture9](./results/img/um_000070.png "Example result picture 9")    |
| ![Result Picture4](./results/img/uu_000052.png "Example result picture 4")  |  ![Result Picture10](./results/img/um_000066.png "Example result picture 10")  |
| ![Result Picture5](./results/img/uu_000058.png "Example result picture 5")  |  ![Result Picture11](./results/img/um_000073.png "Example result picture 11")  |
| ![Result Picture6](./results/img/uu_000098.png "Example result picture 6")  |  ![Result Picture12](./results/img/umm_000059.png "Example result picture 12") |
