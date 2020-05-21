## RANSAC 2020 tutorial starter pack (in progress)

# Data for epipolar geometry training and validation

Training and validation data you can download from http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar

12 scenes, 100k image pairs each, so 1.2M cases for training, 86 Gb.
The data comes from train and validation set of the [CVPR IMW 2020 PhotoTourism challenge](https://vision.uvic.ca/image-matching-challenge/data/)


Jupyter notebook, showing the format of the data and toy evaluation example is [here](parse_EF_data.ipynb).



Your methods can use as an input:

x, y, matching score for fundamental matrix case 

and x,y, matching score, calibration matrices K1, K2 for essential matrix case.

The code for running OpenCV RANSACs evaluation on the validation set is coming soon.

The test data (only input) is coming soon.

[**PyTorch data loader for hdf5 files**](hdf5reader.py)


# Data for training and validation of PnP methods

Training and validation data you can download from http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-PnP.tar.gz (612Mb).
The data is from [EPOS](http://cmp.felk.cvut.cz/epos/) datasset. 

The format is as follows.  There is one txt file for each object model in each image. Note that there may be multiple instances of an object model in an image; e.g. an image from T-LESS may show multiple fuses of the same type. A txt file contains the following:

----------------------------------------------------------------
The 1st line: scene_id image_id object_id
These ID's can be used to find the associated test image, which is at "dataset/test/scene_id/rgb/image_id.png", where dataset is lmo, tless, or ycbv (the datasets can be downloaded from the BOP web). The mapping of object_id to objects can be seen in Figure 1 of the BOP paper.

The 2nd to 4th line: 3x3 intrinsic matrix K.

The 5th line: The number N of ground truth poses (i.e. the number of instances of object_id visible in the image). The poses are saved as 3x4 matrices on the following 3N lines. Each pose P defines the transformation from the model coordinate system to the camera coordinate system: X_c = P * X_m, where X_c and X_m is a 3D point in the camera and the model coordinate system respectively. The 3D object models can be downloaded from the BOP web.

Then follows a line with the number M of predicted 2D-3D correspondences, each is saved on one line with this format:
u v x y z px_id frag_id conf conf_obj conf_frag
where (u, v) is the 2D image location, (x, y, z) is the predicted corresponding 3D location in the model coordinate system, px_id is the ID of a 2D location -- all correspondences with the same (u, v) have the same px_id and vice versa (note that EPOS produces possibly multiple correspondences at each pixel), frag_id is the ID of the corresponding surface fragment, and conf = conf_obj * conf_frag, where conf_obj is the predicted confidence of object object_id being visible at (u, v), and conf_frag is the predicted confidence of fragment frag_id of object object_id being visible at (u, v).

Then follows a line with the number O of the GT correspondences, each is saved on one line with this format:
u v x y z px_id frag_id gt_id
where (u, v) is the 2D image location, (x, y, z) is the corresponding GT 3D location in the model coordinate system, px_id is the ID of a 2D location (note that there is always only one GT correspondence at each pixel), frag_id is the ID of the corresponding surface fragment, and gt_id is the ID of the GT pose (0-based indexing).


**The example of data parset is coming soon**

# Data for hyperparameter tuning for homography. No training data

Coming soon (expect on  May 20-22)


# Other cases (3d point cloud regstration, etc)

TBA if any.



