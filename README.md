## RANSAC 2020 tutorial starter pack (in progress)

# Data for epipolar geometry training and validation

Training and validation data you can download from http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
It was obtained with RootSIFT features and mutual nearest neighbour matching.
12 scenes, 100k image pairs each, so 1.2M cases for training, 86 Gb.


Correspondences, which are obtained without mutual nearest neighbor check are available from [here](http://ptak.felk.cvut.cz/personal/mishkdmy/CVPR2020-RANSAC-Tutorial/RANSAC-Tutorial-Data-uni.tar), 349 Gb.

The data comes from train and validation set of the [CVPR IMW 2020 PhotoTourism challenge](https://vision.uvic.ca/image-matching-challenge/data/)


Jupyter notebook, showing the format of the data and toy evaluation example is [here](parse_EF_data.ipynb).



Your methods can use as an input:

x, y, matching score for fundamental matrix case 

and x,y, matching score, calibration matrices K1, K2 for essential matrix case.

The code for running OpenCV RANSACs evaluation on the validation set is coming soon.

The test data is here http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF-Test.tar

[**PyTorch data loader for hdf5 files**](hdf5reader.py)


# Data for training and validation of PnP methods

Training and validation data you can download from http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-PnP.tar.gz (612Mb).
The data is from [EPOS](http://cmp.felk.cvut.cz/epos/) datasset. 

The description of the dataset format and parser to read the data is in [this notebook](https://github.com/ducha-aiki/ransac-tutorial-2020-data/blob/master/PnP%20parse%20data.ipynb)



# Data for hyperparameter tuning for homography. No training data

Test(without GT) and validation data you can download from http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/homography.tar.gz


Jupyter notebook, showing the format of the data and toy evaluation example is [here](parse_H_data.ipynb).
Your methods can use as an input:

x, y, matching score 

The evaluation metric is mean averacy accuracy over set of thresholds. And the thresholded metric is MAE: mean absolute error on jointly visible part of the both images.
For more details see [metrics.py](metrics.py)


## Example submission

To tune hyperparameters on the validation set and create test set prediction with OpenCV RANSAC, run the following script [create_opencv_homography_submission_example.py](create_opencv_homography_submission_example.py)

It will run hyper parameter search on the validation set and then creates two files: homography_opencv_EVD_submission.h5 and homography_opencv_HPatchesSeq_submission.h5
Each of them has the same format as ground truth homography: h5 file with image pairs names as the key and [3x3] homography as an output.


```bash
python -utt create_opencv_homography_submission_example.py
>

```


# Data for 3D point cloud stitching


## Point clouds from ETHZ Photogrammetry and Remote Sensing Group

Download dataset here:

https://cloudstor.aarnet.edu.au/plus/s/229Wnoez2c35Cmw

The data is organised as follows:
- For each scene, there is a sequence of T lidar scans { keypoint_s[t].pcd }, where t=1,...,T.
- Correspondences are available for only consecutive scans { corr_s[t]_s[t+1].txt }.

In the zip package, there is a Matlab function ``pc_plotter(fname_pc1, fname_pc2, fname_corr)`` to plot pairs of point clouds and their 3D correspondences. The function requires file names for a pair of point clouds and their correspondences. As an example, for Arch, you could run in Matlab
```
pc_plotter('../arch/keypoint_s1.pcd', '../arch/keypoint_s2.pcd', '../arch/corr_s1_s2.txt')
```
to display the 3D correspondences.

The inlier thresholds are ... (TBA)

The ground truth transformation parameters are available here:

https://cloudstor.aarnet.edu.au/plus/s/mDMT09jEk0tZTrL

Use a text editor to open the files.

This data was sourced from Photogrammetry and Remote Sensing Group at ETH Zurich:

https://prs.igp.ethz.ch/research/completed_projects/automatic_registration_of_point_clouds.html

If you use the data in any publication, please credit the original authors properly.

## Point clouds from Microsoft 7 Scenes data.

Download dataset here:

https://cloudstor.aarnet.edu.au/plus/s/9KQBYVFSjYn0PDH

There are two instances in this dataset.

In the zip package, there is a Matlab script `plot_7scenes.m` to plot the point clouds and their correspondences. After you run the script, the inlier threshold and ground truth transformation are available respectively in `inst.conf.th` and `GT`.

This data was sourced from Microsoft RGB-D Dataset 7-Scenes:

https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

If you use the data in any publication, please credit the original authors properly.

# Data for 3D object detection

## RGBD Object Dataset

Download dataset here:

https://cloudstor.aarnet.edu.au/plus/s/ko1F2tFQzAyG1I0

There are two instances in this dataset.

In the zip package, there is a Matlab script `plot_lai.m` to plot the point clouds and their correspondences. After you run the script, the inlier threshold and ground truth transformation are available respectively in `inst.conf.th` and `GT`.

This data was sourced from RGB-D Object Dataset from University of Washington:

https://rgbd-dataset.cs.washington.edu/

If you use the data in any publication, please credit the original authors properly.

# Other cases

TBA if any.



