# Robust Photometric Stereo in Python

written by Yasuyuki Matsushita (yasumat@ist.osaka-u.ac.jp)

based on a part of robust photometric stereo works in 2010-2018 conducted at Microsoft Research Asia and Osaka University
together with external collaborators.


### What is Photometric Stereo?


Photometric Stereo is an approach to determining surface normal of 
a scene from a set of images recorded from a fixed viewpoint but under
varying lighting conditions, originally proposed by Woodham [1].

### Conditions of use

This package is distributed under the GNU General Public License. For
information on commercial licensing, please contact the authors at the
contact address below. If you use this code for a publication, please
consider citing the following papers:


    @inproceedings{RPS2010,
	  	title={Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery},
	  	author={Lun Wu, Arvind Ganesh, Boxin Shi, Yasuyuki Matsushita, Yongtian Wang, and Yi Ma},
	  	booktitle={Proceedings of Asian Conference on Computer Vision (ACCV)},
	  	year={2010}
	}

    @inproceedings{RPS2012,
	  	title={Robust Photometric Stereo using Sparse Regression},
	  	author={Satoshi Ikehata, David Wipf, Yasuyuki Matsushita, and Kiyoharu Aizawa},
	  	booktitle={Proceedings of Computer Vision and Pattern Recognition (CVPR)},
	  	year={2012}
	}

    @article{RPS2014pami,
        title={Photometric Stereo Using Sparse Bayesian Regression for General Diffuse Surfaces},
        author={Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, and Kiyoharu Aizawa},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
        volume={36},
        number={9},
        pages={1078--1091},
        year={2014}
    }

### Dependencies
The code is written in Python 3.6 but should be able to adapt it to Python 2.x if needed.
You might need the following Python packages installed:
* cv2 (OpenCV, used for image I/O)
* glob (used for reading out a list of images)
* numpy (main computation depends on matrix operations)
* sklearn (scikit-learn, used for normalization of array)


### Acknowledgements

This work was supported by Microsoft Research Asia, Osaka University, and JSPS KAKENHI Grant
Number JP16H01732, Japan.

### Contact information

Questions? Comments? Bug reports? Please contact Yasuyuki Matsushita at yasumat@ist.osaka-u.ac.jp.


### References

[1] Woodham, R.J. Photometric method for determining surface orientation from multiple images. 
Optical Engineerings 19, I, 139-144, 1980

