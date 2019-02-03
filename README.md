# DLV Intellifeatures

## Linux/Mac Installation Instructions

1. Install Python 2.7 and pip using the instructions found here: https://wiki.python.org/moin/BeginnersGuide/Download
2. Install virtualenv using pip

           pip install virtualenv

3. Clone the DLV repository using the following command:

           git clone https://github.com/joshua-smith4/DLV_intellifeatures.git <path_to_DLV>

4. Create a python virtual environment in the DLV directory

           cd <path_to_DLV>
           virtualenv -p <python2.7_executable> .
           source bin/activate

5. Download and install opencv with python bindings using the instructions found here (make sure to build python bindings):

           https://docs.opencv.org/3.4.5/d7/d9f/tutorial_linux_install.html
           Make sure to include configuration option PYTHON2_INCLUDE_DIR2 as <path_to_DLV>/include/python2.7

6. Download and install z3 as shown here: https://github.com/Z3Prover/z3 (Remember to follow the instructions for Python install with virtualenv)

           git clone https://github.com/Z3Prover/z3.git <path_to_z3>
           cd <path_to_z3>
           source <path_to_DLV>/bin/activate
           python scripts/mk_make.py --python
           cd build
           make -j4
           make install

7. Install all other dependencies with:

           pip install -r requirements.txt

8. Set up keras environment with a keras config file at ~/.keras/keras.json with the text:

           {
               "floatx": "float32",
               "epsilon": 1e-07,
               "backend": "theano",
               "image_data_format": "channels_last",
               "image_dim_ordering": "th"
           }

9. To run the DLV, checkout the desired branch

           git checkout <desired_branch> (eg. origin/orig_dlv, origin/intellifeatures)
           python main.py --dataset [mnist,cifar10,gtsrb,imageNet]


## Windows Installation Instructions

1. Download and install git by following the instructions here: https://git-scm.com/downloads
2. Download and install the Visual Studio C++ build tools using visual studio installer
3. Install Python 2.7 and pip using the instructions found here: https://wiki.python.org/moin/BeginnersGuide/Download
4. Open a Visual Studio Developer Command Prompt
5. Install virtualenv using pip

           pip install virtualenv

6. Clone the DLV repository using the following command:

           git clone https://github.com/joshua-smith4/DLV_intellifeatures.git <path_to_DLV>

7. Create a python virtual environment in the DLV directory

           cd <path_to_DLV>
           virtualenv -p <python2.7_executable> .
           Scripts\\activate.bat

8. Download and install opencv using the instructions found here (make sure to build python bindings):

           https://docs.opencv.org/3.4.5/d3/d52/tutorial_windows_install.html
           (https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.3/opencv-3.4.3-vc14_vc15.exe/download)

9. Download and install z3 as shown here: https://github.com/Z3Prover/z3 (Remember to follow the instructions for Python install with virtualenv)

           git clone https://github.com/Z3Prover/z3.git <path_to_z3>
           cd <path_to_z3>
           <path_to_DLV>\\Scripts\\activate.bat
           python scripts/mk_make.py --python
           cd <path_to_z3>\\build
           nmake

10. Add <path_to_z3>\\build\\python to the PYTHONPATH environment variable and <path_to_z3>\\build to the PATH environment variable.

11. Install all other dependencies with:

           pip install -r requirements.txt

12. Set up keras environment with a keras config file (keras.json generally found at C:\\Users\\<user>) with the text:

           {
               "floatx": "float32",
               "epsilon": 1e-07,
               "backend": "theano",
               "image_data_format": "channels_last",
               "image_dim_ordering": "th"
           }

13. To run the DLV, checkout the desired branch

           git checkout <desired_branch> (eg. origin/orig_dlv, origin/intellifeatures)
           python main.py --dataset [mnist,cifar10,gtsrb,imageNet]
