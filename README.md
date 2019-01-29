# DLV Intellifeatures

Installation Instructions

1. Install Python 2.7 and pip using the instructions found here: https://wiki.python.org/moin/BeginnersGuide/Download
2. Install virtualenv using pip

      pip install virtualenv

3. This is another instruction

NB: the software is currently under active development. Please feel free to contact the developer by email: xiaowei.huang@cs.ox.ac.uk.

Together with the software, there are two documents in Documents/ directory, one is the theory paper and the other is an user manual. The user manual will be updated from time to time. Please refer to the documents for more details about the software.

(1) Installation:

To run the program, one needs to install the following packages:

           Python 2.7
           Theano
           Keras

(2) Usage:

Use the following command to call the program:

           python main.py

Please use the file ''configuration.py'' to set the parameters for the system to run.



Xiaowei Huang

git clone <DLV_repo>

download and install opencv

virtualenv -p python2.7 <path_to_DLV_base_dir>

source <DLV_repo>/bin/activate

git clone https://github.com/Z3Prover/z3.git
cd <z3_base_dir>
python scripts/mk_make.py --python
cd build
make -j4
make install

pip install -r requirements.txt
