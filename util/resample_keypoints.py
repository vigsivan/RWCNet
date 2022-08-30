#run within slicer
# execfile('C:\\Mike\\Git\\calavera-3d-slicer\\GmmRegistration\\Testing\\exploreGmmRegistrationParameters.py')

import glob
rootDir='E:/mike/repos/optimization-based-registration/datasets/NLST/keypointsTr'
resampledDir='E:/mike/repos/optimization-based-registration/datasets/NLST/keypointsTr/Resampled'
fileList_NLST=glob.glob(rootdir+'/*')

for fileName in fileList_NLST:

    basename=os.path.basename(fileName[0])
    keypointarray = numpy.genfromtxt(fileName, delimiter=',')
    numpy.savetxt(resampledDir+'/'+basename, keypointarray, delimiter=",")