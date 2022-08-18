#run within slicer
# execfile('C:\\Mike\\Git\\calavera-3d-slicer\\GmmRegistration\\Testing\\exploreGmmRegistrationParameters.py')

import glob
rootDir='E:/mike/repos/optimization-based-registration/datasets/NLST/masksTr'
resampledDir='E:/mike/repos/optimization-based-registration/datasets/NLST/masksTr/Resampled'
fileList_NLST=glob.glob(rootdir+'/*')

for fileName in fileList_NLST:

    basename=os.path.basename(fileName[0])
    imageName=os.path.splitext(os.path.basename(fileName[0]))[0]
    slicer.util.loadVolume(fileName)
    ResampleScalarVolumeParameters={}
    ResampleScalarVolumeParameters['InputVolume']=getNode(imageName).GetID()
    outputNode = slicer.vtkMRMLScalarVolumeNode()
    slicer.mrmlScene.AddNode( outputNode )
    ResampleScalarVolumeParameters['OutputVolume']=outputNode.GetID()
    ResampleScalarVolumeParameters['interpolationType']='nearestNeighbor'
    ResampleScalarVolumeParameters['outputPixelSpacing']=3,3,3
    node = slicer.cli.run(modelMaker, None, modelMakerParameters,True)
    slicer.util.saveNode(outputNode,resampledDir+'/'+basename)
    #remove nodes
    slicer.mrmlScene.RemoveNode(getNode(imageName))
    slicer.mrmlScene.RemoveNode(outputNode)