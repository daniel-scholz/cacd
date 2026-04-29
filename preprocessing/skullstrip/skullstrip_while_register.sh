#!/bin/bash
#. /opt/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

conda deactivate
conda activate dataset-utils

curSequence=T1_biasfield_corrected
# path to unprocessed data
dataOrig=~/datasets/ixi/$curSequence

# path to input
dataIn=~/datasets/ixi_reg/$curSequence

# path to output
dataOut=~/datasets/ixi_reg_skullstrip/$curSequence

# find /path/to/dir -type f -name "*T1w.nii.gz" -exec hd-bet -i {} path/to/new/dir $curSequence
mkdir -p $dataOut

echo Output directory: $dataOut
echo Input directory: $dataIn
echo Original directory: $dataOrig

# loop while number of files in dataIn is smaller than in dataOrig
nFilesOrig=$(ls $dataOrig/*.nii.gz 2>/dev/null | wc -l)

while [ "$(ls $dataIn/*.nii.gz 2>/dev/null | wc -l)" -lt "$nFilesOrig" ]; do
    # find files in dataIn that are missing in dataOut and store them in newFiles
    newFiles=$(comm -23 <(ls $dataIn/*.nii.gz | xargs -n1 basename | sort) <(ls $dataOut/*.nii.gz | xargs -n1 basename | sort))

    # if newFiles is empty, continue because all files are skullstripped and we are waiting for the registration to finish
    if [ -z "$newFiles" ]; then
        continue
    fi

    # echo number of files in dataIn compared to dataOrig
    echo "$(ls $dataIn/*.nii.gz 2>/dev/null | wc -l)" / "$nFilesOrig" files registered

    # echo number of files in dataOut compared to dataIn
    echo "$(ls $dataOut/*.nii.gz 2>/dev/null | wc -l)" / "$(ls $dataIn/*.nii.gz 2>/dev/null | wc -l)" files skullstripped

    echo new Files: $newFiles

    # apply hd-bet to all files in newFiles
    for file in $newFiles; do
        hd-bet -i $dataIn/$file -o $dataOut/$file
        echo $dataOut/$file skullstripped
    done

done
