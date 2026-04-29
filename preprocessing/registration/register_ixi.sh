curSequence=T1_biasfield_corrected
# Rigide mit NiftyReg $curSequence (mit skull) -> Atlas (mit Skull) reg_aladin

# path to binary
NiftyReg=~/coding/niftyreg/Release/bin/reg_aladin

# path to input
dataIn=~/datasets/ixi/$curSequence

# path to output
dataOut=~/datasets/ixi_reg/$curSequence

# log directory
logDir=./logs_$curSequence

mkdir -p $dataOut
mkdir -p $logDir

# path to atlas
atlas=~/datasets/atlases/sub-mni152_space-mni_t1.nii.gz

# initialize the counter
i=0

# glob niftis in input directory and store them in allFiles
allFiles=$(ls $dataIn/*.nii.gz)

# number of files is number of elements in allFiles
nFiles=$(ls $dataIn/*.nii.gz | wc -l)

# iterate over all files in the input directory
for file in $allFiles; do
    # get the filename without the path
    filename=$(basename $file)
    # remove the file extension
    filename=${filename%.nii.gz}
    # create the output filename
    output=$dataOut/${filename}.nii.gz

    # if output file already exists, skip the registration
    if [ -f $output ]; then
        echo $output already exists
        i=$((i + 1))
        continue
    fi

    # run the registration asynchroniously and log the output to file
    $NiftyReg -speeeeed -flo $file -ref $atlas -res $output -rigOnly >$logDir/${filename}.log 2>$logDir/${filename}.err

    # increment the counter
    i=$((i + 1))
    # print the new filename
    echo $output registered $i/$nFiles

done
