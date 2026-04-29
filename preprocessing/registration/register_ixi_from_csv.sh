# Rigide mit NiftyReg T1 (mit skull) -> Atlas (mit Skull) reg_aladin

# path to binary
NiftyReg=./niftyreg/bin/reg_aladin

# path to input
dataIn=~/datasets/ixi/T1

# path to output
dataOut=~/datasets/ixi_reg/T1

# log directory
logDir=./logs

mkdir -p $dataOut
mkdir -p $logDir

# path to atlas
atlas=~/datasets/atlases/sub-mni152_space-mni_t1.nii.gz

# speed flag or not
# speedFlag="-speeeeed"
speedFlag=""

# initialize the counter
i=0

csvFn=../verify/failed_registrations.csv

# reach each row in the csv file
allFiles=$(awk -F, '{print $1}' $csvFn)

# concatenate each string in allFiles with a dataIn/ prefix to imitate globbing dataIn/*
allFiles=$(echo $allFiles | awk -v dataIn=$dataIn '{for (i=1; i<=NF; i++) print dataIn"/"$i}')

echo $allFiles

# number of files: len(allFiles)
nFiles=$(echo $allFiles | wc -w)

echo $nFiles files to register

# iterate over all files in the input directory
for file in $allFiles; do
    # get the filename without the path
    filename=$(basename $file)
    # remove the file extension
    filename=${filename%.nii.gz}
    # create the output filename
    output=$dataOut/${filename}.nii.gz

    # run the registration asynchroniously and log the output to file
    $NiftyReg $speedFlag -flo $file -ref $atlas -res $output -rigOnly >$logDir/${filename}.log 2>$logDir/${filename}.err

    # increment the counter
    i=$((i + 1))
    # print the new filename
    echo $output registered $i/$nFiles

done
