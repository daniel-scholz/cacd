# path to input
dataIn=~/datasets/ixi_reg/T1

# path to output
dataOut=~/datasets/ixi_reg_skullstrip/T1

csvFn=../verify/failed_registrations.csv

# reach each row in the csv file
allFiles=$(awk -F, '{print $1}' $csvFn)

mkdir -p $dataOut

echo Output directory: $dataOut
echo Input directory: $dataIn

# loop through all files
for filename in $allFiles; do
    # remove the file extension
    filename=${filename%.nii.gz}
    # create the output filename
    output=$dataOut/${filename}.nii.gz

    # apply hd-bet to all files in newFiles
    hd-bet -i $dataIn/$filename -o $output
    echo $output skullstripped
done
