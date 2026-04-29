# Instructions for using On-harmony dataset

source:<https://openneuro.org/datasets/ds004712/versions/1.0.2>

## download

### prereqs

install datalad:

- macos: `brew install datalad`
- ubuntu: `sudo apt install datalad`
- pip: `pip install datalad`

### the dataset (only anatomy sequences)

1. download the repository into your datasets directory: `git clone https://github.com/OpenNeuroDatasets/ds004712.git on-harmony`
   - this downloads only the directory structure of the dataset
2. `cd on-harmony`
3. download the actual niftis anatomy with datalad: `find . -name anat -exec datalad get {} \;`
