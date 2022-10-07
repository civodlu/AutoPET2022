# root of the project sources, data and logging
# as described in the README.md
root = '/mnt/datasets/ludovic/AutoPET'

# download the nifti.zip from the AutoPET challenge.
# This is where the unzipped files are located
root_nifti = f'{root}/dataset/raw/FDG-PET-CT-Lesions'

# location of the preprocessed datasets
root_datasets = f'{root}/dataset/preprocessed'

# where to export results. When this
# folder becomes too large, swtich
# as it will slow down VSCode
root_logging = f'{root}/logging2'