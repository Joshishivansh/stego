import os
from PIL import Image
import torch
from data.base_dataset import BaseDataset, get_transform


class TripletDataset(BaseDataset):
    """
    A dataset for paired MRI and CT triplets (sagittal, coronal, axial).

    Expected folder structure:
    - dataroot/
        - train/ or test/ or val/
            - patient_001/
                - mri.nii_sagittal.png
                - mri.nii_coronal.png
                - mri.nii_axial.png
                - ct.nii_sagittal.png
                - ct.nii_coronal.png
                - ct.nii_axial.png
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options and rewrite default values."""
        parser.set_defaults(input_nc=3, output_nc=3, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize the dataset."""
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)  # e.g., data/train/
        self.patients = sorted(os.listdir(self.dir))      # List of patient folders
        self.transform = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        patient = self.patients[index]
        patient_dir = os.path.join(self.dir, patient)

        mri_views = ["sagittal", "coronal", "axial"]
        ct_views = ["sagittal", "coronal", "axial"]

        mri_paths = [os.path.join(patient_dir, f"mr.nii_{view}.png") for view in mri_views]
        ct_paths = [os.path.join(patient_dir, f"ct.nii_{view}.png") for view in ct_views]

        mri_imgs = [self.transform(Image.open(p)) for p in mri_paths]
        ct_imgs = [self.transform(Image.open(p)) for p in ct_paths]

        # Stack grayscale images along the channel dimension: shape = [3, H, W]
        A = torch.cat(mri_imgs, dim=0)  # shape: [3, H, W]
        B = torch.cat(ct_imgs, dim=0)  # shape: [3, H, W]
        return {
            'A': A,
            'B': B,
            'A_paths': mri_paths,
            'B_paths': ct_paths
        }

    def __len__(self):
        """Return the total number of patients."""
        return len(self.patients)
