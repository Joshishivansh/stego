import os
from PIL import Image
import torch
from data.base_dataset import BaseDataset, get_transform


class SingleViewDataset(BaseDataset):
    """
    A dataset for paired MRI and CT images for a single view (sagittal, coronal, or axial).

    Expected folder structure:
    - dataroot/
        - train/ or test/ or val/
            - patient_001/
                - mr.nii_axial.png
                - ct.nii_axial.png
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add dataset-specific options."""
        parser.set_defaults(input_nc=1, output_nc=1, direction='AtoB')
        parser.add_argument('--view', type=str, choices=['sagittal', 'coronal', 'axial'],
                            required=True, help='View to use (sagittal, coronal, or axial)')
        return parser

    def __init__(self, opt):
        """Initialize the dataset."""
        BaseDataset.__init__(self, opt)
        self.view = opt.view  # 'sagittal', 'coronal', or 'axial'
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.patients = sorted(os.listdir(self.dir))
        self.transform = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        """Return a single-view MRI and CT pair."""
        patient = self.patients[index]
        patient_dir = os.path.join(self.dir, patient)

        mri_path = os.path.join(patient_dir, f"mr.nii_{self.view}.png")
        ct_path = os.path.join(patient_dir, f"ct.nii_{self.view}.png")

        mri_img = self.transform(Image.open(mri_path))
        ct_img = self.transform(Image.open(ct_path))

        # Shape: [1, H, W] since grayscale
        return {
            'A': mri_img,
            'B': ct_img,
            'A_paths': mri_path,
            'B_paths': ct_path
        }

    def __len__(self):
        return len(self.patients)
