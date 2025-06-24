import os
from PIL import Image
from data.base_dataset import BaseDataset, get_transform


class SingleViewDataset(BaseDataset):
    """
    A PyTorch Dataset class that loads paired axial MRI and CT slices for each patient.
    Each patient has:
        splits/<split>/<patient_id>/MRI/axial/000.png to 255.png
        splits/<split>/<patient_id>/CT/axial/000.png to 255.png

    This dataset returns dictionaries with keys:
        'A': CT slice tensor
        'B': MRI slice tensor
        'A_paths': path to CT image
        'B_paths': path to MRI image
    """

    def __init__(self, opt):
        """
        Args:
            opt (Option class): stores all experiment flags (must include dataroot, phase, preprocess, etc.)
        """
        super().__init__(opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)  # e.g., './data/train'
        self.transform = get_transform(opt, grayscale=True)
        self.pairs = []

        patients = sorted(os.listdir(self.dir))
        for patient in patients:
            mri_dir = os.path.join(self.dir, patient, 'mask.nii', 'axial')
            ct_dir = os.path.join(self.dir, patient, 'ct.nii', 'axial')
            if os.path.isdir(mri_dir) and os.path.isdir(ct_dir):
                mri_slices = sorted(os.listdir(mri_dir))
                ct_slices = sorted(os.listdir(ct_dir))
                if len(mri_slices) == len(ct_slices) == 256:
                    for mri_slice, ct_slice in zip(mri_slices, ct_slices):
                        self.pairs.append((
                            os.path.join(ct_dir, ct_slice),
                            os.path.join(mri_dir, mri_slice)
                        ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ct_path, mri_path = self.pairs[index]
        ct_img = Image.open(ct_path).convert('L')
        mri_img = Image.open(mri_path).convert('L')

        ct_tensor = self.transform(ct_img)
        mri_tensor = self.transform(mri_img)

        return {'A': ct_tensor, 'B': mri_tensor, 'A_paths': ct_path, 'B_paths': mri_path}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add new dataset-specific options and rewrite default values for existing options.
        """
        return parser