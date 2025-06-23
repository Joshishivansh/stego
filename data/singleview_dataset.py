import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PairedAxialMRIDataset(Dataset):
    """
    A PyTorch Dataset class that loads paired axial MRI and CT slices for each patient.
    Each patient has:
        splits/<split>/<patient_id>/MRI/axial/000.png to 255.png
        splits/<split>/<patient_id>/CT/axial/000.png to 255.png

    This dataset returns tuples of (CT_slice_tensor, MRI_slice_tensor)
    """
    def __init__(self, dataroot, split='train', transform=None):
        """
        Args:
            dataroot (str): Root directory of the dataset (e.g., './splits')
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = os.path.join(dataroot, split)
        self.transform = transform if transform else transforms.ToTensor()

        self.pairs = []
        patients = sorted(os.listdir(self.root))
        for patient in patients:
            mri_dir = os.path.join(self.root, patient, 'mask.nii', 'axial')
            ct_dir = os.path.join(self.root, patient, 'ct.nii', 'axial')
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

    def __getitem__(self, idx):
        ct_path, mri_path = self.pairs[idx]
        ct_img = Image.open(ct_path).convert('L')
        mri_img = Image.open(mri_path).convert('L')

        ct_tensor = self.transform(ct_img)
        mri_tensor = self.transform(mri_img)

        return {'A': ct_tensor, 'B': mri_tensor, 'A_paths': ct_path, 'B_paths': mri_path}
