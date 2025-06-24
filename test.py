"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='Stego-GAN')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML







# import os
# import csv
# import numpy as np
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images
# from util import html
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# try:
#     import wandb
# except ImportError:
#     print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

# def tensor2im(input_image):
#     image_tensor = input_image[0].cpu().float().detach()
#     image_numpy = image_tensor.numpy()
#     image_numpy = (image_numpy + 1) / 2.0  # de-normalize
#     image_numpy = np.squeeze(image_numpy)
#     image_numpy = (image_numpy * 255).astype(np.uint8)
#     return image_numpy

# def compute_metrics(fake, real):
#     mse = np.mean((fake.astype(np.float32) - real.astype(np.float32)) ** 2)
#     mae = np.mean(np.abs(fake.astype(np.float32) - real.astype(np.float32)))
#     rmse = np.sqrt(mse)
#     psnr_val = psnr(real, fake, data_range=255)
#     ssim_val = ssim(real, fake, data_range=255)
#     return {'PSNR': psnr_val, 'SSIM': ssim_val, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     opt.num_threads = 0
#     opt.batch_size = 1
#     opt.serial_batches = True
#     opt.no_flip = True
#     opt.display_id = -1

#     dataset = create_dataset(opt)
#     model = create_model(opt)
#     model.setup(opt)

#     if opt.use_wandb:
#         wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
#         wandb_run._label(repo='Stego-GAN')

#     web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
#     if opt.load_iter > 0:
#         web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
#     print('creating web directory', web_dir)
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

#     if opt.eval:
#         model.eval()

#     dataset_size = len(dataset)
#     print('The number of testing images = %d' % dataset_size)

#     all_metrics = []
#     csv_file_path = os.path.join(web_dir, 'evaluation_metrics.csv')
#     with open(csv_file_path, mode='w', newline='') as csv_file:
#         fieldnames = ['Image', 'PSNR', 'SSIM', 'MAE', 'MSE', 'RMSE']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()

#         for i, data in enumerate(dataset):
#             if i >= opt.num_test:
#                 break
#             model.set_input(data)
#             model.test()

#             visuals = model.get_current_visuals()
#             img_path = model.get_image_paths()

#             fake_ct = tensor2im(visuals['fake_B'])
#             real_ct = tensor2im(data['A'])

#             metrics = compute_metrics(fake_ct, real_ct)
#             metrics['Image'] = os.path.basename(img_path[0])
#             writer.writerow(metrics)
#             all_metrics.append(metrics)

#             print(f"processing ({i:04d})-th image... {img_path[0]} | PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}")
#             save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

#         # Write average metrics at the end of CSV
#         avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in ['PSNR', 'SSIM', 'MAE', 'MSE', 'RMSE']}
#         avg_metrics['Image'] = 'Average'
#         writer.writerow(avg_metrics)

#     webpage.save()

#     # Print average metrics summary
#     print("\nEvaluation Summary:")
#     for k, v in avg_metrics.items():
#         if k != 'Image':
#             print(f"{k}: {v:.4f}")
