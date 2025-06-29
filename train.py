"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.metrics import compute_ssim, compute_psnr, compute_mae, compute_mse, compute_rmse
from tqdm import tqdm

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('Seed:{}'.format(torch.seed()))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if opt.checkpoint is not None:
        model.load_networks(opt.checkpoint)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        tqdm_dataset = tqdm(dataset)
        for i, data in enumerate(tqdm_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                tqdm_dataset.set_postfix({
                    "losses":message
                })
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            
            if total_iters % opt.checkpoint_freq == 0:
                print('saving model checkpoint')
                model.save_networks("checkpoint")

            iter_data_time = time.time()
        
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks(epoch)

        # Evaluate metrics at end of epoch (optional visualization/logging)
        print(f"\n📊 Evaluating metrics for epoch {epoch}...")
        sample_data = next(iter(dataset))
        model.set_input(sample_data)
        model.forward()
        model.compute_visuals()
        visuals = model.get_current_visuals()

        real_img = visuals.get('real_A')
        if real_img is None:
            real_img = visuals.get('real')
        fake_img = visuals.get('fake_B')
        if fake_img is None:
            fake_img = visuals.get('fake')

        if real_img is not None and fake_img is not None:
            real_tensor = real_img[0].to(device)
            fake_tensor = fake_img[0].to(device)

            ssim = compute_ssim(real_tensor, fake_tensor)
            psnr = compute_psnr(real_tensor, fake_tensor)
            mae  = compute_mae(real_tensor, fake_tensor)
            mse  = compute_mse(real_tensor, fake_tensor)
            rmse = compute_rmse(real_tensor, fake_tensor)

            print(f"✅ Epoch {epoch} Metrics:")
            print(f"    SSIM: {ssim:.4f}")
            print(f"    PSNR: {psnr:.2f}")
            print(f"    MAE : {mae:.4f}")
            print(f"    MSE : {mse:.4f}")
            print(f"    RMSE: {rmse:.4f}")

            with open("metrics_log.csv", "a") as f:
                if epoch == opt.epoch_count:
                    f.write("epoch,ssim,psnr,mae,mse,rmse\n")
                f.write(f"{epoch},{ssim:.4f},{psnr:.2f},{mae:.4f},{mse:.4f},{rmse:.4f}\n")

            if opt.display_id > 0:
                visualizer.plot_epoch_metrics(epoch, {
                    'SSIM': ssim,
                    'PSNR': psnr,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                })

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
