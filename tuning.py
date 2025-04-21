import numpy as np
import time
from options.tuning_options import TuningOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import util

if __name__ == '__main__':
    opt = TuningOptions().parse()   # get training options

    # Create tuning and validation datasets
    dataset = create_dataset(opt, 'tuning')
    validation = create_dataset(opt, 'valid')

    dataset_size = len(dataset)
    validation_size = len(validation)

    print('The number of tuning images = %d' % dataset_size)
    print('The number of validation images = %d' % validation_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    total_iters = 0
    best_psnr = 0
    best_epoch = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Validation Evaluation
        if epoch % opt.valid_epoch_freq == 0:
            valid_psnr = 0
            model.decoder.eval()
            np.random.seed(11)

            valid_name = opt.valid_name
            valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20, "configs/light.yaml": 1}
            repeat = valid_repeat_times.get(valid_name, 1)

            for _ in range(repeat):
                for i, data in enumerate(validation):
                    model.set_input_val(data)
                    psnr, _ = model.forward_psnr(False)
                    valid_psnr += psnr

            valid_psnr /= validation_size * repeat

            if valid_psnr > best_psnr:
                print('saving the best model of the best epoch %d, iters %d' % (epoch, total_iters))
                best_psnr = valid_psnr
                best_epoch = epoch
                model.save_networks('best')
                model.save_ema('best')
                model.save_state('best')

            print('epoch %d / %d \t Validation: %.4f dB Best_PSNR (Best epoch %d): %.4f dB' %
                  (epoch, opt.n_epochs + opt.n_epochs_decay, valid_psnr, best_epoch, best_psnr))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            model.save_ema('latest')
            model.save_ema(epoch)
            model.save_state('latest')
            model.save_state(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        model.update_learning_rate()
