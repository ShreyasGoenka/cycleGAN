import time
import torch

from dataset import Office31 
from cycle_gan import CycleGAN

num_epochs = 10
print_freq = 50
# save_latest_freq = 1500

ga = []
gb = []
da = []
db = []
cyclea = []
cycleb = []


if __name__ == '__main__':
    
    dataset = Office31('./office31_features/amazon_amazon.csv', './office31_features/amazon_dslr.csv')
    model = CycleGAN()

    print("Model Init")
    print(model.netG_A)
    print(model.netD_A)

    total_iters = 0                # the total number of training iterations

    for epoch in range(0, num_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # total_iters += opt.batch_size
            # epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % print_freq == 0:   # display images on visdom and save images to a HTML file
                losses = model.get_current_losses()
                ga.append(model.loss_G_A)
                gb.append(model.loss_G_B)
                da.append(model.loss_D_A)
                db.append(model.loss_D_B)
                cyclea.append(model.loss_cycle_A)
                cyclea.append(model.loss_cycle_B)
                # print(losses)



            # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #     visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            #     if opt.display_id > 0:
            #         visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # if total_iters % save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()
        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        losses = model.get_current_losses()
        print("loss ", "\nG: ", losses['G_A']+losses['G_B'], "\nD: ", losses['D_A']+losses['D_B'], "\ncycle ", losses['cycle_A']+losses['cycle_B'])
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
