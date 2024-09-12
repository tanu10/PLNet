import os
import logging
from tqdm import tqdm
import torch
import utils
import numpy as np
from net import Generator, SlidingSum

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def evaluate(num_sample, params, bp):
    model_path = os.path.join(params.output_dir, 'model', 'model.pth.tar')
    ker = params.sliding_kernel
    sliding = SlidingSum(ker, 1)
    generator = Generator(params.noise_dims, params.noise_dims)
    checkpoint = torch.load(model_path)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    generator.eval()
    generator.cuda()
    
    # generate spectrum
    z = sample_z(num_sample, params)
    spectrum = generator(z)

    age_tensor = torch.from_numpy(np.full((num_sample,1), 1.0)).float().cuda()
    pred_input = torch.cat((age_tensor, spectrum), 1)
    
    logging.info('Generation is done. \n')
    predicted_biomass = bp(pred_input)
    bm = predicted_biomass.item()
    print(bm)

    spectrum = np.mean(spectrum.cpu().detach().numpy(), axis=0)
    ''' As light recipes were normalized using maximum value present in the data (6.783), scale them back.'''
    spectrum *= 6.783
    print(spectrum.sum())
    fig_path = params.output_dir + '/figures/' + str(
        params.ppfd) + "_" + str(params.smooth)
    utils.plot_spectrum(spectrum, bm, fig_path+".png")
    return spectrum

def train(generator, optimizer, scheduler, bp, params):

    generator.train()
    # width of sliding window, odd number
    ker = params.sliding_kernel
    sliding = SlidingSum(ker, 1)
    s_loss = torch.nn.L1Loss()
    iter0 = 0
    # training loop
    with tqdm(total=params.numIter) as t:
        it = 0  
        while True:
            it +=1 
            params.iter = it + iter0

            # save model 
            if it % 5000 == 0:
                model_dir = os.path.join(params.output_dir, 'model','iter{}'.format(it+iter0))
                os.makedirs(model_dir, exist_ok = True)
                utils.save_checkpoint({'iter': it + iter0 - 1,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_state_dict': optimizer.state_dict(),
                                       'scheduler_state_dict': scheduler.state_dict(),
                                       },
                                       checkpoint=model_dir)
            if it > params.numIter:
                model_dir = os.path.join(params.output_dir, 'model')
                os.makedirs(model_dir, exist_ok = True)
                utils.save_checkpoint({'iter': it + iter0 - 1,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_state_dict': optimizer.state_dict(),
                                       'scheduler_state_dict': scheduler.state_dict(),
                                       },
                                       checkpoint=model_dir)
            # terminate the loop
            if it > params.numIter:
                return 
           
            # sample  z
            z = sample_z(params.batch_size, params)           

            # generate a batch of recipes
            gen_spectrum = generator(z)

            sliding_avg = torch.squeeze(sliding(gen_spectrum))/ker

            age_tensor = torch.from_numpy(np.full((params.batch_size,1), 1.0)).float().cuda()
            pred_input = torch.cat((age_tensor, gen_spectrum), 1)

            # free optimizer buffer 
            optimizer.zero_grad()

            predicted_biomass = bp(pred_input) 
            x = torch.sum(gen_spectrum, axis=1)

            ''' keep ppfd less than 400. while training, ppfd was normalized with max value (6.783)'''
            ppfd_loss = torch.where(x-59 > 0, x-59, 0).mean()

            ''' try to keep curve of the generated recipe smooth'''
            smoothness_loss = s_loss(sliding_avg, gen_spectrum)

            ''' learn to predict high biomass, hence minimize the negative of predicted biomass'''
            pred_biomass_loss = -(predicted_biomass).mean()           

            g_loss = pred_biomass_loss + params.ppfd * ppfd_loss + params.smooth * smoothness_loss 

            # train the generator
            g_loss.backward()
            optimizer.step()

            # learning rate decay
            scheduler.step()

            t.set_postfix_str(
                f"Biomass {pred_biomass_loss.item():.3f}, Smoothness {smoothness_loss.item():.3f}, PPFD {ppfd_loss.item():.4f}")
            t.update()

def sample_z(batch_size, params):
    '''
    sample noise vector z
    '''
    return (torch.rand(batch_size, params.noise_dims).type(Tensor)*2.-1.) * params.noise_amplitude

