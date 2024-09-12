import os
import logging
import argparse
import numpy as np
from train_and_evaluate import evaluate, train
from net import Generator, BiomassPredictor
import utils
import torch
import pandas as pd
 


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='results', help="Results folder")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # Load the directory from commend line
    args = parser.parse_args()

    set_seed(100)

    # Set the logger
    utils.set_logger(os.path.join(args.output_dir, 'train.log'))

    # Load parameters from json file
    json_path = './param.json'
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Add attributes to params
    params.output_dir = args.output_dir
    params.cuda = torch.cuda.is_available()
    params.numIter = int(params.numIter)
    params.noise_dims = int(params.noise_dims)

    # make directory
    os.makedirs(args.output_dir, exist_ok = True)
    os.makedirs(args.output_dir + '/outputs', exist_ok = True)
    os.makedirs(args.output_dir + '/model', exist_ok = True)
    os.makedirs(args.output_dir + '/figures', exist_ok = True)
    

    bp = BiomassPredictor(params.num_wavelength+1, 1)
    # pre-trained biomass predictor model (from BioNet paper)
    bp.load_state_dict(torch.load("results/model/nn.pth"))
    bp.cuda()
    bp.eval()

    
    '''Different sliding window size
        tested values [3, 5, 7]'''
    for ker in [3]:#, 5, 7]:
        results = {}
        fn = params.output_dir + '/outputs/spectrum_sliding_window_'+str(ker)+'.csv'
        params.sliding_kernel = ker
        '''coefficent for loss computed if ppfd is more than 400
            tested range [0.0, 0.0001, 0.001,  0.01, 0.1, 1.0]'''
        for ppfd in [0.001]:
            '''coefficinet for loss corresponding to smooth curve
                tested range [0.0, 1.0, 10, 20, 50, 100]'''
            for smooth in [20]:
                params.ppfd = ppfd
                params.smooth = smooth
                suffix = str(params.ppfd) + "_" + str(params.smooth)
                print(ker, suffix)
                # Define the models 
                generator = Generator(params.noise_dims, params.noise_dims)
                    
                # Move to gpu if possible
                if params.cuda:
                    generator.cuda()

                # Define the optimizer
                optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr)
                
                # Define the scheduler
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma = params.gamma)

                
                # Train the model and save 
                if params.numIter != 0 :
                    logging.info('Start training')   
                    train(generator, optimizer, scheduler, bp, params)

                # Generate recipes and save 
                logging.info('Start generating devices')
                results["spec_"+suffix] = evaluate(1, params, bp)

        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv (fn, index = False, header=True)




