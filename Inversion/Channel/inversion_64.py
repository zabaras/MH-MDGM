import numpy as np 
from vallina_mcmc import inference
import argparse
import os
import torch
from mean_std import mean_std_plot
from error_bars import plot_error_bar



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim',type = int ,default = 256,help = 'dim of prams')
    parser.add_argument('--steps',type = int ,default = 30000,help = 'steps')
    parser.add_argument('--burn_in',type = int ,default = 28000,help = 'burn_in')
    parser.add_argument('--proposal',type = str ,default = 'pCN',help = 'proposal_type')
    parser.add_argument('--scale',type = str ,default = 'scale_64',help = 'scale_size')
    parser.add_argument('--noise',type = str ,default = '0.05',help = 'scale_size')
    parser.add_argument('--exp',type = str ,default = 'inversion_64',help = 'scale_size')
    args = parser.parse_args()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") # training on GPU or CPU
    # load experiment data#
    obs = np.loadtxt(os.getcwd()+f'/Channel/test_data/obs_{args.noise}.dat')
    sigma = np.loadtxt(os.getcwd()+f'/Channel/test_data/sigma_{args.noise}.dat')
    true_permeability = np.loadtxt(os.getcwd()+f'/Channel/test_data/true_permeability_{args.noise}.dat')
    true_pressure = np.loadtxt(os.getcwd()+f'/Channel/test_data/true_pressure_{args.noise}.dat')
    obs_position = np.loadtxt(os.getcwd()+f'/Channel/test_data/obs_position_{args.noise}.dat')
    inference = inference(args.steps,args.burn_in,args.dim,obs,sigma,true_permeability,true_pressure,obs_position,args.scale,device)
    np.random.seed(22)
    init_state = np.random.randn(args.dim,)
    if args.proposal == 'pCN':
        beta = 0.08
    output_dir = os.getcwd()+ f'/Channel/plot_inversion_state/{args.exp}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # samples, post_samples_z,fx_obs,accept, log_likelihood = inference.pCN_MH(init_state,beta,args.exp,args.noise,output_dir)
    save_dir = os.getcwd()+f'/Channel/prediction/{args.exp}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save inference result #
    # np.savetxt(save_dir+f'/{args.proposal}_step{beta}_{args.noise}_{args.steps}_{args.burn_in}_all_samples.dat',samples)
    # np.savetxt(save_dir+f'/{args.proposal}_step{beta}_{args.noise}_{args.steps}_{args.burn_in}_aceeptratio.dat',accept)
    # np.savetxt(save_dir+f'/{args.proposal}_step{beta}_{args.noise}_{args.steps}_{args.burn_in}_fx_obs.dat',fx_obs)
    # np.savetxt(save_dir+f'/{args.proposal}_step{beta}_{args.noise}_{args.steps}_{args.burn_in}_log_likelihood.dat',log_likelihood)
    # np.savetxt(save_dir+f'/{args.proposal}_step{beta}_{args.noise}_{args.steps}_{args.burn_in}_posterior.dat',post_samples_z)
    # plot mean std samples #
    post_samples_z = np.loadtxt(save_dir+f'/{args.proposal}_step{beta}_{args.noise}_{args.steps}_{args.burn_in}_posterior.dat')

    
    scale_size = 64
    plot = mean_std_plot(device,scale_size,args.exp,args.noise)
    pos_samples_x = plot.compute_mean_std(post_samples_z,true_permeability,save_dir)
    # plot error bar #
    filename = '/channel_64_errorbar.pdf'
    plot_err_bar = plot_error_bar(device,save_dir,filename)
    plot_err_bar.error_bar(true_permeability,pos_samples_x)
