# Stochastic Gradient Estimation for Artificial Neural Networks 
## Introduction
We investigate a new approach to compute the gradients of artificial neural networks (ANNs), based on
the so-called push-out likelihood ratio method. Unlike the widely used backpropagation (BP) method that
requires continuity of the loss function and the activation function, our approach bypasses this requirement
by injecting artificial noises into the signals passed along the neurons. We show how this approach has
a similar computational complexity as BP, and moreover is more advantageous in terms of removing the
backward recursion and eliciting transparent formulas. We also formalize the connection between BP, a
pivotal technique for training ANNs, and infinitesimal perturbation analysis, a classic path-wise derivative
estimation approach, so that both our new proposed methods and BP can be better understood in the context
of stochastic gradient estimation. Our approach allows efficient training for ANNs with more flexibility on
the loss and activation functions, and shows empirical improvements on the robustness of ANNs under
adversarial attacks and corruptions of natural noises.

## Citation

If you find generalized likelihood ratio method useful in your research, please consider citing:

    @article{peng2020stochastic,
        Author = {Yijie Peng, Li Xiao, Bernd Heidergott,Jeff L. Hong, Henry Lam},
        Title = {Stochastic Gradient Estimation for Artificial Neural Networks},
        Journal = {Preprint with DOI: 10.2139/ssrn.3318847},
        Year = {2019}
    }
    
      @article{Li2020brain-like,
        Author = {Li Xiao, Yijie Peng,Jeff L. Hong, Zewu Ke, Shuhuai Yang},
        Title = {Training Artificial Neural Networks by Generalized Likelihood Ratio Method: Exploring Brain-like Learning to Improve Robustness},
        Journal = {IEEE International Conference on Automation Science and Engineering (CASE)},
        Year = {2020}
    } 

## Documents :
---MNIST&Fashion Experiments on MNISt and Fashion MNIST  
   ---train Training code，including BP，BP+，LRS，LRT，LRS with 0-1 loss， LRT with 0-1 loss  
   ---adv Testing on corruption and adversarial attacks  
---tinyImageNet Experiments on tinyImageNet 
   ---train Training code，including BP-1, BP-2, LRS, LRT, LRS with 0-1 loss, LRT with 0-1 loss, LR with relu  
   ---adv Testing on corruption and adversarial attacks   
---var statistical variance

## Operation environment 
python==3.7.4  pytorch==1.6.0  
Each file basically exists independently. Users can execute  run each file directly


## Experimental Results  
### MNIST  
#### Robustness on adversarial attacks
|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark| 96.34\%| 57.29\%| 28.37\%|  
| Sigmoid| 95.54\%| 77.60\%| 45.91\%|   
| Threhold|95.33\%|73.30\%|54.35\%|  

| Activations + 0-1 loss |  Orig |  AdvLBFGS |  AdvFGSM| 
|  ----  | ----  |  ----  | ----  |  
| Sigmoid| 84.54\%| 73.50 \%| 58.05\%|  
| Threhold| 90.22\%| 77.74\%| 61.36\%|  



### Robustness on corruption attacks
|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  
|BP becnmark|96.34\%|96.35\%|91.95\%|86.16\%|67.25\%|  
|Sigmoid|95.54\%|95.37\%|93.78\%|87.79\%|76.98\%|  
|Threhold|95.33\%|94.17\%|95.01\%|86.76\%|68.13\%|  

|Activations + 0-1 loss | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  
|Sigmoid|84.54\%|82.53\%|81.23\%|79.25\%|80.92\%|  
|Threhold|90.22\%|90.54\%|90.10\%|87.19\%|79.81\%|  

### FashionMnist

|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark|86.45\%|49.61\%|17.86\%|  
|Sigmoid|83.93\%|53.54\%|52.85\%|  
|Threhold|85.54\%|52.92\%|42.62|  

|Activations + 0-1 loss | Orig | AdvLBFGS | AdvFGSM|
|  ----  | ----  |  ----  | ----  |  
|Sigmoid|58.68\%|51.00\%|46.35\%|  
|Threhold|62.05\%|60.15\%|57.19\%|  


|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|BP becnmark|86.45\%|81.05\%|65.40\%|55.02\%|41.05\%|  
|Sigmoid|83.93\%|82.53\%|80.23\%|79.09\%|49.96\%|  
|Threhold|85.54\%|82.79\%|82.65\%|77.16\%|45.82\%|   

|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|Sigmoid|58.68\%|54.26\%|52.43\%|51.05\%|40.77\%|  
|Threhold|62.05\%|57.93\%|57.25\%|53.13\%|44.68\%|  


### TinyImageNet

|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark|21.70\%|11.6\%|8.70\%|  
|Sigmoid|20.30\%|14.6\%|17.04\%|  
|Threhold|17.30\%|15.6\%|14.34\%|  

Activations + 0-1 loss | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|Sigmoid|15.22\%|7.00 \%|14.35\%|  
|Threhold|19.57\%|13.6\%|16.52\%|  


|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|BP becnmark|21.70\%|12.68\%|12.52\%|13.40\%|11.36\%|  
|Sigmoid|20.30\%|17.47\%|17.52\%|17.68\%|16.28\%|  
|Threhold|17.30\%|16.24\%|16.48\%|16.96\%|14.88\%|  

|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|Sigmoid|15.22\%|14.56\%|14.47\%|14.68\%|14.40\%|  
|Threhold|19.57\%|15.40\%|15.48\%|15.92\%|13.99\%|  

