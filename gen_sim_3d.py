import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mrc_insar_common.data import data_reader


@jit(nopython=True)  #to fast the code like C. nopython=true is used for not falling back numba
def wrap(phase):
    return np.angle(np.exp(1j * phase)) #return angle value in radians and np.exp returns only complex value

def gen_sim_3d(mr,
               he,
               stack_length,
               bperp_scale=2000.,
               dday_stepmax=4,
               dday_scale=11,
               conv1_scale=-0.0000573803,
               conv1_shift=-0.0110171730107,
               conv2_scale=-0.00073405573,
               conv2_shift=-0.00086772422789899997):
    """gen_sim_3d.
    Generate simulated unwraped recon phase with given mr and he
    Args:
        mr: motion rate with shape [H, W]
        he: height error with shape [H, W]
        stack_length: length of stack size
    """
    ddays = np.random.randint(low=1, high=dday_stepmax, size=stack_length) * dday_scale  # shape: [stack_length]
    bperps = (np.random.rand(stack_length) -0.5) * bperp_scale  # shape: [stack_length]
    conv1 = np.random.rand() * (conv1_scale) + conv1_shift #random single number
    conv2 = np.random.rand() * (conv2_scale) + conv2_shift #random simgle number
    unwrap_recon_phase = conv1 * ddays * (np.expand_dims(mr, -1)) + conv2 * bperps * (np.expand_dims(he, -1))  # shape: [H, W, stack_length]
    
    return unwrap_recon_phase, ddays, bperps, conv1, conv2 # end of function




if __name__ == "__main__":
    # load 3vG mr and he
    mr_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/elabra.tsx.sm_dsc.4624.118.1500.1500/fit_hr/def_fit_cmpy' 
    he_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/elabra.tsx.sm_dsc.4624.118.1500.1500/fit_hr/hgt_fit_m' 

    mr = data_reader.readBin(mr_path, 1500, 'float') #shape(1500,1500)
    he = data_reader.readBin(he_path, 1500, 'float')
    
    

    SIM_STACK_LEN = 10 

    # check gen_sim_3d interface and implementation at: https://github.com/UAMRC-3vG/MRC-InSAR-Common/blob/main/mrc_insar_common/util/sim.py#L11
    # unwrap_phase, ddays, bperps, conv1, conv2 = sim.gen_sim_3d(mr, he, SIM_STACK_LEN)
    unwrap_phase, ddays, bperps, conv1, conv2 = gen_sim_3d(mr, he, SIM_STACK_LEN)


    # vsulize sample patchs in a batch
    print(wrap(unwrap_phase).shape)
    print(ddays.shape)
    print(bperps.shape)
    print(conv1)
    print(conv2)
    # print(mr)
    fig, axs = plt.subplots(1,4, figsize=(9,2))
    for i in range(SIM_STACK_LEN): # size of stack
        # im = axs[i].imshow(wrap(unwrap_phase[:,:, i]), cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='None')
        im = axs[i].imshow(wrap(unwrap_phase[:,:, i]), cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()
    # plt.savefig('/home/niloy/Desktop/ps.mr.png',dpi=200,bbox_inches='tight')
