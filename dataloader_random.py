from mrc_insar_common.data import data_reader
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import glob
# import tqdm
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mrc_insar_common.data import data_reader
from gen_sim_3d import gen_sim_3d, wrap


''' wrap finction is used to return the wrap phase from unwrap phase '''

# @jit(nopython=True)  # to fast the code like C. nopython=true is used for not falling back numba
# def wrap(phase):
#     return np.angle(np.exp(1j * phase)) # return angle value in radians and np.exp returns only complex value



# ''' gen_sim_3d generates simulated unwraped recon phase with given mr and he
#             Args:
#                 mr: motion rate with shape [H, W]
#                 he: height error with shape [H, W]
#                 stack_length: length of stack size '''

# def gen_sim_3d(mr,
#                he,
#                stack_length,
#                bperp_scale  =   2000.,
#                dday_stepmax =   4,
#                dday_scale   =   11,
#                conv1_scale  =   -0.0000573803,
#                conv1_shift  =   -0.0110171730107,
#                conv2_scale  =   -0.00073405573,
#                conv2_shift  =   -0.00086772422789899997
#                ):
           
#                 ddays = np.random.randint(low = 1, high = dday_stepmax, size = stack_length) * dday_scale  # shape: [stack_length]
#                 bperps = (np.random.rand(stack_length) -0.5) * bperp_scale  # shape: [stack_length]
#                 conv1 = np.random.rand() * (conv1_scale) + conv1_shift #random single number
#                 conv2 = np.random.rand() * (conv2_scale) + conv2_shift #random simgle number
#                 unwrap_recon_phase = conv1 * ddays * (np.expand_dims(mr, -1)) + conv2 * bperps * (np.expand_dims(he, -1))  # shape: [H, W, stack_length]
                
#                 return unwrap_recon_phase, ddays, bperps, conv1, conv2 # end of function


 
class SpatialTemporalDataset(Dataset): # inherit the dataset class

    def __init__(   self,
                    filt_dir,
                    filt_ext, 
                    coh_dir,
                    coh_ext,
                    width,
                    height,
                    ref_mr_path,
                    ref_he_path,
                    patch_size = 38,
                    stride = 0.5
                    ):


        self.filt_paths = sorted(glob.glob('{}/*{}'.format(filt_dir, filt_ext))) #  provide a list of sorted data file path of filt_ext fometed file 
        # self.bperp_paths = sorted(glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))# provide a list of sorted data file path of bperp_ext fometed file 
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))# rovide a list of sorted data file path of coh_ext fometed file 
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path


    

        # self.ref_mr_path = data_reader.readBin(ref_mr_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))
             # crop function mainly crop from the coord coordinator like (x,y) = (904,532) and crop patch_size * patch_size
        # self.ref_he_path = data_reader.readBin(ref_he_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))
        # self.conv1 = conv1
        # self.conv2 = conv2
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.stride = stride

        self.stack_size = len(self.filt_paths) #find out the lenght of the sorted self.filt_path                                                                                  self.stack_size)
                                                                                        
        # self.ddays = np.zeros(self.stack_size) #make a 1d list 0 of length of 
        # self.bperps = np.zeros(self.stack_size)


        # for idx in tqdm.tqdm(range(self.stack_size)): #tqdm >>it's just add a animation of loading 
            # read delta days
            # bperp_path = self.bperp_paths[idx]# taking every element in the list one by one 
            # date_string = bperp_path.split('/')[-1].replace(bperp_ext, "") 
            # split the the element in the list with respect to "/" [-1] represent the last element in the splited list
            # then replace the extention(bparp_ext) part with space 


            # delta_day = get_delta_days(date_string)
            # self.ddays[idx] = delta_day

            # read bperp
            # self.bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]
            # print(self.bperps)
        
        self.all_sample_coords = [(row_idx, col_idx)
                                  for row_idx in range(0, self.height - self.patch_size - 1, int(self.patch_size * stride)) 
                                    # range(start,end,skip)
                                    # 0,1500-28-1, int(28*0.5) 
                                    # 0,1471,14
                                    # to see the example please run this code
                                    # x =[(i,j) for i in range(0,1471,14) for j in range(0,1471,14)]
                                    # print(x)
                                  for col_idx in range(0, self.width - self.patch_size - 1, int(self.patch_size * stride))]

   
 
    def __len__(self):
        return len(self.all_sample_coords)

    def __getitem__(self, idx):
        coord = self.all_sample_coords[idx] # print (coord) # print coord to check the output of the coord
        mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size)) # crop function mainly crop from the coord coordinator like (x,y) = (904,532) and crop patch_size * patch_size
        he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))


        # mr = data_reader.readBin(self.ref_mr_path, self.width, 'float') # crop function mainly crop from the coord coordinator like (x,y) = (904,532) and crop patch_size * patch_size
        # he = data_reader.readBin(self.ref_he_path, self.width, 'float')
        
        filt_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])  # [N, h ,w] for a single training sample, 
        coh_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])   # [N, h ,w] for a single training sample

        self.unwrap_recon_phase, self.ddays, self.bperps, self.conv1, self.conv2 = gen_sim_3d(mr_target, he_target, self.stack_size)
        # self.unwrap_recon_phase, self.ddays, self.bperps, self.conv1, self.conv2 = gen_sim_3d(mr, he,self.stack_size)

        wrap_recon_phase = wrap(self.unwrap_recon_phase)

        for i in range(self.stack_size):
            # !! here is an example that only uses phase information 
            filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(coord[0], coord[1], self.patch_size, self.patch_size)))  # MRC InSAR Library - https://pypi.org/project/MRC-InSAR-Common/  follow this link  for datareader of readbin

            coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))

        return{ 'input' : filt_input, # 3D data   
                'coh'   : coh_input, # feature 3D
                'mr'    : np.expand_dims(mr_target, 0), # label expand dims is used for convert the array in matrix
                                                        # axis = 0 means increase in column, axis = 1 means increase in row
                                                        # 'mr': np.array(mr_target),
                # 'mr'    : mr_target,
                'he'    : np.expand_dims(he_target, 0), # same as mr
                'ddays' : self.ddays,    # ddays and bperps are shared for all training samples in a stack, it can be used in a more effecient way, here is just an example
                'bperps': self.bperps,   # feature single value 
                'conv1' : self.conv1,
                'conv2' : self.conv2,
                'unwrap_recon_phase': self.unwrap_recon_phase,
                'wrap_recon_phase'  : wrap_recon_phase
                }

if __name__ == "__main__":
 
    db = SpatialTemporalDataset(filt_dir    =   '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/',
                                filt_ext    =   '.diff.orb.statm_cor.natm.filt', 
                                coh_dir     =   '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/',
                                coh_ext     =   '.diff.orb.statm_cor.natm.filt.coh',
                                width       =   1500,
                                height      =   1500,
                                ref_mr_path =   '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/def_fit_cmpy',
                                ref_he_path =   '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/hgt_fit_m',
                                patch_size  =   28,
                                stride      =   0.5
                                )
    
    random_dataloader = DataLoader( db,
                                    batch_size  =   4, 
                                    shuffle     =   True,
                                    num_workers =   4
                                    )

    
    
    


    ''' Output & Visualize the data dtructure'''
    
    
    print('db length {}'.format(len(db)))
    print('type of db {}'.format(type(db)))


    for batch_idx, batch in enumerate(random_dataloader):
        print('Batch Index = {}'.format(batch_idx))
        print('Input Shape = {}'.format(batch['input'].shape))
        print('Coh Shape = {}'.format(batch['coh'].shape))
        print('mr Shape = {}'.format(batch['mr'].shape))
        # print(batch['mr']) # to check the output of mr 
        print('he Shape = {}'.format(batch['he'].shape))
        print('ddays Shape = {}'.format(batch['ddays'].shape))
        print('bperps Shape = {}'.format(batch['bperps'].shape))
        print('conv1 Shape = {}'.format(batch['conv1'].shape))
        print('conv2 Shape = {}'.format(batch['conv2'].shape))
        print('Wrap recon phase Shape = {}'.format(batch['wrap_recon_phase'].shape))


        # print(np.angle(1*np.exp(batch['input'][0][0]-batch['wrap_recon_phase'][0][0])))

        # print(np.angle(1*np.exp(batch['input'][0][0][0]-wrap(batch['unwrap_recon_phase'][0][0][0]))))
        # for i in range(len(batch['input'])):
        #     for j in range(len(batch['input'][i])):
        #         for k in range(28):
        #             print(batch["input"][i][j][k])

        break
  


    ''' vsulize sample patches in a batch'''


    fig, axs = plt.subplots(1,4, figsize=(8,2))
    input_shape = batch['input'][0].shape # first training example
    # print (input_shape[0])
    for i in range(input_shape[2]): # size of stack
        im = axs[i].imshow(batch['input'][0][i], cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()
    

    fig, axs = plt.subplots(1,4, figsize=(8,2))
    input_shape = batch['coh'][0].shape # first training example
    for i in range(input_shape[2]): # size of stack
        im = axs[i].imshow(batch['coh'][0][i], cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()


    # fig, axs = plt.subplots(1,4, figsize=(9,2))
    # input_shape = batch['wrap_recon_phase'][0].shape 
    # for i in range(input_shape[2]): # size of stack
    #     im = axs[i].imshow(batch['wrap_recon_phase'][:,:,i]), cmap='jet', vmin=-np.pi, vmax=np.pi)
    #     fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
    #     if i == 3: 
    #         break
    # fig.tight_layout()
    # plt.show()
    

    # idea to plot the the difference between input and recon phase
    
    # def wrap(phase):
    #     return np.angle(np.exp(1j * phase))

    # def recon_phase(self, mr, he, conv1, conv2, ddays):
    #     return self.ddays * self.conv1 * mr + self.bperps * self.conv2 * he

    # recon_phase = self.recon_phase(ref_mr_path, ref_he_path, sample_conv1, sample_conv2, 'ddays')

    # np.angle(1*np.exp(filter_ifg_phase-wrap(recon_ifg_ifg)))