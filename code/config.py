from datetime import datetime


class HyperParameter:
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.kfold = 5 

        self.data_root = './data/5fold_split'  
        self.dataset = 'davis'
        self.running_set = 'novel-pair'
        self.dataset_columns = ['drug_id', 'prot_id', 'label']
        self.is_esm = True
        
        self.use_esmc = False
        self.esmc_model = "esmc_6b"
        
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_drug_pretrain.pkl'
        
        if self.use_esmc:
            if self.esmc_model == "esmc_6b":
                self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_6b_pretrain.pkl'
            elif self.esmc_model == "esmc_600m":
                self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_600m_pretrain.pkl'
            else:
                self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_pretrain.pkl'
        else:
            self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'           
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'   

        self.Learning_rate = 1e-4
        self.Epoch = 1
        self.Batch_size = 256
        self.max_patience = 20

        self.drug_max_len = 100
        self.substructure_max_len = 100
        self.prot_max_len = 1022
        self.mol2vec_dim = 300
        
        if self.use_esmc:
            if self.esmc_model == "esmc_300m":
                self.protvec_dim = 960
            elif self.esmc_model == "esmc_600m":
                self.protvec_dim = 1152
            elif self.esmc_model == "esmc_6b":
                self.protvec_dim = 2560
        else:
            self.protvec_dim = 1280
        
        self.latent_dim = 512     
        self.com_dim = 2048
        self.mlp_dim = [1024, 512, 1]

        self.num_experts = 4
        self.top_k = 2
        self.moe_noise_std = 0.1
        self.load_balance_weight = 0.01

        self.cuda = "1"

    def set_dataset(self, data_name):
        self.dataset = data_name
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_drug_pretrain.pkl'
        
        if self.use_esmc:
            if self.esmc_model == "esmc_6b":
                self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_6b_pretrain.pkl'
            elif self.esmc_model == "esmc_600m":
                self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_600m_pretrain.pkl'
            else:
                self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_pretrain.pkl'
        else:
            self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'
            
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'
