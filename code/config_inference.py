from datetime import datetime


class HyperParameter:
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')                      
        self.word2vec_pth = './data/model_300dim.pkl'
        
        self.pred_dataset = 'simple-case'
        self.sep = ','
        
        self.pred_pair_pth = './data/simple-Case/predict.csv'
        self.pair_col_name = ['drug_id', 'prot_id','drug_smile', 'prot_seq']
        
        self.pred_drug_dir = './data/EGFR-Case/drug.tsv'
        self.pred_prot_dir = './data/EGFR-Case/prot.tsv'
        self.d_col_name = ['drug_id', 'drug_smile']
        self.p_col_name = ['prot_id','prot_seq']
        
        self.model_fromTrain = './savemodel/All-kiba-Jan25_09-05-47.pth'        
                
        self.drug_max_len = 100
        self.substructure_max_len = 100
        self.prot_max_len = 1022
        self.mol2vec_dim = 300
        
        self.use_esmc = True
        self.esmc_model = "esmc_300m"
        
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
        
        self.cuda = "0"
