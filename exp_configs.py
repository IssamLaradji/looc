from haven import haven_utils as hu
import itertools, copy

EXP_GROUPS = {}

       
EXP_GROUPS["looc_trancos"] = hu.cartesian_exp_group({
        'batch_size': 1,
        'num_channels':1,
        'dataset': [
                {'name':'trancos', 'n_classes':1},
                    ],
        "attention": [{'name':"semseg",  
                                                'ratio_top':0.01, 
                                                'select_prob':0.5,
                                                'agg_score_method':'mean', 
                                                'box_score_method':'center'}],
        'dataset_size':[
                # {'train':10, 'val':1, 'test':1},
                {'train':'all', 'val':'all'}
                ],
        'max_epoch': [100],
        'optimizer': [ "adam"], 
        'lr': [1e-5],
        'model': {'name':'semseg_looc',
                        'n_classes':1, 'base':'fcn8_vgg16', 'n_channels':3, 
                        'loss':'att_lcfcn'},
        })
