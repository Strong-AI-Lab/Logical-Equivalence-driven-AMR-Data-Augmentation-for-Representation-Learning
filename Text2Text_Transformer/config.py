
class Config:
    # 0. gpu
    use_gpu = True
    gpu_id = "5"
    #device = torch.device("cuda:{}".format(gpu_id) if (torch.cuda.is_available() and use_gpu) else "cpu")

    flag = "finetune_t5_large"  ## finetune_gpt2, finetune_t5
    ## Initialization parameters
    model_type = 't5-large'  # same as model_name_or_path
    checkpoint_num = "checkpoint-2839"
    dataset_path = './output_result/'  ## if the task is sen2rdf, set the dataset_path to 'data/webnlg_challenge_2017_sen2rdf/'
    rdf2sen_augment_data_tag = False
    dart_rdf2sen_augment_data_tag = False
    sen2rdf_augment_data_tag = False
    lower_threshold = 50
    upper_threshold = 80
    coverage = 0.8
    augment_tag = "contraposition" # contraposition, all
    if augment_tag == "contraposition":
        # train_data_file = "/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/output_result/Synthetic_xfm_t5wtense_logical_equivalence_list_contraposition_train.csv"
        train_data_file = "/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/Synthetic_xfm_t5wtense_contraposition_list_text2text_train.csv"
        eval_data_file = "/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/output_result/Synthetic_xfm_t5wtense_logical_equivalence_list_contraposition_dev.csv"
        test_data_path = "/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/output_result/Synthetic_xfm_t5wtense_logical_equivalence_list_contraposition_train.csv"
    elif augment_tag == "all":
        train_data_file = "/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/Synthetic_xfm_t5wtense_logical_equivalence_list_text2text_train.csv"
        eval_data_file = "/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/output_result/Synthetic_xfm_t5wtense_logical_equivalence_validation_v5.csv"
        test_data_path = "/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/output_result/Synthetic_xfm_t5wtense_logical_equivalence_train_v5.csv"
    
    
    paraphrased_tag = False
    num_return_sequences = 5  # 5
    if flag == "finetune_gpt2" and rdf2sen_augment_data_tag != True and paraphrased_tag == False and augment_tag == "all":
        directory_name = "model_gpt2"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"  ## dataset_path + directory_name + "/" + checkpoint_num + "/" + "evaluate_total_test_result.csv"
    elif flag == "finetune_gpt2_large" and rdf2sen_augment_data_tag != True and paraphrased_tag == False and augment_tag == "all":
        directory_name = "model_gpt2_large"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"  ## dataset_path + directory_name + "/" + checkpoint_num + "/" + "evaluate_total_test_result.csv"
    elif flag == "finetune_t5_base" and rdf2sen_augment_data_tag != True and paraphrased_tag == False and augment_tag == "all":
        directory_name = "model_t5_base"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"  ## dataset_path + directory_name + "/" + checkpoint_num + "/" + "evaluate_total_test_result.csv"
    elif flag == "finetune_t5_large" and rdf2sen_augment_data_tag != True and paraphrased_tag == False and augment_tag == "contraposition":
            directory_name = "model_t5_large_contraposition"
            output_dir = dataset_path+directory_name
            model_path = dataset_path+directory_name+"/"+checkpoint_num
            output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
            evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"  ## dataset_path + directory_name + "/" + checkpoint_num + "/" + "evaluate_total_test_result.csv"    
    elif flag == "finetune_t5_large" and rdf2sen_augment_data_tag != True and paraphrased_tag == False and augment_tag == "all":
            directory_name = "model_t5_large_all"
            output_dir = dataset_path+directory_name
            model_path = dataset_path+directory_name+"/"+checkpoint_num
            output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
            evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"  ## dataset_path + directory_name + "/" + checkpoint_num + "/" + "evaluate_total_test_result.csv"    
            
    num_train_epochs = '30'
    block_size = '512'
    save_steps = '2000'
    logging_steps = '10'
    learning_rate = '0.00001'
    gradient_accumulation_steps = '4'
    per_gpu_train_batch_size = '6'
    per_gpu_eval_batch_size = '6'
    sts_model = 'paraphrase-distilroberta-base-v1'

    ## The parameter for Generate function
    test_batch_size = 16
    min_length = 10
    max_length = 100
    do_sample = True
    top_p = 0.9
    top_k = 40

    # This configuration is used to define the number of predictions we want to evaluate on while probing bleu score
    # See generate.generate_predictions_for_bleu_score_probing(..) for details
    test_num_return_sequences = 2#2
    logging_steps_generate = 50

    ## The parameter for evaluate function
    logging_steps_evaluate = 500
    

def parse(self, kwargs):
    '''
    user can update the default hyperparamter
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            raise Exception('opt has No key: {}'.format(k))
        setattr(self, k, v)

    print('*************************************************')
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print("{} => {}".format(k, getattr(self, k)))

    print('*************************************************')


Config.parse = parse
opt = Config()