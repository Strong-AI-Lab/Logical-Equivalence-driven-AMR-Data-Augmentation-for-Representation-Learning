import numpy as np
import pandas as pd

ensemble_list = []

debertav2_our_model_pn_1_3 = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-our-model-v5-pos-neg-1-3/test_preds.npy")
debertav2_our_model_pn_1_2 = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-our-model-v5-pos-neg-1-2/test_preds.npy")
debertav2_xxlarge_contraposition = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-contraposition/test_preds.npy")
debertav2_xxlarge_contraposition_double_negation_implication = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-contraposition-double-negation-implication/test_preds.npy")
deberta_v2_xxlarge_our_model_v5_bs_8_lr_3e_6 = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/deberta-v2-xxlarge-our-model-v5-bs-8-lr-3e-6/test_preds.npy")
deberta_v2_xxlarge_our_model_v5_bs_8_lr_3e_6_merged = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/deberta-v2-xxlarge-our-model-v5-bs-8-lr-3e-6-merged/test_preds.npy")
# debertav2_xxlarge_contraposition_double_negation = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-contraposition-double-negation/test_preds.npy")

for item in range(len(debertav2_our_model_pn_1_3)):
    model1 = debertav2_our_model_pn_1_3[item]
    model2 = debertav2_our_model_pn_1_2[item]
    model3 = debertav2_xxlarge_contraposition[item]
    model4 = debertav2_xxlarge_contraposition_double_negation_implication[item]    
    model5 = deberta_v2_xxlarge_our_model_v5_bs_8_lr_3e_6[item]
    model6 = deberta_v2_xxlarge_our_model_v5_bs_8_lr_3e_6_merged[item]
    # model6 = debertav2_xxlarge_contraposition_double_negation[item]
    
    dict_list = []
    dict_list.append(model1)
    dict_list.append(model2)
    dict_list.append(model3)
    dict_list.append(model4)
    dict_list.append(model5)
    dict_list.append(model6)
    
    maxlabel = max(dict_list,key=dict_list.count)
    ensemble_list.append(maxlabel)

final_numpy = np.array(ensemble_list)
np.save("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/ensemble_contraposition_1_2_1_3_our_model_v5_con_dou_imp_merged/merged_test_predict_data.npy", final_numpy)




# import numpy as np
# import pandas as pd

# ensemble_list = []

# debertav2_our_model_pn_1_3 = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-our-model-v5-pos-neg-1-3/test_preds.npy")
# debertav2_our_model_pn_1_2 = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-our-model-v5-pos-neg-1-2/test_preds.npy")
# debertav2_xxlarge_contraposition = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-contraposition/test_preds.npy")
# debertav2_xxlarge_contraposition_double_negation_implication = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-contraposition-double-negation-implication/test_preds.npy")
# deberta_v2_xxlarge_our_model_v5_bs_8_lr_3e_6 = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/deberta-v2-xxlarge-our-model-v5-bs-8-lr-3e-6/test_preds.npy")
# # debertav2_xxlarge_contraposition_double_negation = np.load("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/debertav2-xxlarge-contraposition-double-negation/test_preds.npy")

# for item in range(len(debertav2_our_model_pn_1_3)):
#     model1 = debertav2_our_model_pn_1_3[item]
#     model2 = debertav2_our_model_pn_1_2[item]
#     model3 = debertav2_xxlarge_contraposition[item]
#     model4 = debertav2_xxlarge_contraposition_double_negation_implication[item]    
#     model5 = deberta_v2_xxlarge_our_model_v5_bs_8_lr_3e_6[item]
#     # model6 = debertav2_xxlarge_contraposition_double_negation[item]
    
#     dict_list = []
#     dict_list.append(model1)
#     dict_list.append(model2)
#     dict_list.append(model3)
#     dict_list.append(model4)
#     dict_list.append(model5)
#     # dict_list.append(model6)
    
#     a = {}
#     for i in dict_list:
#         if dict_list.count(i)>1:
#             a[i] = dict_list.count(i)
    
#     L = sorted(a.items(),key=lambda item:item[1],reverse=True)
#     L=L[:2]
#     if len(L) > 1:
#         if L[0][1] == L[1][1]:
#             ensemble_list.append(model3)        
#         elif L[0][1] > L[1][1]:
#             ensemble_list.append(L[0][0])        
#         elif L[0][1] < L[1][1]:
#             ensemble_list.append(L[1][1])
#     else:
#         maxlabel = max(dict_list,key=dict_list.count)
#         ensemble_list.append(maxlabel)

# final_numpy = np.array(ensemble_list)
# np.save("/data/qbao775/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/BERT/Checkpoints/reclor/ensemble_contraposition_1_2_1_3_our_model_v5_rank2/merged_test_predict_data.npy", final_numpy)
