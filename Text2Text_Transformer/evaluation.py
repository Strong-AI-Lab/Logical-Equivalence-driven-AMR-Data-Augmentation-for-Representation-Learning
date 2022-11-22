import Data2TextScorer as scorer
import pandas as pd
import os
import ast
from config import opt

def batch_generate(input_path: str, output_path: str, logging_steps: int):
    print("The evaluation result saved path is: ", output_path)
    samples, labels = load_examples_and_label(input_path)
    predictions = load_predictions(input_path)
    begin_tag = 0
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if df.shape[0] > 0:
            begin_tag = df.shape[0]
    for i in range(begin_tag, len(samples), logging_steps):
        results = []
        score_result = scorer.batch_score(samples[i:i+logging_steps], labels[i:i+logging_steps], predictions[i:i+logging_steps])
        results.extend(score_result)
        write_into_csv(output_path, results)


def load_examples_and_label(test_data_path):
    src_text = []
    label_text = []
    data_samples = pd.read_csv(test_data_path).values.tolist()
    for sample in data_samples:
        src_text.append(sample[0])
        label_text.append(sample[1])
        # labels = []
        # sample_list = ast.literal_eval(sample[1])
        # # sample_list = [sample[1]]
        # for label in sample_list:
        #     labels.append(''.join(label))
        # label_text.append(labels)

    return src_text, label_text

def load_predictions(pred_path):
    data = pd.read_csv(pred_path).values.tolist()
    predictions = []
    for row in data:
        # Skip input and label columns
        prediction = row[2:]
        prediction_list = []
        prediction_list_2 = []
        for i in prediction:
            prediction_list.append(str(i))
        for i in prediction_list:
            if i == 'nan':
                prediction_list_2.append("")
            else:
                prediction_list_2.append(i)
        predictions.append(prediction_list_2)
    return predictions


def write_into_csv(output_path, results):
    df = pd.DataFrame(results)
    append = False
    if os.path.isfile(output_path):
        append = True
    df.to_csv(output_path, index=False, mode='a' if append else 'w', header=not append)


if __name__ == '__main__':
    # batch_generate(
    #     input_path=opt.output_path,
    #     output_path=opt.evaluate_output_path,
    #     logging_steps=opt.logging_steps_evaluate
    # )
    batch_generate(
        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/model_gpt2_fc_layer_without_prefix/checkpoint-8414/train_result.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/model_gpt2_fc_layer_without_prefix/checkpoint-8414/evaluate_train_result_updated_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_bart_base_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_bart_base_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_NLPAUG_5_updated.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_NLPAUG_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_ssmba_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_ssmba_5_latest_full.csv",

        input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/model_gpt2_prefix/checkpoint-16828/train_result.csv",
        output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/model_gpt2_prefix/checkpoint-16828/evaluate_train_prefix_result_updated_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/model_gpt2_fc_layer_without_prefix/checkpoint-24040/train_result.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/model_gpt2_fc_layer_without_prefix/checkpoint-24040/evaluate_train_result_updated_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_BART_base_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_bart_base_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_NLPAUG_5_updated.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_NLPAUG_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_ssmba_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_ssmba_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/model_gpt2_prefix/checkpoint-27045/train_result.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/model_gpt2_prefix/checkpoint-27045/evaluate_train_prefix_result_updated_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/model_gpt2_fc_layer_without_prefix/checkpoint-83552/train_result.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/model_gpt2_fc_layer_without_prefix/checkpoint-83552/evaluate_train_result_updated_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/model_gpt2_prefix/checkpoint-104440/train_result.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/model_gpt2_prefix/checkpoint-104440/evaluate_train_prefix_result_updated_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_bart_base_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_bart_base_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_NLPAUG_5_updated.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_NLPAUG_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_ssmba_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_ssmba_5_latest_full.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_t5_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_t5_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_t5_small_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_t5_small_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_BART_base_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_bart_base_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_t5_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_t5_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_t5_small_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_t5_small_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_bart_base_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_bart_base_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_t5_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_t5_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_t5_small_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_t5_small_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_NLPAUG_5_updated.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_NLPAUG_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_bart_base_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_bart_base_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_NLPAUG_5_updated.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_NLPAUG_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_NLPAUG_5_updated.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_NLPAUG_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_bart_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_bart_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_bart_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_bart_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_ssmba_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/evaluation_train_paraphrased_ssmba_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_ssmba_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/evaluation_train_paraphrased_ssmba_5.csv",

        # input_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_ssmba_5.csv",
        # output_path="/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/evaluation_train_paraphrased_ssmba_5.csv",
        logging_steps=opt.logging_steps_evaluate
    )
