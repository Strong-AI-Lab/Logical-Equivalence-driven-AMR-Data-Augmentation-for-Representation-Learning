# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

import csv
import glob
import json
import logging
import os
from typing import List
import warnings
import json_lines

import tqdm

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
DEPRECATION_WARNING = (
    "This {0} will be removed from the library soon, preprocessing should be handled with the ?? Datasets "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py"
)

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


class ReclorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test_logical_equivalence_example.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label']  # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id=id_string,
                    question=question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label=label
                )
            )
        return examples

class LogiQAProcessor(DataProcessor):
    """Processor for the LogiQA data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "Train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "Eval.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "Test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label']  # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id=id_string,
                    question=question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label=label
                )
            )
        return examples


class reclor_MRC_paraphrasedstage_1_Processor(DataProcessor):
    """Processor for the ReClor and paraphrased reclor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train_paraphrased_MRC.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label']  # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id=id_string,
                    question=question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label=label
                )
            )
        return examples

class PararuleProcessor(DataProcessor):
    """Processor for the Pararule and Pararule Plus data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train_task1.jsonl")), "train")
        return self._create_examples(os.path.join(data_dir, "train_task1.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "val_task1.jsonl")), "dev")
        return self._create_examples(os.path.join(data_dir, "val_task1.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test_depth_1.jsonl")), "test")
        return self._create_examples(os.path.join(data_dir, "test_depth_1.jsonl"), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, input_file, type):
        """Creates examples for the training and dev sets."""
        examples = []
        # for d in lines:
        with open(input_file) as f:
            for d in json_lines.reader(f):
                context = d['context']
                for q in d['questions']:
                    question = q['text']
                    # answers = d['answers']
                    # label = q['label'] # for test set, there is no label. Just use 0 for convenience.
                    if q['label'] == True:
                        label = 1
                    elif q['label'] == False:
                        label = 0
                    id_string = q['id']
                    examples.append(
                        InputExample(
                            example_id=id_string,
                            question=question,
                            contexts=[context, context],  # this is not efficient but convenient
                            # endings=[answers[0], answers[1], answers[2], answers[3]],
                            endings=["false", "true"],
                            label=label
                        )
                    )
        return examples

class PararuleProcessorDepth1(DataProcessor):
    """Processor for the Pararule and Pararule Plus data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train_task1.jsonl")), "train")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-1-train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "val_task1.jsonl")), "dev")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-1-dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test_depth_1.jsonl")), "test")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-1-dev.jsonl"), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, input_file, type):
        """Creates examples for the training and dev sets."""
        examples = []
        # for d in lines:
        with open(input_file) as f:
            for d in json_lines.reader(f):
                context = d['context']
                for q in d['questions']:
                    question = q['text']
                    # answers = d['answers']
                    # label = q['label'] # for test set, there is no label. Just use 0 for convenience.
                    if q['label'] == True or q['label'] == "true":
                        label = 1
                    elif q['label'] == False or q['label'] == "false":
                        label = 0
                    id_string = q['id']
                    examples.append(
                        InputExample(
                            example_id=id_string,
                            question=question,
                            contexts=[context, context],  # this is not efficient but convenient
                            # endings=[answers[0], answers[1], answers[2], answers[3]],
                            endings=["false", "true"],
                            label=label
                        )
                    )
        return examples

class PararuleProcessorDepth1Reparaphrased(DataProcessor):
    """Processor for the Pararule and Pararule Plus data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train_task1.jsonl")), "train")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-1-train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "val_task1.jsonl")), "dev")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-logical_equivalence_reparaphrased_depth-1-dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test_depth_1.jsonl")), "test")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-logical_equivalence_reparaphrased_depth-1-dev.jsonl"), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, input_file, type):
        """Creates examples for the training and dev sets."""
        examples = []
        # for d in lines:
        with open(input_file) as f:
            for d in json_lines.reader(f):
                context = d['context']
                for q in d['questions']:
                    question = q['text']
                    # answers = d['answers']
                    # label = q['label'] # for test set, there is no label. Just use 0 for convenience.
                    if q['label'] == True or q['label'] == "true":
                        label = 1
                    elif q['label'] == False or q['label'] == "false":
                        label = 0
                    id_string = q['id']
                    examples.append(
                        InputExample(
                            example_id=id_string,
                            question=question,
                            contexts=[context, context],  # this is not efficient but convenient
                            # endings=[answers[0], answers[1], answers[2], answers[3]],
                            endings=["false", "true"],
                            label=label
                        )
                    )
        return examples

class PararuleProcessorDepth2(DataProcessor):
    """Processor for the Pararule and Pararule Plus data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train_task1.jsonl")), "train")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-1-train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "val_task1.jsonl")), "dev")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-2-dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test_depth_1.jsonl")), "test")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-2-dev.jsonl"), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, input_file, type):
        """Creates examples for the training and dev sets."""
        examples = []
        # for d in lines:
        with open(input_file) as f:
            for d in json_lines.reader(f):
                context = d['context']
                for q in d['questions']:
                    question = q['text']
                    # answers = d['answers']
                    # label = q['label'] # for test set, there is no label. Just use 0 for convenience.
                    if q['label'] == True or q['label'] == "true":
                        label = 1
                    elif q['label'] == False or q['label'] == "false":
                        label = 0
                    id_string = q['id']
                    examples.append(
                        InputExample(
                            example_id=id_string,
                            question=question,
                            contexts=[context, context],  # this is not efficient but convenient
                            # endings=[answers[0], answers[1], answers[2], answers[3]],
                            endings=["false", "true"],
                            label=label
                        )
                    )
        return examples

class PararuleProcessorDepth2Reparaphrased(DataProcessor):
    """Processor for the Pararule and Pararule Plus data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train_task1.jsonl")), "train")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-depth-1-train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "val_task1.jsonl")), "dev")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-logical_equivalence_reparaphrased_depth-2-dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test_depth_1.jsonl")), "test")
        return self._create_examples(os.path.join(data_dir, "NonNegationRule-logical_equivalence_reparaphrased_depth-2-dev.jsonl"), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, input_file, type):
        """Creates examples for the training and dev sets."""
        examples = []
        # for d in lines:
        with open(input_file) as f:
            for d in json_lines.reader(f):
                context = d['context']
                for q in d['questions']:
                    question = q['text']
                    # answers = d['answers']
                    # label = q['label'] # for test set, there is no label. Just use 0 for convenience.
                    if q['label'] == True or q['label'] == "true":
                        label = 1
                    elif q['label'] == False or q['label'] == "false":
                        label = 0
                    id_string = q['id']
                    examples.append(
                        InputExample(
                            example_id=id_string,
                            question=question,
                            contexts=[context, context],  # this is not efficient but convenient
                            # endings=[answers[0], answers[1], answers[2], answers[3]],
                            endings=["false", "true"],
                            label=label
                        )
                    )
        return examples

class reclor_binary_paraphrasedstage_1_Processor(DataProcessor):
    """reclor binary classification Processor for the reclor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            id_string = d['id_string']
            label = 0 if type == "test" else d['label']
            for i in range(len(answers)):
                if label == i:
                    new_label = 1
                else:
                    new_label = 0
                examples.append(
                    InputExample(
                        example_id=id_string,
                        question=question,
                        contexts=[context],  # this is not efficient but convenient
                        # endings=[answers[0], answers[1], answers[2], answers[3]],
                        endings=[answers[i]],
                        label=new_label
                    )
                )
        return examples

def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, attention_mask = inputs["input_ids"], inputs['attention_mask']

            token_type_ids = inputs["token_type_ids"]

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label, ))

    return features


processors = {"race": RaceProcessor, "swag": SwagProcessor, "arc": ArcProcessor, "reclor": ReclorProcessor,
              "pararule": PararuleProcessor, "reclor_mrc_paraphrasedstage_1": reclor_MRC_paraphrasedstage_1_Processor,
              "reclor_binary_paraphrasedstage_1": reclor_binary_paraphrasedstage_1_Processor, "logiqa": LogiQAProcessor, "pararule_depth_1": PararuleProcessorDepth1,
              "pararule_depth_1_reparaphrased": PararuleProcessorDepth1Reparaphrased, "pararule_depth_2": PararuleProcessorDepth2,
              "pararule_depth_2_reparaphrased": PararuleProcessorDepth2Reparaphrased}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race": 4, "swag": 4, "arc": 4, "reclor": 4, "pararule": 2, "reclor_mrc_paraphrasedstage_1":4,
                                    "reclor_binary_paraphrasedstage_1": 2, "logiqa": 4, "pararule_depth_1": 2, "pararule_depth_1_reparaphrased": 2, "pararule_depth_2": 2, "pararule_depth_2_reparaphrased": 2}
