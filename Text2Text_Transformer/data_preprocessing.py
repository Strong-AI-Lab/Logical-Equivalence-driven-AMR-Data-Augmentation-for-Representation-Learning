from pathlib import Path
import csv
import pandas as pd
import random


def read_file(file_path):
    output_file = file_path.rsplit('.txt', 1)[0] + '.csv'
    # pd.read_csv(file_path, sep='\n\n')
    with open(file_path) as f:
        data = f.readlines()
    samples = []
    sample = []
    pattern = "\\w"
    for loop in data:
        loop = loop.strip()
        if not any(c.isalpha() for c in loop):
            if sample:
                if len(sample[0].split()) > 30:
                    sample = []
                    continue
                elif not any(r.isascii() for r in sample[0]):
                    sample = []
                    continue
                elif sum(not c.isalnum() for c in ''.join(sample[0].split())) > 30:
                    sample = []
                    continue
                elif len(sample) > 1:
                    for label in sample[1:]:
                        label_temp = label.split(' ', 1)[1]
                        if label_temp.startswith('Context'):
                            label_temp = '(' + label_temp.rsplit('):(', 1)[1]
                        sample[sample.index(label)] = label_temp
                    # samples.append(','.join(sample))
                    samples.append(sample)
                    sample = []
                else:
                    sample = []
                    continue
            else:
                continue
        else:
            sample.append(loop)

    for sample in samples:
        if len(sample) < 2:
            samples.remove(sample)
    with open(output_file, "a") as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    return output_file


def split_data(file_path):
    dataset_path = file_path.rsplit('/', 1)[0] + '/dataset_'
    train_data = dataset_path + '/train.csv'
    dev_data = dataset_path + '/dev.csv'
    test_data = dataset_path + '/test.csv'
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(file_path, header=None, names=range(2)).values.tolist()
    random.shuffle(samples)
    length = len(samples)
    length_tr = int(length*0.9)
    length_dev = length_tr + int(length*0.05)

    f = open(train_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[:length_tr])
    f = open(dev_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_tr:length_dev])
    f = open(test_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_dev:])


def add_indicator(file_path):
    dataset_path = file_path.rsplit('/', 1)[0] + '/dataset_indicator_'
    train_data = dataset_path + '/train.csv'
    dev_data = dataset_path + '/dev.csv'
    test_data = dataset_path + '/test.csv'
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(file_path, header=None, names=range(2)).values.tolist()
    for sample_id in range(len(samples)):
        samples[sample_id][0] = 'extract triples from sentence: ' + samples[sample_id][0].strip()
    random.shuffle(samples)
    length = len(samples)
    length_tr = int(length*0.9)
    length_dev = length_tr + int(length*0.05)

    f = open(train_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[:length_tr])
    f = open(dev_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_tr:length_dev])
    f = open(test_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_dev:])


def clean_and_add_indicator(file_path):
    dataset_path = file_path.rsplit('/', 1)[0] + '/dataset_single_triples_cleaned'
    train_data = dataset_path + '/train.csv'
    dev_data = dataset_path + '/dev.csv'
    test_data = dataset_path + '/test.csv'
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(file_path, header=None, names=range(2)).values.tolist()
    for sample_id in range(len(samples)):
        samples[sample_id][0] = samples[sample_id][0].strip()
    sample_cl = []
    for sample_id in range(len(samples)):
        flag = True
        for item in samples[sample_id][1:]:
            item = item.replace('(', "").replace(')', "")
            if any(a.strip() == "" for a in item.split(';')):
                flag = False
                break
            elif any(a.strip()[0].isdigit() for a in item.split(';')):
                flag = False
                break
            elif any('L:' in a.strip() for a in item.split(';')):
                flag = False
                break
            elif any('T:' in a.strip() for a in item.split(';')):
                flag = False
                break
        if flag:
            sample_cl.append(samples[sample_id])

    samples = sample_cl
    random.shuffle(samples)
    length = len(samples)
    length_tr = int(length*0.9)
    length_dev = length_tr + int(length*0.05)

    f = open(train_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[:length_tr])
    f = open(dev_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_tr:length_dev])
    f = open(test_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_dev:])


def multiple_labels(file_path):
    dataset_path = file_path.rsplit('/', 1)[0] + '/dataset_multiple_'
    train_data = dataset_path + '/train.csv'
    dev_data = dataset_path + '/dev.csv'
    test_data = dataset_path + '/test.csv'
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(file_path, header=None, names=range(3)).values.tolist()
    for sample_id in range(len(samples)):
        samples[sample_id][0] = 'extract triples from sentence: ' + samples[sample_id][0].strip()
    sample_cl = []
    for sample_id in range(len(samples)):
        flag = True
        label = []
        for item in samples[sample_id][1:]:
            if isinstance(item, str):
                item_c = item.replace('(', "").replace(')', "")
                if any(a.strip() == "" for a in item_c.split(';')):
                    flag = False
                    break
                elif any(a.strip()[0].isdigit() for a in item_c.split(';')):
                    flag = False
                    break
                elif any('L:' in a.strip() for a in item_c.split(';')):
                    flag = False
                    break
                elif any('T:' in a.strip() for a in item_c.split(';')):
                    flag = False
                    break
                label.append(item)
        if flag:
            sample_cl.append([samples[sample_id][0], " | ".join(label)])

    samples = sample_cl
    random.shuffle(samples)
    length = len(samples)
    length_tr = int(length*0.9)
    length_dev = length_tr + int(length*0.05)

    f = open(train_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[:length_tr])
    f = open(dev_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_tr:length_dev])
    f = open(test_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_dev:])


def multiple_labels_without_prefix(file_path):
    dataset_path = file_path.rsplit('/', 1)[0] + '/dataset_multiple_without_prefix'
    train_data = dataset_path + '/train.csv'
    dev_data = dataset_path + '/dev.csv'
    test_data = dataset_path + '/test.csv'
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(file_path, header=None, names=range(3)).values.tolist()
    for sample_id in range(len(samples)):
        samples[sample_id][0] = samples[sample_id][0].strip()
    sample_cl = []
    for sample_id in range(len(samples)):
        flag = True
        label = []
        for item in samples[sample_id][1:]:
            if isinstance(item, str):
                item_c = item.replace('(', "").replace(')', "")
                if any(a.strip() == "" for a in item_c.split(';')):
                    flag = False
                    break
                elif any(a.strip()[0].isdigit() for a in item_c.split(';')):
                    flag = False
                    break
                elif any('L:' in a.strip() for a in item_c.split(';')):
                    flag = False
                    break
                elif any('T:' in a.strip() for a in item_c.split(';')):
                    flag = False
                    break
                label.append(item)
        if flag:
            sample_cl.append([samples[sample_id][0], " | ".join(label)])

    samples = sample_cl
    random.shuffle(samples)
    length = len(samples)
    length_tr = int(length*0.9)
    length_dev = length_tr + int(length*0.05)

    f = open(train_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[:length_tr])
    f = open(dev_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_tr:length_dev])
    f = open(test_data, 'w')
    writer = csv.writer(f)
    writer.writerows(samples[length_dev:])


def create_dataset(file_path):
    # raw_file = read_file(file_path)
    clean_and_add_indicator("data/TupleInfKB/4thGradeOpenIE.csv")


if __name__ == '__main__':
    create_dataset('4thGradeOpenIE.txt')
