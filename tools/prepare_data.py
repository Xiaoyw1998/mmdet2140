import os
import xml.etree.ElementTree as ET
import numpy as np
import json
import pickle

# CLASSES = {'clothes': 0, 'no_clothes': 1, 'person_clothes': 2, 'person_no_clothes': 3}


def create_data_ann(data_root):
    print('prepare data...')
    files = os.listdir(data_root)
    # jpg_files = [f for f in files if f.endswith('jpg')]
    xml_files = [os.path.join(data_root, f) for f in files if f.endswith('xml')]

    num_total = len(xml_files)
    num_val = 5
    val_index = np.random.choice(num_total, num_val, replace=False).tolist()
    train_index = []
    for i in range(num_total):
        if i in val_index:
            continue
        train_index.append(i)

    val_xmls = [xml_files[i] for i in val_index]
    train_xmls = [xml_files[i] for i in train_index]
    val_data_info, val_files = xml2ann(val_xmls)
    train_data_info, train_files = xml2ann(train_xmls)
    if not os.path.exists('data'):
        os.makedirs('data')
    with open("data/val.pkl", "wb") as f:
        pickle.dump(val_data_info, f)
    with open("data/train.pkl", "wb") as f:
        pickle.dump(train_data_info, f)
    # with open("data/val.json", "w") as f:
    #     json.dump(val_data_info, f)
    # with open("data/train.json", "w") as f:
    #     json.dump(train_data_info, f)
    with open("data/val.txt", "w") as f:
        for file in val_files:
            f.writelines(file+'\n')
    with open("data/train.txt", "w") as f:
        for file in train_files:
            f.writelines(file+'\n')


def xml2ann(xmls):
    data_infos = []
    filenames = []
    for xml in xmls:
        tree = ET.ElementTree(file=xml)
        root = tree.getroot()
        filename = root.find('filename').text
        filenames.append(filename)
        assert xml.split('/')[-1][:-4] == filename[:-4]
        size = list(root)[1]
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        bboxes = []
        labels = []
        for node in root.iter('object'):
            child = list(node)
            classname = child[0].text
            if classname == 'no_clothes':
                continue
            elif classname == 'clothes':
                labels.append(0)
            elif classname.startswith('person'):
                labels.append(1)
            else:
                print(classname)
                raise RuntimeError('unknown class')
            bbox = []
            for atr in child[1]:
                bbox.append(int(atr.text))
            bboxes.append(bbox)

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        ann = {'bboxes': bboxes, 'labels': labels}
        img_info = {'filename': filename, 'width': width, 'height': height, 'ann': ann}
        data_infos.append(img_info)

    return data_infos, filenames


def test():
    with open("data/val.pkl", "rb") as f:
        val = pickle.load(f)
    with open("data/train.pkl", "rb") as f:
        train = pickle.load(f)
    print(val)
    print(train)


if __name__ == '__main__':
    create_data_ann('/media/HD2T/cpmpetition/fanguangyi/data')
    # test()
