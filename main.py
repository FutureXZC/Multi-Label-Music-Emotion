# -*- coding: utf-8 -*-
from musicLoad import Music
from GUI import Console

if __name__ == '__main__':
    m = Music()
    m.figureSelect()
    m.figureMapping()
    c = Console(m)
    # # 查看emotions数据集的特征和标记信息
    # from skmultilearn.dataset import load_dataset
    # X_train,  Y_train, feature_names, label_names = load_dataset('emotions', 'train')
    # feature_names = [item[0] for item in feature_names]
    # label_names = [item[0] for item in label_names]
    # data = {
    #     'mean': feature_names[:16],
    #     'mean_std': feature_names[16:32],
    #     'std': feature_names[32:48],
    #     'std_std': feature_names[48:64],
    #     'rhythmic': feature_names[64:] + [' '] * 8
    # }
    # frame = pd.DataFrame(data)
    # print(frame)
    # print(label_names)   
    