import fiftyone
import pandas as pd
import os
import argparse
import shutil

def preprocessing(regenerate_train_images,
                  max_samples=1500,
                  download_classes=["Pen", "Computer mouse", "Computer keyboard", "Computer monitor", "Tablet computer"]):
    if regenerate_train_images=='True':
        data_list = ['validation', 'train', 'test']
        for data_type in data_list:
            source_dir = './openimage_downloads/' + data_type + '/data'
            target_img_dir = './custom_img/images/' + data_type

            check_if_image_exist = os.path.isdir(source_dir)
            if not check_if_image_exist:
                print("Data not found, downloading data")
                fiftyone.zoo.download_zoo_dataset("open-images-v6",
                                                  splits=['train', 'test', 'validation'],
                                                  label_types=["detections"],
                                                  classes=download_classes,
                                                  max_samples=max_samples, seed=2022, overwrite=True,
                                                  dataset_dir='./openimage_downloads')

            file_names = os.listdir(source_dir)
            os.makedirs(target_img_dir, exist_ok=True)

            for file_name in file_names:
                shutil.copy(os.path.join(source_dir, file_name), target_img_dir)

    data_list = ['validation', 'train', 'test']
    mapper = {'/m/0k1tl': '0', '/m/020lf': '1', '/m/01m2v': '2', '/m/02522': '3', '/m/0bh9flk': '4'}
    for data_type in data_list:
        print('reading data')
        img_path = './custom_img/images/' + data_type
        lab_path = './openimage_downloads/' + data_type + '/labels'
        write_path = './custom_img/labels/' + data_type
        os.makedirs(write_path, exist_ok=True)
        archive = './archive/' + data_type
        filenames = [os.path.splitext(filename)[0] for filename in os.listdir(img_path)]
        detection = pd.read_csv(lab_path + '/detections.csv')
        detection['LabelName'] = detection['LabelName'].replace(mapper)
        detection = detection.loc[detection.LabelName.isin(['0', '1', '2', '3', '4'])]

        for filename in filenames:
            print('creating variables')
            df_temp = detection.loc[detection.ImageID == filename]
            df_temp['x_width'] = df_temp.XMax - df_temp.XMin
            df_temp['y_height'] = df_temp.YMax - df_temp.YMin
            df_temp['x_center'] = (df_temp.XMax + df_temp.XMin) / 2
            df_temp['y_center'] = (df_temp.YMax + df_temp.YMin) / 2

            df_temp = df_temp[['LabelName', 'x_center', 'y_center', 'x_width', 'y_height']]
            # print('writing to ' + filename)
            #         np.savetxt(write_path + '/' + filename + ".txt", df_temp.values, fmt='%d')
            #         asd
            df_temp.to_csv(write_path + '/' + filename + ".txt", index=False, header=False, sep=' ')

        print('writing .txt')
        path = './custom_img/'
        filenames = os.listdir(path + '/images/' + data_type)
        with open(path + '/' + data_type + '.txt', 'w') as file:
            for filename in filenames:
                line = './images/' + data_type + '/' + filename
                file.write(f"{line}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate_train_images', type=str, default=False, help='initial weights path')
    parser.add_argument('--classes', type=str, default='', help='model.yaml path')
    parser.add_argument('--max_samples', type=str, default=1500, help='data.yaml path')

    opt = parser.parse_args()
    if opt.classes == '':
        opt.classes = ["Pen", "Computer mouse", "Computer keyboard", "Computer monitor", "Tablet computer"]
    preprocessing(opt.regenerate_train_images, max_samples=opt.max_samples, download_classes=opt.classes)
