import os
import pandas as pd
import shutil

df_train=pd.read_csv('train_dataset_crops_single_animal_template_captions_T1T7_ID.csv')
df_val=pd.read_csv('val_dataset_crops_single_animal_template_captions_T1T7_ID.csv')
df_test=pd.read_csv('test_dataset_crops_single_animal_template_captions_T1T8T10.csv')

info_folder={'train':df_train,'val':df_val,'test':df_test}
path='snaphsot_serengeti_cropped_single_animals'
for split_folder,info in info_folder.items():
  print(split_folder)
  for _,row in info.iterrows():
    species_id=str(row['species_id'])
    crop_path=row['crop_path']
    crop_path=os.path.join(path,crop_path)
    if os.path.exists(crop_path):
      img_name=crop_path.split('/')[-1]
      move_path=os.path.join('img',split_folder,species_id)
      if not os.path.exists(move_path):
        os.makedirs(move_path)
      move_path=os.path.join('img',split_folder,species_id,img_name)
      shutil.copy(crop_path,move_path)
