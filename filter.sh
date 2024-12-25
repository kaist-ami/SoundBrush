#python dataset_creation/filtering_synthetic.py --seed_path /local_data/sung/Instructpix2pix/dataset/logs/0803_env_final/metadata.json
#python dataset_creation/filtering_algorithms_ssl.py --seed_path /local_data/sung/Instructpix2pix/dataset/ssl_synthetic/seeds.json
#python dataset_creation/filtering_algorithms.py --seed_path /local_data/sung/Instructpix2pix/dataset/0730_env_images/filtered.json --img_dir /local_data/sung/Instructpix2pix/dataset/0730_env_images/images --aud_dir /local_data/sung/Instructpix2pix/dataset/0730_sanity_env

#python dataset_creation/filtering_algorithms_final.py --seed_path /local_data/sung/Instructpix2pix/dataset/logs/0803_env_final/filtered.json --img_dir /local_data/sung/Instructpix2pix/dataset/logs/0803_env_final/images --aud_dir /local_data/sung/Instructpix2pix/dataset/logs/0730_sanity_env --test_aud_dir /local_data/sung/Instructpix2pix/dataset/logs/0730_sanity_env_test



#python dataset_creation/quantitative_eval.py --seed_path /local_data/sung/Instructpix2pix/dataset/0811_js/output.json --img_dir /local_data/sung/Instructpix2pix/dataset/0811_js/imgs --aud_dir /local_data/sung/Instructpix2pix/dataset/0811_js/wavs
python dataset_creation/quantitative_eval.py --seed_path /local_data/sung/Instructpix2pix/logs/train_sb_0809_all_mapping_peft_V1_7/generated/quantitative/seeds.json --img_dir /local_data/sung/Instructpix2pix/dataset/0811_js/imgs --aud_dir /local_data/sung/Instructpix2pix/dataset/0811_js/wavs