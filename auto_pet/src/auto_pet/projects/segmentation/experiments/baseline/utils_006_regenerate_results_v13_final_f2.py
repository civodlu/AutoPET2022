if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy('file_system')

    import os
    import traceback
    from argparse import Namespace
    from functools import partial

    from trw.transforms import TransformRandomFlip, criteria_is_array_n_or_above, TransformSqueeze, TransformUnsqueeze
    from auto_pet import create_datasets_reservoir_map, transform_feature_3d_v2, default, \
        configure_startup, read_splits, transform_augmentation_random_affine_transform, transform_surface_loss_preprocessing, \
        TransformCleanFeatures, transform_export_maximum_intensity_projection
    from auto_pet.projects.segmentation import run_trainer
    from auto_pet.projects.segmentation import load_case_hdf5, load_case_hdf5_random128_m3200
    from auto_pet.projects.segmentation import PreprocessDataV4_lung_soft_tissues_hot as Preprocessing
    from auto_pet.projects.segmentation.model_unet_multiclass_deepsupervision_configured_v1 import SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_sensitivity as Model
    from auto_pet.projects.segmentation.config.data import root_datasets, root_logging
    from auto_pet.projects.segmentation import TransformGeneric, train_suv_augmentations_v2, train_ct_augmentations, train_suv_augmentations
    from auto_pet.projects.segmentation.transform_change_lesion_contrast import TransformChangeLesionContrast
    import trw
    import numpy as np


    #
    # Configuration: this should be source controlled so that a result
    # can be associated EXACTLY with a training setup. Only 
    # non-hyperparameters dynamically configured by environment variable
    # as those will not affect the training results. 
    #
    configuration = {
        'datasets': {
            'data_root': default('data_root', root_datasets),
            'datasets_version': 'v1_3',
            'splits_name': default('splits_name', 'autopet_stratified_v1_kfold2.json'),
        },
        'training': {
            'device': default('device', 'cuda:1'),
            'batch_size': 8,
            'data_loading_workers': default('data_loading_workers', 2, output_type=int),
            'map_workers': default('map_workers', 3, output_type=int),
            'nb_epochs': 0,
            'sheduler_nb_steps': 6,
            'learning_rate': 0,
            'weight_decay': 0,
            'eval_every_X_epoch': default('eval_every_X_epoch', 100, output_type=int),
            'eval_inference_every_X_epoch': default('eval_inference_every_X_epoch', 200, output_type=int),
            'run_name': default('run_name', os.path.splitext(os.path.basename(__file__))[0]),
            'logging_directory': default('logging_directory', root_logging),
            'vscode_batch_size_reduction_factor': 1,
            'gradient_update_frequency': 1,
            '3d_based_model': True,
            'mixed_precision_enabled': True,

            'test_time_augmentation_axis': True,
            # only useful for the refiner approach 
            'inference_discard_train': True,
            'inference_export_3d_volumes': True,
            'export_inference_prob': True,
        },
        'data': {
            'fov_half_size': np.asarray((64, 48, 48)),
            'samples_per_patient': 1,
            'preprocessing': Preprocessing(),
            'load_case_train': load_case_hdf5_random128_m3200,
            'load_case_valid': load_case_hdf5,
            'config_start': '/mnt/datasets/ludovic/AutoPET/logging2/094_v13_strat_f2_sensitivity/last.model'
        },
        'tracking': {
            'derived_from': '094_v13_strat_f2_sensitivity.py',
            'info': 'Regenerated results'
        }
    }
    configuration = Namespace(**configuration)


    def main():
        configure_startup(configuration)

        # with chunking, we SHOULD have z-axis always valid, else
        # we will sample too many batches with missing z slices!
        features_fn_train = partial(transform_feature_3d_v2, configuration=configuration, sample_volume_name='ct', only_valid_z=True, nb_samples=configuration.data['samples_per_patient'])
        # sample independently for the valid: we want a reliable estimate, so using more samples!
        features_fn_valid = partial(transform_feature_3d_v2, configuration=configuration, sample_volume_name='ct', only_valid_z=True, nb_samples=4)

        config_start = configuration.data.get('config_start')
        datasets = create_datasets_reservoir_map(
            read_splits(configuration), 
            configuration=configuration,
            load_case_fn=None,
            load_case_train_fn=configuration.data['load_case_train'],
            load_case_valid_fn=configuration.data['load_case_valid'], 
            preprocess_data_train_fn=[
                configuration.data['preprocessing'],
                TransformCleanFeatures(['bounding_boxes_min_max']) # incompatible type (#BB different for each case!)
            ],
            preprocess_data_test_fn=[
                configuration.data['preprocessing'],
                TransformCleanFeatures(['bounding_boxes_min_max']) # incompatible type (#BB different for each case!)
            ],
            transform_train=[
                partial(trw.train.default_collate_fn, device=None),
                # all slices must belong to the same volume, create a virtual axis 0
                features_fn_train,
                #trw.transforms.TransformMoveToDevice(torch.device(configuration.training['device']))
                ],
            transform_test=[
                #flatten_lists,
                features_fn_valid,
                #trw.transforms.TransformMoveToDevice(torch.device(configuration.training['device']))
            ],
            max_reservoir_samples=1,  # nb_samples % (samples_per_patient * batch_size) should be 0 for optimality
            min_reservoir_samples=1,
            nb_map_workers=configuration.training['map_workers'],
            nb_reservoir_workers=configuration.training['data_loading_workers'],
            max_reservoir_jobs_at_once_factor=40
        )

        try:        
            optimizer = None

            model = Model()
            if config_start is not None:
                trw.train.TrainerV2.load_state(model, config_start, device=torch.device(configuration.training['device']), strict=True)
            run_trainer(configuration, datasets, model, optimizer)
        except Exception as e:
            print(f'Exception caught={e}')
            print('-------------- Stacktrace --------------')
            traceback.print_exc()
            print('----------------------------------------')
        
        del datasets
        print('Datasets deleted!')

    main()
    print('ALL DONE!!!!')

    # just in case processes might be deadlocked
    # let's make sure we are not blocking a GPU!
    killer = trw.utils.graceful_killer.GracefulKiller()
    killer.exit_gracefully(None, None)