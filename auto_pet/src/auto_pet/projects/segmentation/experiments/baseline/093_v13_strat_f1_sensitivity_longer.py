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
        TransformCleanFeatures, transform_export_maximum_intensity_projection, create_datasets_reservoir_map_weighted, transform_feature_3d_resampled
    from auto_pet.projects.segmentation import run_trainer, load_model
    from auto_pet.projects.segmentation import load_case_hdf5, load_case_hdf5_random128_m3200
    from auto_pet.projects.segmentation import PreprocessDataV4_lung_soft_tissues_hot as Preprocessing
    from auto_pet.projects.segmentation.model_unet_multiclass_deepsupervision_configured_v1 import SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_sensitivity as Model
    from auto_pet.projects.segmentation.config.data import root_datasets, root_logging
    from auto_pet.projects.segmentation import TransformGeneric, train_suv_augmentations_v2, train_ct_augmentations, train_suv_augmentations
    from auto_pet.projects.segmentation.transform_change_lesion_contrast import TransformChangeLesionContrast
    from auto_pet.projects.segmentation.transform_label_seg import transform_label_segmentation
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
            'splits_name': default('splits_name', 'autopet_stratified_v1_kfold1.json'),
        },
        'training': {
            'device': default('device', 'cuda:6'),
            'batch_size': 10,
            'data_loading_workers': default('data_loading_workers', 2, output_type=int),
            'map_workers': default('map_workers', 3, output_type=int),
            'nb_epochs': default('nb_epochs', 12000, output_type=int),
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'eval_every_X_epoch': default('eval_every_X_epoch', 200, output_type=int),
            'eval_inference_every_X_epoch': default('eval_inference_every_X_epoch', 200, output_type=int),
            'run_name': default('run_name', os.path.splitext(os.path.basename(__file__))[0]),
            'logging_directory': default('logging_directory', root_logging),
            'vscode_batch_size_reduction_factor': 1,
            'gradient_update_frequency': 1,
            '3d_based_model': True,
            'mixed_precision_enabled': True,
        },
        'data': {
            'fov_half_size': np.asarray((64, 48, 48)),
            'samples_per_patient': 1,
            'preprocessing': Preprocessing(),
            'load_case_train': load_case_hdf5_random128_m3200,
            'load_case_valid': load_case_hdf5,
            'config_start': '/mnt/datasets/ludovic/AutoPET/logging2/093_v13_strat_f1_sensitivity/last.model',
        },
        'tracking': {
            'derived_from': '069_v13_fold2_hdf5_preprocessing_v4_ce_longer2.py',
            'info': 'Short FoV to encourage sensitivity, to be used with a post-processing to remove FP. Trained ONLY with foreground cases. Extra sensitivity loss'
        }
    }
    configuration = Namespace(**configuration)


    def main():
        configure_startup(configuration)

        config_start = configuration.data.get('config_start')

        criteria_fn = partial(criteria_is_array_n_or_above, dim=3)

                # with chunking, we SHOULD have z-axis always valid, else
        # we will sample too many batches with missing z slices!
        features_fn_train = partial(transform_feature_3d_v2, configuration=configuration, sample_volume_name='ct', only_valid_z=True, nb_samples=configuration.data['samples_per_patient'])
        # sample independently for the valid: we want a reliable estimate, so using more samples!
        features_fn_valid = partial(transform_feature_3d_v2, configuration=configuration, sample_volume_name='ct', only_valid_z=True, nb_samples=4)


        datasets = create_datasets_reservoir_map(
            read_splits(configuration), 
            configuration=configuration,
            load_case_fn=None,
            #path_classes=os.path.join(here, '../../config/foreground_only.json'),
            load_case_train_fn=configuration.data['load_case_train'],
            load_case_valid_fn=configuration.data['load_case_valid'], 
            preprocess_data_train_fn=[
                #TransformChangeLesionContrast(),
                #TransformGeneric(train_suv_augmentations_v2(), 'suv'),
                #TransformGeneric(train_ct_augmentations(), 'ct'),
                TransformGeneric(train_suv_augmentations(), 'suv'),
                configuration.data['preprocessing'],
                TransformUnsqueeze(axis=0, criteria_fn=criteria_fn),
                TransformRandomFlip(axis=1),
                TransformRandomFlip(axis=2),
                TransformRandomFlip(axis=3), 
                TransformSqueeze(axis=0),
                transform_augmentation_random_affine_transform,
                #partial(transform_export_maximum_intensity_projection, export_path='/mnt/datasets/ludovic/AutoPET/tmp'),

                # must be the last one for accurate distance transform
                #partial(transform_surface_loss_preprocessing, segmentation_name='seg', nb_classes=2, discard_background=False, normalized=True),
                #transform_label_segmentation,
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
            max_reservoir_samples=96 * 2,  # nb_samples % (samples_per_patient * batch_size) should be 0 for optimality
            min_reservoir_samples=48,
            nb_map_workers=configuration.training['map_workers'],
            nb_reservoir_workers=configuration.training['data_loading_workers'],
            max_reservoir_jobs_at_once_factor=40
        )

        try:                    
            optimizer = trw.train.OptimizerAdamW(
                learning_rate=configuration.training['learning_rate'], 
                weight_decay=configuration.training['weight_decay']).scheduler_cosine_annealing_warm_restart_decayed(
                    T_0=configuration.training['eval_inference_every_X_epoch'],
                    decay_factor=0.9
            ).clip_gradient_norm()

            model = Model()
            if config_start is not None:
                load_model(model, config_start, device=torch.device(configuration.training['device'].split(';')[0]), strict=True)
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