def test_common_import():
    from common.change_condition import delay_secondary_path
    from common.loading_real_wave_noise import print_stats
    from common.Reading_path_test import loading_paths

def test_data_import():
    from data.Adaptive_control_filter_generator_batch import adaptive_control_filter_batch
    from data.Automatic_Hard_Label_Subfilters_batch import create_data_loader
    from data.DataSet_construction import BandlimitedNoise_generation
    from data.DFT_Filter_Decompose import Creating_Filter
    from data.Disturbance_generation import Disturbance_generation_from_real_noise
    from data.Pretraining_sub_control_filters import save_mat__

def test_training_import():
    from training.Bcolors import bcolors
    from training.MyDataLoader import minmaxscaler
    from training.Testing_M5_Network_original import create_data_loader
    from training.Train_validate import init_weights

def test_fxlms_import():
    from fxlms.FxLMS_algorithm_v1 import FxLMS
    from fxlms.FxLMS_algorithm_v2 import FxLMS

def test_gfanc_import():
    from gfanc.Control_filter_selection import load_weigth_for_model
    from gfanc.Control_filter_selection_pre_now import load_weigth_for_model
    from gfanc.Fixed_filter_noise_cancellation_subfilters import Fixed_filter_controller
    from gfanc.M5_Network import CNN

def test_sfanc_fxlms_import():
    from sfanc_fxlms.Combine_SFANC_with_FxNLMS import FxNLMS
    from sfanc_fxlms.Control_filter_selectionSFANC import load_weigth_for_model
    from sfanc_fxlms.SFANC import SFANC_FxNLMS