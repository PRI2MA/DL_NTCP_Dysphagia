"""
Configuration file: containing global variables.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Whether to preprocess UMCG data or other data
use_umcg = True

# Paths
# TODO: temporary, for MDACC
if use_umcg:
    data_dir = '//zkh/appdata/RTDicom/PRI2MA/{}'
    data_collection_dir = '//zkh/appdata/RTDicom/HungChu/Data/PRI2MA'
    data_dir_citor = '//zkh/appdata/RTDicom/PRI2MA/{}'  # Contains all 1453 CITOR RTSTRUCTs
    save_dir_citor = 'D:/MyFirstData/raw_rtstructs_citor'  # Contains CITOR RTSTRUCTs for our patients in our cohort
    data_dir_dlc = '//zkh/appdata/RTDicom/PRI2MA/DLC'  # Contains (valid) DLC RTSTRUCTs
    save_root_dir = 'D:/MyFirstData'  # Contains saved files from scripts such as processed data, loggings, etc.
    save_root_dir_2 = '//zkh/appdata/RTDicom/HungChu/Data/PRI2MA'  # Contains saved files for dataset
    save_dir = os.path.join(save_root_dir_2, 'dicom_processed')
else:
    data_dir = '//zkh/appdata/RTDicom/PRI2MA/MDACC/{}'
    data_collection_dir = '//zkh/appdata/RTDicom/HungChu/Data/MDACC'
    data_dir_citor = '//zkh/appdata/RTDicom/PRI2MA/MDACC/{}'
    save_dir_citor = 'D:/MyFirstData/raw_rtstructs_mdacc'  # Contains CITOR RTSTRUCTs for our patients in our cohort
    data_dir_dlc = '//zkh/appdata/RTDicom/PRI2MA/MDACC/DLC UMCG'  # Contains (valid) DLC RTSTRUCTs
    save_root_dir = 'D:/MyFirstData_MDACC'  # Contains saved files from scripts such as processed data, loggings, etc.
    save_root_dir_2 = '//zkh/appdata/RTDicom/HungChu/Data/MDACC'  # Contains saved files for dataset
    save_dir = os.path.join(save_root_dir_2, 'dicom_processed')

# data_preproc_rtdose.py
save_dir_ct = save_dir
save_dir_rtdose = save_dir

# TODO: temporary, for MDACC
if use_umcg:
    # data_preproc_segmentation_map_citor.py
    save_dir_segmentation_map_citor = os.path.join(save_root_dir, 'segmentation_map_citor')  # CITOR segmentation_map
    # data_preproc_ct_segmentation_map_dlc.py
    save_dir_segmentation_map_dlc = os.path.join(save_root_dir, 'segmentation_map_dlc')  # DLC segmentation_map
    # data_preproc_ct_segmentation_map.py
    save_dir_segmentation_map = save_dir  # Every segmentation_map (CITOR + DLC)
    # data_preproc.py
    save_dir_dataset_full = os.path.join(save_root_dir, 'dataset_full')
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/preprocessed/dataset_full'
    save_dir_dataset = os.path.join(save_root_dir, 'dataset')
    # check_data_preproc.py
    save_dir_figures = os.path.join(save_root_dir, 'figures')
    save_dir_figures_all = os.path.join(save_dir_figures, 'all')
else:
    # data_preproc_segmentation_map_citor.py
    save_dir_segmentation_map_citor = os.path.join(save_root_dir, 'segmentation_map_mdacc')  # CITOR segmentation_map
    save_dir_segmentation_map_dlc = os.path.join(save_root_dir, 'segmentation_map_dlc_mdacc')  # DLC segmentation_map
    save_dir_segmentation_map = save_dir
    save_dir_dataset_full = os.path.join(save_root_dir, 'dataset_full_mdacc')
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/MDACC/preprocessed/dataset_full_mdacc'
    save_dir_dataset = os.path.join(save_root_dir, 'dataset_mdacc')
    save_dir_figures = os.path.join(save_root_dir, 'figures_mdacc')
    save_dir_figures_all = os.path.join(save_dir_figures, 'all')


# data_folder_types.py
# Available patient's folder types (e.g. 'with_contrast'/'no_constrast')
patient_folder_types = ['with_contrast', 'no_contrast']
# check_folder_structure.py
# TODO: temporary, for MDACC
# File extensions: required for finding the files. Usually this is '.dcm', but MDACC has no file extension, so ''.
# Note: these variables are only used in check_folder_structure.py, because in the remainder of the code
# we can assume that the folder structure is correct and hence glob('/*') and glob('/*.dcm') would output the same.
if use_umcg:
    ct_file_ext = '.dcm'  # '.dcm' (UMCG) | '' (MDACC)
    rtdose_file_ext = '.dcm'
    rtstruct_file_ext = '.dcm'
else:
    ct_file_ext = ''  # '.dcm' (UMCG) | '' (MDACC)
    rtdose_file_ext = ''
    rtstruct_file_ext = ''

# Filenames
# data_folder_types.py
filename_patient_folder_types_csv = 'patient_folder_types.csv'
# check_rtstruct.py
filename_overview_all_structures_csv = 'overview_all_structures.csv'
# data_preproc_ct_rtdose.py
filename_rtdose_dcm = 'Echte_Original_plan.dcm'
# check_folder_structure.py
filename_check_folder_structure_logging_txt = 'check_folder_structure_logging.txt'
filename_patients_incorrect_folder_structure_csv = 'patients_incorrect_folder_structure_{}.csv'
# check_data.py
filename_structures_values_json = 'structures_values.json'
filename_check_data_logging_txt = 'check_data_logging.txt'
filename_patients_in_citor_and_dlc_txt = 'patients_in_citor_and_dlc.txt'
# folders.py
filename_frame_of_reference_uid_csv = 'frame_of_reference_uid.csv'
filename_invalid_uid_csv = 'invalid_uid.csv'
# data_preproc_ct_rtdose.py
filename_data_preproc_ct_rtdose_logging_txt = 'data_preproc_ct_rtdose_logging.txt'
filename_ct_npy = 'ct.npy'  # CT (Numpy array)
filename_rtdose_npy = 'rtdose.npy'  # RTDOSE (Numpy array), has the same shape as ct.npy
filename_rtdose_metadata_json = 'rtdose_metadata.json'  # meta-data of the RTDOSE data
filename_ct_metadata_json = 'ct_metadata.json'  # meta-data of the CT data
# check_data_preproc_ct_rtdose.py
filename_check_data_preproc_ct_rtdose_logging_txt = 'check_data_preproc_ct_rtdose_logging.txt'
filename_ct_npy = 'ct.npy'
filename_rtdose_npy = 'rtdose.npy'
filename_overview_ct_metadata_csv = 'overview_ct_metadata.csv'
filename_overview_ct_csv = 'overview_ct.csv'
filename_overview_rtdose_csv = 'overview_rtdose.csv'
# data_preproc_segmentation_map_citor.py
filename_data_preproc_ct_segmentation_map_citor_logging_txt = 'data_preproc_ct_segmentation_map_citor_logging.txt'
filename_data_preproc_ct_segmentation_map_best_logging_txt = 'data_preproc_ct_segmentation_map_best_logging.txt'
filename_structures_with_contouring_issues_citor_i_txt = 'structures_with_contouring_issues_citor_{i}.txt'
filename_patients_with_contouring_issues_citor_txt = 'patients_with_contouring_issues_citor.txt'
filename_patient_ids_with_filename_csv = 'patient_ids_with_filename.csv'
filename_overview_all_structures_count_citor_csv = 'overview_all_structures_count_citor.csv'
filename_overview_structures_count_citor_csv = 'overview_structures_count_citor.csv'
# data_preproc_ct_segmentation_map_dlc.py
filename_data_preproc_ct_segmentation_map_dlc_logging_txt = 'data_preproc_ct_segmentation_map_dlc_logging.txt'
# data_preproc_ct_segmentation_map.py
filename_data_preproc_ct_segmentation_map_logging_txt = 'data_preproc_ct_segmentation_map_logging.txt'
filename_data_preproc_ct_segmentation_map_cropping_logging_txt = 'data_preproc_ct_segmentation_map_cropping_logging.txt'
filename_segmentation_map_npy = 'segmentation_map.npy'
filename_segmentation_map_i_npy = 'segmentation_map_{i}.npy'  # segmentation_map of all structures
filename_structures_value_count_json = 'structures_value_count.json'
filename_structures_value_count_i_json = 'structures_value_count_{i}.json'  # for each relevant structure: contains
# segmentation label map value and the number of voxel data points (count)
filename_overview_structures_count_dlc_csv = 'overview_structures_count_dlc.csv'
# most voxel data points + corresponding count
filename_structures_with_contouring_issues_dlc_txt = 'structures_with_contouring_issues_dlc.txt'
# filename_structures_with_contouring_issues_i_txt = 'structures_with_contouring_issues_{i}.txt'
filename_patients_with_contouring_issues_dlc_txt = 'patients_with_contouring_issues_dlc.txt'
# filename_patients_with_contouring_issues_txt = 'patients_with_contouring_issues.txt'
filename_best_count_files_json = 'best_count_files.json'  # for each relevant structure: filename that contains
# most voxel data points + the corresponding count
filename_cropping_regions_csv = 'cropping_regions.csv'  # for each patient and each dimension: contains the upper and
# lower cropping coordinate and the length (i.e. distance between upper and lower coordinate)
# count >= minimal_count
# check_data_preproc_ct_segmentation_map.py
filename_check_data_preproc_ct_segmentation_map_logging_txt = 'check_data_preproc_ct_segmentation_map_logging.txt'
filename_cropping_regions_stats_csv = 'cropping_regions_stats.csv'
filename_segmentation_fraction_outside_ct_csv = 'segmentation_fraction_outside_ct.csv'
# filename_all_structures_csv = 'all_structures.csv'  # all structures of all patients
# filename_valid_and_relevant_structures_csv = 'valid_and_relevant_structures.csv'  # valid structure if
# data_preproc.py
filename_data_preproc_array_logging_txt = 'data_preproc_array_logging.txt'
filename_data_preproc_features_logging_txt = 'data_preproc_features_logging.txt'
filename_endpoints_csv = 'endpoints.csv'
filename_patient_id_npy = '{patient_id}.npy'  # model's input (Numpy array), concatenation of: CT, RTDOSE and RTSTRUCT
filename_features_csv = 'features.csv'
filename_stratified_sampling_csv = 'stratified_sampling.csv'
# segmentation arrays
filename_overview_structures_count_csv = 'overview_structures_count.csv'
# check_data_preproc.py
filename_check_data_preproc_logging_txt = 'check_data_preproc_logging.txt'
# data_preproc_exclude_patient.py
filename_exclude_patients_csv = 'exclude_patients.csv'
# main.py
filename_main_logging_txt = 'main_logging.txt'

# Modelling
# data_preproc.py: filename_endpoints_csv
patient_id_col = 'PatientID'
endpoint = 'Dysphagia_6MO'
baseline_col = []
submodels_features = [
    ['OralCavity_Ext_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'BSLtox_grade0_1', 'BSLtox_grade2', 'BSLtox_grade3_4', 'Loctum_Pharynx', 'Loctum_Larynx', 'Loctum_OC'],
    ['PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'BSLtox_grade0_1', 'BSLtox_grade2', 'BSLtox_grade3_4', 'Loctum_Pharynx', 'Loctum_Larynx', 'Loctum_OC'],
]  # Features of submodels. Should be a list of lists. len(submodels_features) = nr_of_submodels. If None, then
# no fitting of submodels.
features = ['OralCavity_Ext_meandose', 'PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 
            'BSLtox_grade2', 'BSLtox_grade3_4', 'Loctum_Pharynx', 'Loctum_Larynx']  # Features of final model. Elements in submodels_features should
# be a subset of `features`, i.e. submodels_features can have more distinct features than the final model.
lr_coefficients = None  # [-4.5710, 0.0341, 0.0267, 0.0107, 0.0151, 1.0627, 1.4610, -0.7115, -0.8734]  # Values starting with coefficient for `intercept`,
# followed by coefficients of `features` (in the same order). If None, then no predefined coefficients will be used.
# on our own training dataset
ext_features = ['CT+C_available', 'CT_Artefact', 'Photons', 'Split']
# (Stratified Sampling)
# data_preproc.py, check_data_preproc_ct_segmentation_map.py: relevant segmentation structures
parotis_structures = ['parotis_li', 'parotis_re']
submand_structures = ['submandibularis_li', 'submandibularis_re']
lower_structures = ['crico', 'thyroid']
mandible_structures = ['mandible']
other_structures = ['glotticarea', 'oralcavity_ext', 'supraglottic', 'buccalmucosa_li',
                    'buccalmucosa_re', 'pcm_inf', 'pcm_med', 'pcm_sup', 'esophagus_cerv']
structures_compound_list = [parotis_structures, submand_structures, lower_structures, mandible_structures,
                            other_structures]
structures_uncompound_list = [i for x in structures_compound_list for i in x]
# E.g. structures_uncompound_list: ['parotid_li', 'parotid_re', 'submandibular_li', ...]
structures_values = dict(zip(structures_uncompound_list,
                             range(1, len(structures_uncompound_list) + 1)))
# E.g.: structures_values: {'parotid_li': 1, 'parotid_re': 2, 'submandibular_li': 3, ...}
max_outside_fraction = 0.1  # maximum fraction that the segmentation_mask of a structure is allowed to be outside CT
max_outside_lower_limit = -200  # lower limit CT intensity (HU) for which segmentation_mask should correspond to

# Data collection/preparation
# data_collection.py
filename_patient_ids = 'Patient_ids_{}.csv'
# TODO: important note: currently only len(mode_list) == len(save_dir_mode_list) == 1 is allowed!
# TODO: a possibility to extend is to add '/{}'.format(d_i) to save_root_dir
# TODO: temporary, for MDACC
if use_umcg:
    mode_list = ['now']  # 'now', 'later', 'imputation', 'new_patients']
    save_dir_mode_list = [
     'Now_Has_endpoint_for_at_least_1_toxicity']  # 'Now_Has_endpoint_for_at_least_1_toxicity', 'Later - Could have endpoint later', 'Patients for imputation', 'to_be_processed_by_hung_do_not_change_folder']
else:
    mode_list = ['mdacc']
    save_dir_mode_list = ['DICOM_DATA_anonymized_cleaned']

# mode_list = ['new_patients']  # , 'later', 'imputation']
# save_dir_mode_list = ['New_patients']  # , 'Later - Could have endpoint later', 'Patients for imputation']
# save_dir_mode_list = ['Postoperative patients/Now - Has endpoint for at least 1 toxicity',
#                       'Postoperative patients/Later - Could have endpoint later',
#                       'Postoperative patients/Patients for imputation']
# folders.py
# study_uid_tag = '0020000D'  # Study Instance UID
# series_uid_tag = '0020000E'  # Series Instance UID
frame_of_ref_uid_tag = '00200052'
patient_id_tag = '00100020'
# data_preproc_ct_rtdose.py/data_preproc_ct_segmentation_map.py
ct_dtype = 'int16'
rtdose_dtype = 'uint16'
# segmentation_dtype = 'int16'
# data_preproc_ct_segmentation_map_citor.py/data_preproc_ct_segmentation_map.py
keep_segmentation_map_elements = True  # In case of adding segmentation maps with overlapping elements: whether to keep
# overlapping elements from existing segmentation map (True) or the new segmentation map (False)
# data_preproc_ct_segmentation_map.py
minimal_count = 100  # Count lower bound: for each patient: if the number of voxels for a structure is more than
# or equal to this value, then the segmentation_map of the structure will be considered to be valid
nr_of_decimals = 2  # Number of decimals used for saving in (csv) files
# data_preproc.py, data_preproc_ct_segmentation_map.py
# Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int, because of Excel file) to
# patient_id = '0111766' (type=str)
unfill_holes = False  # Whether or not to consider unfilled holes of contours (see oralcavity_ext structure)
# TODO: temporary, for MDACC
if use_umcg:
    patient_id_length = 7
else:
    patient_id_length = 10
perform_spacing_correction = True
perform_cropping = True
perform_clipping = True
perform_transformation = False
spacing = [2., 2., 2.]  # (only if perform_spacing=True) desired voxel spacing
y_center_shift_perc = 0.0  # (only if perform_cropping=True) percentage of total num_rows to shift the y_center. If
# y_center_shift_perc > 0, then the bounding box will be placed on higher y-indices of the array. Vice versa for
# y_center_shift_perc < 0. If y_center_shift_perc = 0, then y_center will not be shifted.
bb_size = [100, 100, 100]  # (only if perform_cropping=True) bounding box (/cropping) with
# shape = [z_size, y_size, x_size] = [num_slices, num_rows, num_columns] (default of Numpy) given in the spacing (!!!)
input_size = [128, 128, 128]  # (only if perform_transformation=True) model input size,
# shape = [z, y, x] = [num_slices, num_rows, num_columns] (default of Numpy)
label_resize_mode = 'nearest'  # (only if perform_transformation=True)
non_label_resize_mode = 'trilinear'  # (only if perform_transformation=True)
ct_min = -1000  # (only if perform_clipping=True) CT pixel value lower bound (Hounsfield Units)
ct_max = 1000  # (only if perform_clipping=True) CT pixel value upper bound (Hounsfield Units)
# data_preproc_exclude_patient.py
rtdose_lower_limit = 6000  # Minimum rtdose.max() allowed
rtdose_upper_limit = 8500  # Maximum rtdose.max() allowed

# Plotting
# check_data_preproc.py
ct_ww = 600  # (plotting) Window Width: the range of the numbers (i.e. max - min)
ct_wl = 100  # (plotting) Window Level: the center of the range
plot_nr_images = 36  # If None, then consider all slices
figsize = (12, 12)
plot_nr_images_multiple = 8
figsize_multiple = (18, 9)
# Colormap
ct_cmap = 'gray'
# Custom cmap for RTDOSE: sample from the colormap 'jet', consider 256 colors in total.
# The upper part (0.8 - 0.9) (orange --> red) of plt.cm.jet will be mapped to upper int(256 * 0.05) = 12 values
# out of 256 values
# The middle part (green --> orange) of plt.cm.jet will be mapped to mid 256 - 12 - 201 = 43 values out of 256 values
# The lower half (blue --> green) of plt.cm.jet will be mapped to lower 201 values out of 256 values
# Result: map `blue (0) --> green (3325) --> red (6650)` to `blue (0) --> green (5225) --> orange () --> red (6650)`
# Source: https://stackoverflow.com/questions/49367144/modify-matplotlib-colormap
rtdose_vmax = 7000 * 0.95  # = 6650
rtdose_vmid = 5500 * 0.95  # = 5225
rtdose_vmin = 0  # optional (else: None)
# 0.9: jet colormap at 90% is red (whereas 100% is dark red)
rtdose_cmap_upper = plt.cm.jet(np.linspace(0.8, 0.9, int(256 * 0.05)))
rtdose_cmap_mid = plt.cm.jet(np.linspace(0.5, 0.8, 256 - int(256 * 0.05) - int(rtdose_vmid / rtdose_vmax * 256)))
rtdose_cmap_lower = plt.cm.jet(np.linspace(0, 0.5, int(rtdose_vmid / rtdose_vmax * 256)))  # int(5225/6650 * 256) = 201
rtdose_cmap = colors.LinearSegmentedColormap.from_list('rtdose_cmap', np.vstack((rtdose_cmap_lower, rtdose_cmap_mid,
                                                                                 rtdose_cmap_upper)))
segmentation_cmap = 'gray'
draw_contour_structures = parotis_structures + submand_structures  # create contour of these structures in the figures
draw_contour_color = 'r'  # color of the contour line
draw_contour_linewidth = 0.25  # width of the contour line
ct_ticks_steps = 100  # optional (else: None)
rtdose_ticks_steps = 1000  # optional (else: None)
segmentation_ticks_steps = 1  # optional (else: None)
ct_colorbar_title = 'HU'
rtdose_colorbar_title = 'cGy'
segmentation_colorbar_title = ''

# Run whole data_preproc pipeline for a small number of patients, useful for testing
test_patients_list = None
# ['1041973121',
#  '1164546699',
#  '1173728658',
#  '1194116893',
#  '1338095994',
#  '1410220807',
#  '1650313147',
#  '1661689317',
#  '1681729379',
#  '1702727338',
#  '1758588673',
#  '1838804685',
#  '2269261955',
#  '2278293118',
#  '2279280705',
#  '2304398819',
#  '2426779986',
#  '2592917372',
#  '2674720843',
#  '2719881586',
#  '2767317435',
#  '2767720520',
#  '2843895295',
#  '2929571068',
#  '3377099994',
#  '6630811724',
#  '7093179577',
#  '7439564662',
#  '8609286814',
#  '9312513183']
# test_patients_list = None  #  ['1014175846']  # ['0345451', # correct, but too high]
# ['0020715', '0059896', '0170637', '0194349', '0339740', '0345451', '0400509', '0401057', '0507025',
#                       '0585519', '0662515', '0669746', '0694884', '0707556', '0810853', '0851961', '0916891', '0996854',
#                       '1004430', '1051551', '1052515', '1053592', '1165826', '1270521', '1320826', '1437642', '1448368',
#                       '1453495', '1495796', '1512121', '1525367', '1621848', '1737477', '1837605', '1845884', '1872399',
#                       '1942221', '1973344', '2000655', '2016613', '2021247', '2022769', '2022790', '2024557', '2027942',
#                       '2029133', '2030769', '2041536', '2044268', '2044374', '2046282', '2049861', '2060828', '2061972',
#                       '2063466', '2064239', '2067099', '2068165', '2069690', '2072318', '2078256', '2078693', '2080097',
#                       '2080453', '2080643', '2080913', '2081189', '2081433', '2083033', '2083415', '2087361', '2093206',
#                       '2097425', '2102599', '2102777', '2103483', '2103516', '2103554', '2107399', '2110034', '2110613',
#                       '2111096', '2114930', '2115835', '2116597', '2120032', '2122959', '2125979', '2130043', '2133724',
#                       '2136180', '2141516', '2141822', '2142665', '2143076', '2145422', '2145827', '2146277', '2146594',
#                       '2147034', '2147836', '2148052', '2148795', '2149291', '2153786', '2155717', '2156098', '2159400',
#                       '2160438', '2160642', '2160810', '2161834', '2161948', '2162043', '2163243', '2163523', '2164819',
#                       '2164854', '2167267', '2167798', '2168954', '2168981', '2171148', '2171499', '2171627', '2171769',
#                       '2175318', '2176979', '2179303', '2179534', '2179568', '2180403', '2180850', '2182845', '2186039',
#                       '2188517', '2188849', '2190498', '2191427', '2194374', '2195692', '2196033', '2197393', '2198770',
#                       '2199947', '2200789', '2201084', '2201781', '2201897', '2202350', '2202380', '2204685', '2208559',
#                       '2209518', '2210901', '2215342', '2219962', '2220893', '2222276', '2241013', '2241472', '2244814',
#                       '2246159', '2248028', '2297238', '2309651', '2397864', '2551710', '2580821', '2584522', '2596991',
#                       '2678623', '2740797', '2781091', '2970220', '2972492', '3052392', '3144104', '3196406', '3205561',
#                       '3220326', '3274619', '3307103', '3325216', '3414277', '3482925', '3511140', '3534035', '3649834',
#                       '3761226', '3902763', '3941688', '3979255', '4012516', '4022677', '4223355', '4224256', '4432174',
#                       '4610810', '4617509', '4623812', '4640464', '4667104', '4673928', '4693746', '4777783', '4787329',
#                       '4847222', '4889766', '4911587', '5088300', '5222055', '5262401', '5471790', '5571619', '5586522',
#                       '5610140', '5637480', '5641953', '5669388', '5720176', '5762672', '5776290', '5827453', '5843359',
#                       '5860527', '5896858', '5908531', '5921911', '5935505', '5952236', '6038065', '6152392', '6226482',
#                       '6469382', '6477418', '6504420', '6661876', '6786511', '6803098', '6898633', '6959182', '7000793',
#                       '7019226', '7040261', '7061207', '7173670', '7207910', '7209497', '7288956', '7351271', '7390463',
#                       '7499421', '7689583', '7776845', '7805267', '7829060', '7873700', '7876032', '7982805', '8127703',
#                       '8341776', '8343775', '8348176', '8515413', '8534991', '8667722', '8756306', '8780297', '8818981',
#                       '8959829', '9028258', '9152961', '9325321', '9462318', '9577277', '9692367', '9946201',
#                       '9980149']  # ['2145827'] # ['2159400', '2160810', '2180850'] # List of patient_ids, else None
# ['1621848', '2041536', '2167798', '2201897', '5860527', '8350445', '8667722']
# label 0: 2041536, 2201897, 8350445,
# label 1: 1621848, 2167798, 5860527, 8667722
