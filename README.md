# Deep Learning NTCP Dysphagia modelling
This code is adapted from https://github.com/PRI2MA/DL_NTCP_Xerostomia to make it work for dysphagia predictions. 


## Running the code
##### Requirements
- Raw dataset folder: see `Data collection` in section `Detailed description` below.
- `endpoints.csv`: for each patient_id: endpoint values (0 or 1) and features for logistic regression 
model and/or Deep Learning model. The features should be in the form such that it does not require any preprocessing 
(e.g. normalization) anymore, because those values in this file will be used directly by the model(s).

##### Data preprocessing
```
# Change directory to data preprocessing folder
$ cd D:\Github\dl_ntcp_dysphagia\data_preproc

# Run main script
$ python main.py
```
##### Requirements
- `dataset` folder from the data preprocessing step, containing the folder `0`, `1` and the file `features.csv`.
- `variables.csv` (only if `sampling_type = 'stratified'`): for each `patient_id`: group of variables and their values to 
perform stratified sampling on.

##### Model training and evaluation
```
# Change directory to parent folder
$ cd D:\Github\dl_ntcp_xerostomia

# Run main script
$ python main.py
```


## Warnings and data bugs

(Data)bugs and their corresponding corrections can be found by searching for `# BUG` in the code files.

- (`data_preproc_ct_segmentation_map_citor.py`) The notations in CITOR RTSTRUCTs of a certain structure can 
be slightly different between different files. These corrections are dealt with and can be found in the function 
modify_structure_citor().
  - `PMC_...` (incorrect) instead of `PCM...` (correct) (`patient_id = xxxx`)
  - `Cricopharyngeus` instead of `crico`
  - `Supraglotticlarynx` instead of `supraglottic` (`patient_id = xxxx`)
- (`data_preproc_ct_segmentation_map_citor.py`) There is an incorrect contour part for `patient_id = xxxx`: slice 101, 
y: 100 - 102, x: 120 - 122 (see segmentation_map[101,97:106,117:126]), structure: `oralcavity_ext`. If we use the 
segmentation_maps of all structures (or: at least `oralcavity_ext`) for determining the `x`-dimension cropping region, 
then this will mess up `x_lower` and hence `x_center`.


#### Notes
- Mirada's Deep Learning Contour (DLC) segmentation model was trained on CITOR segmentations.
- Toxicities (to be) used as endpoint/label/class/target/output are: 
  - Xerostomia (dry mouth)
  - Sticky saliva (i.e. mucus)
  - Dysphagia (i.e. swallowing dysfunction)
  - Taste alteration
  - Tube feeding dependence


## Detailed description


#### Data collection
1. For every patient: same data with folder structure (e.g. `patient_id = 0123456`):
   - `0123456`:
     - `with_contrast` / `no_contrast`:
       - `CT`: X DICOM files, where X = number of slices.
       - `RTDOSE`: one DICOM file allowed.
       - `RTSTRUCT`: multiple DICOM files allowed.
2. For some patients with missing structures: create and save Mirada's DLC in another 
folder with folder structure (e.g. `patient_id = 0123456`):
     - `0123456`
       - `with_contrast` / `no_contrast`:
         - `RTSTRUCT`:
           - one DICOM file allowed.


#### Data_prep

Summary: 
1. Register RTDOSE to CT.
2. Register segmentation map of the relevant structures to CT.
3. Combine the three arrays CT+RTDOSE+segmentations into single array of shape
(3, z, y, x) = (num_input_types, num_slices, num_rows, num_columns).
4. Extract features for logistic regression model and/or Deep Learning model.


##### Terminologies
- `Structure` == `Organ At Risk` (OAR). In the code files we use the notation `structure` (instead of
`Organ At Risk` / `OAR`) because this is conforming the notation in the RTSTRUCT files.
- (Segmentation) mask vs. map: both are arrays, but a mask is binary (0, 1) while
    a map has integer values (0, 1, 2, ...). Each value corresponds to a different
    structure. E.g. if parotid has value = 6, then map == value will return a mask.
- Relevant structures: structures for which their segmentation that are expected to have impact on the NTCP modeling 
performance. The names of the relevant structures can be found in `data_preproc_config.structure_names_compound_list`. 
IMPORTANT: the relevant structures are assumed to be non-overlapping. If a relevant structure does overlap,
then we keep the overlapped voxel of the relevant structure with the highest priority. The priority is defined in
`data_preproc_config.structures_compound_list` (highest priority first).
- Own consistent value map for the structures: different RTSTRUCTs have different segmentation label for the same
structure, e.g. the structure `Parotid_L` could have value = 6 in the segmentation_map of an RTSTRUCT, but it could 
have value = 7 in the segmentation_map of another RTSTRUCT. To deal with this, we map these inconsistent values to 
consistent values, see `structure_names_values` in `data_preproc_config.py`. For example, `parotis_li` has always 
value = 1 in our segmentation_map.
- Valid segmentation: if number of voxels for that segmentation >= `data_preproc_config.minimal_count`.


##### Output folders
- `dataset` (from `data_preproc.py`): contains subfolders of each endpoint / classification labels. The subfolders contain 
the Numpy arrays of the patients from the corresponding endpoint. 
- `dicom_processed` (from `data_preproc_ct_rtdose.py`): contains CT and RTDOSE (resampled to CT) as Numpy array 
    and are used in the last data preparation step (i.e. `data_preproc.py`).
- `figures` (from `check_data_preproc.py`): contains 2D subplots of CT, RTDOSE and segmentation_map for a number of 
equally-distant slices.
- `segmentation_map` (from `data_preproc_ct_segmentation_map.py`): contains the final segmentation maps as Numpy array 
    and are used in the last data preparation step (i.e. `data_preproc.py`). These segmentation maps are 
    combinations of the intermediate segmentation maps from CITOR and DLC.
- `segmentation_map_citor` (from `data_preproc_ct_segmentation_map_citor.py`): contains intermediate segmentation 
    maps as Numpy array, obtained from CITOR RTSTRUCT(s). The segmentation maps only contain voxels of the 
    relevant structures.
- `segmentation_map_dlc` (from `data_preproc_ct_segmentation_map_dlc.py`): contains intermediate segmentation maps 
    as Numpy array, obtained from DLC RTSTRUCT. The segmentation maps only contain voxels of the relevant 
    structures.



##### Steps

In general, RTDOSE and segmentation_maps will be matched with the CTs. Functions in `data_preproc_functions.py` 
are general functions used by the data_preproc files below.


1. `data_folder_types.py`
- For every patient: determine which `folder_type` we have to consider, i.e. either `with_contrast` or `no_contrast`. 
This is necessary for loading the correct folders inside the patient's folder. Important note: if a patient has both
`with_contrast` and `no_contrast` folder, then the data preprocessing pipeline will load the data in all folders of 
`with_contrast`. However, if certain data folders are not present, then we will load the data in those folders of the
`no_contrast` counterpart. So this only holds for the folders that are not available in `with_contrast`.


2. `data_collection.py`
- For every patient: using the data in the DICOM headers, check whether the Frame of Reference (FoR) UID 
(0020,0052) between CT, RTDOSE and RTSTRUCT matches or not. More information on FoR UID can be found here:
  - https://comp.protocols.dicom.narkive.com/RO1W4rJG/frame-of-reference-uid-tag-question
  - https://discourse.itk.org/t/compose-image-from-different-modality-with-different-number-of-slices/2286/15
  - https://stackoverflow.com/questions/30814720/dicom-and-the-image-position-patient
  - https://dicom.innolitics.com/ciods/rt-plan/frame-of-reference/00200052
  - https://www.ihe.net/uploadedFiles/Documents/Radiation_Oncology/IHE_RO_Suppl_MMRO-III_Rev1.1_TI_2016-11-16.pdf
    (see multimodality_image_registration.pdf in the documents folder).
- Create `invalid_uid.csv` file: list of patient_ids with non-matching CT, RTDOSE and/or RTSTRUCT Frame of Reference UID.
- If the RTDOSE does not match with the CT, then the data preprocessing of this patient will still continue. 
This is for time-efficiency: to make sure that the whole data preprocessing pipeline does not suddenly stop.
The patients for which RTDOSE and CT do not match will be added to `exclude_patients.csv` and hence will not be 
considered for the modeling, see `load_data.py`. After their data files have been fixed, we can re-run the 
data preprocessing pipeline for these patients by adding them in `data_preproc.config.test_patients_list`. 
This will prevent us from re-running the data preprocessing pipeline for the whole dataset.
- Note: non-matching of Frame of Reference UID of RTSTRUCTs can be resolved by creating Mirada's DLC using the CT slices 
of interest.


3. `data_preproc_ct_rtdose.py`
- Register RTDOSE array and CT array using sitk.Resample.
- Save CT and RTDOSE metadata (such as `Direction`, `Origin`, `Size`/`Shape`, `Spacing`). The CT metadata is needed for
spacing correction of the different arrays: CT, RTDOSE, segmentation_map.
- Save RTDOSE array (registered to CT) and CT array.


4. `data_preproc_ct_segmentation_map_citor.py`
- Patients can have multiple RTSTRUCTs (`i`=0, 1, ...), each RTSTRUCT can have different structures:
  - Load and save each segmentation_map (`segmentation_map_{i}.npy`) and structures_value_count + modified_name
  (`structures_value_count_{i}.json`) to `segmentation_map_citor` folder, where modified_name is our own consistent
  notation of the structure names, because different sources (e.g. CITOR vs. DLC) have different notations for the 
  same structure (e.g. `Parotid_L` vs. `DLC_Parotid_L` --> map both to `parotis_li`).
  - Combine segmentation_maps to obtain best_segmentation_map: 
    - Start with zeros Numpy array.
    - For each relevant structure: 
      - Check which (of the multiple) segmentation has largest count (using `structures_value_count_{i}.json`): 
        - Load the corresponding segmentation_map and add its voxels to initial zeros array with value from 
        own consistent value map. 
    - The combined segmentation_map (`segmentation_map_citor.npy`) and will be saved to `segmentation_map_citor` folder.
    - The combined segmentation_map only contains relevant + valid structures.
- Create `overview_all_structures_count_citor.csv` file: contains ALL structures (so for ALL RTSTRUCTs) and their 
corresponding counts. We can use this overview to check all structure names in all RTSTRUCTs, potentially typos. 
For example, for `patient_id = 0111766` the structure `PCM` was notated as `PMC`.
- Create `overview_structures_count_citor.csv` file: contains the relevant structures for the best RTSTRUCTs and their
corresponding counts. We can use this overview to see whether the relevant structures in the CITOR segmentation_maps
exists, and if so, with how many voxels. If the number of voxels for one or more structure is 0, then we will use
the segmentation map of DLC (see below).


5. `data_preproc_ct_segmentation_map_dlc.py`
- Similar to `data_preproc_ct_segmentation_map_citor.py`. DLC segmentation_maps are created for the patients for which
we do not have CITOR segmentation_maps, or if the CITOR segmentation_maps misses voxels for certain relevant structures. 
- (Difference) Patients are assumed to have only one RTSTRUCT file. 
- Create DLC segmentation_map, and for the output segmentation_map (segmentation_map_dlc.npy): 
  - Start with zeros Numpy array (with same shape as DLC segmentation map).
  - For each relevant structure: 
    - Add its voxels to initial zeros array using own consistent value map. 
  - The segmentation_map (`segmentation_map_dlc.npy`) will be saved to `segmentation_map_dlc` folder.
  - The segmentation_map only contains the relevant structures.
- Create `overview_structures_count_dlc.csv`: contains the relevant structures and their corresponding counts. We can use 
this overview to see whether the relevant structures in the DLC segmentation_maps exists, and if so, with how many 
voxels.
- Note: if a patient has a segmentation map for a relevant structure in both CITOR and DLC, then we will consider the
segmentation map from CITOR instead of DLC.

	
6. `data_preproc_ct_segmentation_map.py`
- Here we combine the CITOR and DLC segmentation_maps. The output contains voxels for every relevant structure (see 
`data_preproc_config.structures_compound_list`), and have value according to `data_preproc_config.structures_values`. 
This is our own consistent value map for the relevant structures. 
IMPORTANT: the relevant structures are assumed to be non-overlapping. If a relevant structure does overlap,
then we keep the overlapped voxel of the relevant structure with the highest priority. The priority is defined in
`data_preproc_config.structures_compound_list` (highest priority first).
- For each patient: load `segmentation_map_citor.npy` (if any) and `segmentation_map_dlc.npy` (if any)
- For each relevant structure:
  - Check whether the structure value is valid in `segmentation_map_citor.npy`. 
  - If it is not valid, set `segmentation_map_citor.npy` value for that structure to 0 and add segmentation from 
  `segmentation_map_dlc.npy`. Observe that we will use all valid structures from CITOR and that DLC will only be used 
  to make up for missing structures in CITOR.
  - Note: if the structure has more voxels in DLC than in CITOR, then we will still consider the segmentation
  of CITOR instead of DLC.
  - Perform spacing_correction if `data_preproc_config.perform_spacing_correction = True`. 
- Save the segmentation_map (`segmentation_map.npy`) to `segmentation_map` (later in: DICOM_processed) folder.
- The segmentation_map only contains the relevant structures and is potentially spacing-corrected.
- Create `overview_structures_count.csv`: contains the relevant structures and their corresponding counts. We can use 
this overview to see whether the relevant structures in the final segmentation_maps exists, and if so, with how many 
voxels.
- Determine cropping region from (potentially spacing-corrected) segmentation_map. The cropping region will be used
for cropping a bounding box area of CT, RTDOSE and segmentation_map if `data_preproc_config.perform_cropping = True`.
  - `z`: obtain most upper and most lower voxel (with value > 0) in z-dimension of the segmentation map (containing
  all relevant structures).
  - `y`: obtain most upper and most lower voxel (with value > 0) in y-dimension of the segmentation map (containing
  all relevant structures).
  - `x`: obtain most upper and most lower voxel (with value > 0) in x-dimension of the segmentation map (containing
  all relevant structures).
        
	
7. `data_preproc.py`
- Create `dataset_full` folder: create and store the input Numpy array for all patients by concatenating their 
CT+RTDOSE+RTSTRUCT. Note: we also create such arrays for patients that are not included for a specific study (e.g. 
prediction of xerostomia). Those patients are included in one or more other studies, e.g. prediction of dysphagia and/or 
taste. 
- Create `dataset` folder: in this folder we make a distinction between patients of different endpoints/labels 
(`0`, `1`). This is required for PyTorch Dataset and DataLoader class objects. Moreover, the dataset folder should 
contain subfolders, where the subfolder names are the endpoint values. The subfolders should contain the Numpy array of 
the corresponding patients. E.g., the subfolder `0` should contain the arrays of the patients with endpoint `0`. 
Similarly for endpoint `1`, endpoint `2` (if any), etc.
- Load CT, RTDOSE and segmentation_map Numpy arrays and perform none or some of the following operations. The input 
modalities that could undergo the corresponding operation are indicated in parentheses below.
  - `(z, y, x)`-spacing_correction (CT, RTDOSE). Note: segmentation_map is already spacing-corrected in 
  `data_preproc_ct_segmentation_map.py` if `data_preproc_config.perform_spacing_correction = True`. Therefore, we do not 
  have to perform spacing_correction here.
  - Cropping subset area using bounding box size (CT, RTDOSE, segmentation_map). For the cropping we:
    - Determine the `z/y/x-center` coordinate separately using `z/y/x-upper` and `-lower limits` in `cropping_regions.csv`.
    - Construct a bounding box around this center. The `z/y/x-size` of the bounding box is defined by `bb_size` in 
    `data_preproc_config.py`. The bounding box z-coordinate range will be `[z-center - z_size / 2, z_center + z_size / 2]`, 
    similarly for `y` and `x-coordinates`.
    - Crop the array to keep the voxels inside the bounding box. The cropped array will have shape `bb_size`. 
  - Value clipping (CT).
  - Transformation (e.g. resizing) (CT, RTDOSE, segmentation_map).
- These operations are performed based on the boolean value of perform_spacing_correction, perform_cropping, 
perform_clipping and perform_resize in `data_preproc_config.py`.
- Add channel dimension, i.e. convert array shape from `(z, y, x)` to `(1, z, y, x)`. This is required for concatenating
the different input modalities to single array.
- Save the concat([CT, RTDOSE, segmentation_map]) Numpy array to the corresponding subfolder of `dataset` folder.
- Load csv with features (and labels), and save the relevant features: baseline xerostomia (`baseline`), mean doses 
contralateral parotid gland (`D_contra_0`).


8. `data_preproc_exclude_patients.py`
- Create a csv file with patient_ids with invalid data. That is, at least one of the following criteria is met:
  - Invalid dose values, see `data_preproc_config.rtdose_lower_limit` and `.rtdose_upper_limit`.
  - Invalid segmentatioon, i.e. large segmentation fraction outside CT, see `data_preproc_config.max_outside_fraction`
  and `data_preproc_config.max_outside_lower_limit`.
  - Non-matching UID between CT and RTDOSE. See `data_collection.py`.
- Note: we do perform data_preproc for all patients, but those in `exclude_patients.csv` will not be considered in 
the modeling phase, see `load_data.py`.


#### Checks

We perform checks and logging of the different files to make sure that the intermediate and final outputs are 
correctly.

1. `check_folder_structure.py`
- For every patient: check whether its folder structure is as expected. Else the data preprocessing will not work
correctly.
- Create `patients_incorrect_folder_structure.csv`: table with patient_ids for which the folder structure is incorrect.

2. `check_rtstruct.py`
- Currently not relevant.

3. `check_rtdose.py`
- Currently not relevant.

4. `check_data.py`
- Save `data_preproc_config.structures_values`: {'parotis_li': 1, 'parotis_re': 2, 'submandibularis_li': 3, ...} as json 
file.
- Print patient_id's for which we have CITOR data (CT, RTDOSE and/or segmentation), DLC segmentation or neither.

5. `check_data_preproc_ct_rtdose.py`
- Create overview_ct_metadata.csv: table with the CT metadata for all patients.

6. `check_data_preproc_ct_segmentation_map.py`
- Create cropping_regions_stats.csv: descriptive statistics of cropping regions based on (potentially spacing-corrected)
segmentation_map.

7. `check_data_preproc.py`
- For all patients: create 2D subplots of CT, RTDOSE and segmentation_map for a number of equally-distant slices. 
The figures always contain the subplot of the first and last slice.
- For each patient: the different subplots of the same figure have the same minimum and maximum pixel value.
- Save the subplots to `figures` folder.
- Plot CT: we perform windowing by changing the `Window Width (WW) = 600` and `Window Level (WL) = 100`. Moreover, 
the CT values in the range `[WL - WW / 2, WL + WW / 2] = [-200, +400]`. 
- Plot RTDOSE: contain dose values (cGy unit), where `minimum value = rtdose_arr.min()` (usually = 0) and 
`maximum value = rtdose_arr.max()` (usually around 7000).
- Plot segmentation: we convert segmentation_map (with values 0, 1, 2, ..., `len(cfg.structures_uncompound_list)`) to 
segmentation_mask, so this is a binary plot.
- For each patient and each structure: for segmentation_map: calculate percentage of voxels that are at CT's voxel,
where CTs voxel values are less than or equal to some value (e.g. -200 HU). This is to check which structure
lies outside the actual human body. This could indicate incorrect segmentation. 
Note: `main_segmentation_outside_ct()` with `ct < max_outside_lower_limit` does not work for slices with metal artefacts.
That is, the metal artefacts changes the HU values of inside CT. As a result  many voxels will be considered 
"outside" the CT, but are actually inside the CT.


