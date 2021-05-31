# Condition monitoring system using machine learning

### Content
This project examines a technical dataset using Random Forest, which is a supervised learning technique from the field of machine learning.
In addition to the numerous sensor data measured with a frequency of 1 to 100 Hz, state data are available with a sampling rate of 1/60 Hz.
Since the condition data are available as categorical features and the goal is to determine critical operating conditions in terms of CBM (Condition Based Maintenance), the present use case is classified as a classification problem.

### Data

The data set examined in this project belongs to ZeMA GmbH. which has been published freely accessible under the title "Condition monitoring of hydraulic systems Data Set".
This dataset was presented by Helwig, Pignanelli and Schütze in [1] and all credit goes to them.
	
##### Context
The data set addresses the condition assessment of a hydraulic test rig based on multi sensor data. Five fault types are superimposed with several severity grades impeding selective quantification.
	
##### Content
The data set was experimentally obtained with a hydraulic test rig. This test rig consists of a primary working and a secondary cooling-filtration circuit which are connected via the oil tank [1], [2].
The system cyclically repeats constant load cycles (duration 60 seconds) and measures process values such as pressures, volume flows and temperatures while the condition of four hydraulic components (cooler, valve, pump and accumulator) is quantitatively varied.

For more information on this dataset, please visit the original source at UCI:
https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems

##### Relevant Papers
[1] Nikolai Helwig, Eliseo Pignanelli, Andreas Schütze, "Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics",
in Proc. I2MTC-2015 - 2015 IEEE International Instrumentation and Measurement Technology Conference, paper PPS1-39, Pisa, Italy, May 11-14, 2015,
doi: 10.1109/I2MTC.2015.7151267.

[2] N. Helwig, A. Schütze, "Detecting and compensating sensor faults in a hydraulic condition monitoring system",
in Proc. SENSOR 2015 - 17th International Conference on Sensors and Measurement Technology, oral presentation D8.1, Nuremberg, Germany, May 19-21, 2015,
doi: 10.5162/sensor2015/D8.1.

[3] Tizian Schneider, Nikolai Helwig, Andreas Schütze, "Automatic feature extraction and selection for classification of cyclical time series data",
tm - Technisches Messen (2017), 84(3), 198—206, doi: 10.1515/teme-2016-0072.

Creator: ZeMA gGmbH, Eschberger Weg 46, 66121 Saarbrücken
Contact: t.schneider '@' zema.de, s.klein '@' zema.de, m.bastuck '@' lmt.uni-saarland.de, info '@' lmt.uni-saarland.de

### Instruction and scripts

Just download scripts and xlsx file as zip file, unzip it and run the code.

Scripts:
- main.py: main script
- myfunctions.py: used functions

.xlsx-file:
- attribute_information.xlsx: To divide the data sets into input and state data, an .xlsx file has been created.
The 1st worksheet 'Sensor Data' contains a list of all sensors including units and sampling rate.
The other worksheets contain the most important information of the different state data and are used for Data Labeling.
Hereby has been defined, which data are critical, ALARP (as low as reasonably practicable) and non-critical.
In a first calculation, each data point defined as critical was assigned the label "bad". Accordingly, state data defined as ALARP or non-critical were assigned the label "good".
For further information: see code section Data Labeling.

### Code sections
- Import modules
- Variables
- Folder Architecture and Data Akquise: Creation of folders and data download from URL (if not already available)
  - 01_information: All information related to the dataset. This information is only used to get to know the dataset better and will not be used further.
  - 02_data: Examined records that are imported and analysed.
  - 03_export: Destination folder where plots and results are exported. For this purpose, a folder is created during the calculation. The folder name corresponds to the
    running datetime. This folder in turn contains a sub-folder with data plots and a sub-folder with test results.
  	- YYYY-MM-DD (01_data_plotting, 02_results)
- Export settings: Creating Folder for export files
- Export variable configuration: Export of variable settings as xlsx.file
- Data import:
  - Raw Data --> import_data (key 'Output' contains state data)
  - Data Properties (attribute_information.xlsx) --> data_properties
- Preprocessing:
The data were measured by sensors with different frequencies (1 - 100 Hz). The state data are available with a frequency of 1/60 Hz.
To obtain an nxm matrix and to take into account the sampling time of each data point, each data set is transposed and transformed according to the frequency.
Furthermore, the data sets are smoothed using moving average. Depending on the sensor, different window sizes are used.
  - Raw Data --> raw_data
  - Smoothed Data --> preproc_data 
- Data Labeling:
In a first calculation, each data point defined as critical was assigned the label "bad". Accordingly, state data defined as ALARP or non-critical were assigned the label "good".
If data points that were defined as ALARP should also be defined as critical: Simply add 'ALARP' to the list in the variable vector (key: 'critical').
- Data Plotting: Plotting and export of raw and smoothed Data
- Missing Data:
Since machine learning algorithms do not support data with missing values (NaN), they must be handled.
  - Sensor data: For NaN data located between 2 data points, a value is assigned using linear interpolation.
  - State data: In contrast to the measured values, the state data are categorical data. Under the assumption that the operating state does not change between 2 existing data points, each NaN value is assigned the previous value.
  - However, both methods do not work for NaN values before the first data point. Due to the small number of lines, these first rows are removed (approx. 0.15% of 13 million)
- Train-Test-Split:
In order to obtain a statistically valid result, the data is divided into training and test data using cross-validation.
To keep the ratio of critical and non-critical data constant in each iteration, stratified k-fold cross-validation is used as method.
- Training & Test:
Random Forest is used as algorithm.
The maximum depth of each tree is 5, the maximum number of trees is 10 and the splitting criterion is Gini Impurity.
- Results:
Plotting and exporting different metrics

###### Background: This project served me to familiarize with Python and its various modules. Since my previous Data Science projects were programmed with Matlab, I use Spyder as IDE.
