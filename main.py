
############################################################################

# Data analysis of "Predictive Maintenance Of Hydraulics System"
# Structure:
# --> IMPORT MODULES
#       - Clear console
#       - Import modules
#       - Start running time
# --> VARIABLES
# --> FOLDER ARCHITECTURE AND DATA AKQUISE
# --> EXPORT SETTINGS
# --> EXPORT VARIABLE CONFIGURATION
# --> DATA IMPORT
#       - Listing all files
#       - Import attribute informations
#       - Import raw data into dictionary
# --> DATA ANALYSIS
# --> PREPROCESSING
# --> DATA LABELING
# --> DATA PLOTTING
# --> MISSING DATA
# --> TRAIN-TEST-SPLIT
# --> TRAINING & TEST
# --> RESULTS
#       - ROC curve
#       - Feature Importances in random forest
#       - Confusion Matrices of each iteration
# --> END OF SIMULATION

############################################################################

#######################
### IMPORT MODULES
#######################
print("IMPORT MODULES")

# Clear variable explorer and console
from IPython import get_ipython
get_ipython().magic("reset -sf")
get_ipython().magic("clear")

# System modules
import os
import shutil
# import sys # Add sys.exit() to stop the run
import copy
from datetime import datetime
import time
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

# Data Science modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# My functions
import helperFunctions 

############################################################################

#######################
### VARIABLES
#######################
print("VARIABLES")

# Start running time
start_time = time.time()

# Variables
variables = {
    "window_size": {
        "Cooling efficiency (virtual)" : 100,
        "Pressure, Sensor 1" : 5000,
        "Pressure, Sensor 2" : 5000,
        "Pressure, Sensor 3" : 1000,
        "Pressure, Sensor 4" : 500,
        "Pressure, Sensor 5" : 100,
        "Pressure, Sensor 6" : 100,
        "Motor power" : 5000,
        "Volume flow, Sensor 1" : 2000,
        "Volume flow, Sensor 2" : 100,
        "Temperature, Sensor 1" : 100,
        "Temperature, Sensor 2" : 100,
        "Temperature, Sensor 3" : 100,
        "Temperature, Sensor 4" : 100,
        "Vibration" : 100,
        "Cooling efficiency (virtual)" : 100,
        "Cooling power (virtual)" : 100,
        "Efficiency factor" : 100
        }, # Window size for moving average
    "critical": ["NOT ACCEPTABLE"], # Possible critical conditions: "NOT ACCEPTABLE", "ALARP" *
    "kFold_splits": 5, # Number of folds
    "clf_max_depth": 5, # Maximum depth of the tree
    "clf_n_estimators": 10, # Number of trees in the forest
    "clf_criterion": "gini", # Function (Criterion) to measure the quality of a split
    "targets": {
        "Cooler condition" : True,
        "Valve condition" : False,
        "Internal pump leakage" : False,
        "Hydraulic accumulator" : False,
        "stable flag" : False
        }  # Inspected targets **
    }

# *  ALARP = as low as reasonably practicable
#
# ** Currently only one label can be examined.
#    For this reason all but one should be defined as false.
#    In the future, the number of targets should be freely selectable.

############################################################################

#############################################
### FOLDER ARCHITECTURE AND DATA AKQUISE
#############################################

# Create folder architecture, if not existent existent
folder_list = os.listdir('.')
if "01_information" not in folder_list:
    print("FOLDER ARCHITECTURE AND DATA AKQUISE")
    os.mkdir("01_information")
if "02_data" not in folder_list:
    os.mkdir("02_data")
if "03_exports" not in folder_list:
    os.mkdir("03_exports")

# Download Data, if not existent existent
_, _, filenames1 = next(os.walk("01_information"))
_, _, filenames2 = next(os.walk("02_data"))

if len(filenames1) == 0 or len(filenames1) == 0:
    # Download data (as .zip-file)
    zipurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip"
    zipresp = urlopen(zipurl)
    data_zipfile = ZipFile(BytesIO(zipresp.read()))
    
    # filenames of .zip-file
    filenames = data_zipfile.namelist()
    infodata_list = ['description.txt','documentation.txt']
    sensordata_list = copy.deepcopy(filenames)
    for i in range(len(infodata_list)):
        sensordata_list.remove(infodata_list[i])
    
    # Exctract Data Sets into folder
    for i in range(len(infodata_list)):
        data_zipfile.extract(infodata_list[i],"01_information")
    for i in range(len(sensordata_list)):
        data_zipfile.extract(sensordata_list[i],"02_data")

############################################################################

##############################
### EXPORT SETTINGS
##############################
print("EXPORT SETTINGS")

dirpath = "03_exports"
_, sub_folder, filenames = next(os.walk(dirpath)) # filename
time_now = datetime.now().strftime("%Y-%m-%d") # Today's date in YYYY-MM-DD
dirpath = dirpath + "/" + time_now # directory path

# If folder 'YYYY-MM-DD' (containing export files) is already existent, it will be removed
if time_now in sub_folder:
    shutil.rmtree(dirpath)

# Creating empty folder architecture
os.mkdir(dirpath)
os.mkdir(dirpath + "/01_data_plotting")
os.mkdir(dirpath + "/02_results")

# Set the chunk size (Plotting configuration)
plt.rcParams['agg.path.chunksize'] = 10000

############################################################################

##############################################
### EXPORT VARIABLE CONFIGURATION
##############################################
print("EXPORT VARIABLE CONFIGURATION")

# Create Export file
directory = dirpath + "/"
filename = "variables configuration"
file_format = ".xlsx"
csv_name = directory + filename + file_format
writer = pd.ExcelWriter(csv_name)

# Preprocessing
data = variables["window_size"]
df = pd.DataFrame.from_dict(
    data,
    orient = "index",
    columns = ["Window Size"]
    )
df.to_excel(
    writer,
    sheet_name = "Preprocessing"
    )

# Data Labelling
data = {
        "critical" : variables["critical"]
        }
df = pd.DataFrame.from_dict(
    data,
    orient = "index"
    )
df.to_excel(
    writer,
    sheet_name = "Data Labelling"
    )

# Train-Test-Split
data = {
        "Strategie" : "Stratified kFold Cross Validation",
        "Number of Folds" : variables["kFold_splits"]
        }
df = pd.DataFrame.from_dict(
    data,
    orient = "index"
    )
df.to_excel(
    writer,
    sheet_name = "Train-Test-Split"
    )

# Training
data = {
        "Model" : "Random Forest",
        "Maximum depth of the tree" : variables["clf_max_depth"],
        "Maximum number of trees" : variables["clf_n_estimators"],
        "Splitting criterion" : variables["clf_criterion"]
        }
df = pd.DataFrame.from_dict(
    data,
    orient = "index",
    columns = ["Value"]
    )
df.to_excel(
    writer,
    sheet_name = "Training"
    )

# Test
data = variables["targets"]
df = pd.DataFrame.from_dict(
    data,
    orient = "index",
    columns = ["Inspected?"]
    )
df.to_excel(
    writer,
    sheet_name = "Test"
    )

# Data Labelling
writer.save()

############################################################################

#######################
### DATA IMPORT
#######################
print("DATA IMPORT")

# List all files of folder "import_dirpath" into "filenames"
import_dirpath = "02_data"
_, _, filenames = next(os.walk(import_dirpath))
filenames = sorted(
    filenames,
    key = str.lower
    )
filenames = helperFunctions.listallwith(
    filenames,
    [".txt", ".csv"]
    )

# Import data properties
data_properties = pd.read_excel(
    "attribute_information.xlsx",
    sheet_name = None,
    header = 0
    )

#### Import raw data ####
# 1. Import raw data (.txt-files) into DataFrame
# 2. Insert raw data and associated characteristics into dictionary "dict_sel"
# 3. Insert (temporarily created) dictionary "dict_sel" 
#    into final raw data dictionary "import_data"
import_data = {}
col = data_properties["Sensor Data"]
for i in range(len(filenames)):
    sensor = filenames[i].strip(".txt")
    df = pd.read_csv(
        import_dirpath + "/" + filenames[i],
        index_col = None,
        header = None,
        delimiter = "\t"
        )
    if sensor != "profile":
        idx = col[col["Sensor"] == sensor].index.values[0]
        key = col.iloc[idx]["Physical quantity"]
        dict_sel = {
            key : [df,col.iloc[idx]]
            }
        import_data.update(dict_sel)
    else:
        key = "Output"
        output_columns = list(data_properties.keys())
        output_columns.remove("Sensor Data")
        df.columns = output_columns
        dict_sel = copy.deepcopy(data_properties)
        del dict_sel["Sensor Data"]
        dict_sel = {
            key: [df, dict_sel]
            }
        import_data.update(dict_sel)

############################################################################

#######################
### DATA ANALYSIS
#######################
print("DATA ANALYSIS")

# Statistical description
description_data = {}
for i in range(len(import_data)):
    sensor = list(import_data.keys())[i]
    df = import_data[sensor][0]
    attribute = import_data[sensor][1]
    if sensor != "Output":
        count_isnull = df.isnull().values.sum()
        minimum = df.min().values.min()
        maximum = df.max().values.max()
        dict_sel = {
            key: [df,col.iloc[idx]]
            }
        description_data.update(dict_sel)

############################################################################

#######################
### PREPROCESSING
#######################
print("PREPROCESSING")

# Determination of ...
# --> maximum number of cycles
# --> maximum frequency
count_cycles = 0 # maximum number of cycles
max_frequency = 0 # maximum frequency
for i in range(len(import_data)):
    sensor = list(import_data.keys())[i]
    if sensor != "Output":
        cycles_dummy = import_data[sensor][0].shape[0]
        if cycles_dummy > count_cycles:
            count_cycles = cycles_dummy
        frequency = int(import_data[sensor][0].shape[1] / 60)
        if frequency > max_frequency:
            max_frequency = frequency

starting_time = 1/(60*max_frequency) # time of the first measurement data point
count_points = count_cycles * max_frequency * 60 # Number of rows in transposed DataFrames
time_vector = pd.Series(np.linspace(starting_time,count_cycles,count_points))
raw_data = pd.DataFrame({
    "time":time_vector
    }) # Raw Data
preproc_data = pd.DataFrame({
    "time":time_vector
    }) # Preprocessed Data

# 1. Transposing & preprocessing sensor data.
# 2. Import raw data into "raw_data" and preprocessed data into "preproc_data".
for i in range(len(import_data)):
    sensor = list(import_data.keys())[i]
    df = import_data[sensor][0]
    if sensor != "Output":
        frequency = int(len(df.columns) / 60)
        if frequency == max_frequency:
            df_transposed = pd.DataFrame(
                df.values.ravel(), columns = [sensor]
                ) # Transposing Data
            
            raw_data[sensor] = df_transposed[sensor] # Import into Raw DataFrame
            preproc_data[sensor] = df_transposed[sensor].rolling(
                window = variables['window_size'][sensor]
                ).mean() # Import into Preprocessed DataFrame
        else:
            raw_data_transposed = pd.DataFrame({
                sensor:df.values.ravel()
                }) # Transposing Raw Data
            raw_data_array = np.empty((1,len(time_vector)))
            raw_data_array[:] = np.NaN
            
            mov_averaged_data_transposed = raw_data_transposed.rolling(
                window = variables["window_size"][sensor]
                ).mean() # Moving average of Raw Data
            mov_averaged_data_array = np.empty((1,len(time_vector)))
            mov_averaged_data_array[:] = np.NaN
            
            for ii in range(len(raw_data_transposed)):
                row_index = int(
                    (max_frequency/frequency * (ii+1))-1
                    )
                raw_data_array[0][row_index] = raw_data_transposed[sensor][ii]
                mov_averaged_data_array[0][row_index] = mov_averaged_data_transposed[sensor][ii]
            
            raw_data[sensor] = raw_data_array[0] # Import into Raw DataFrame
            preproc_data[sensor] = mov_averaged_data_array[0] # Import into Preprocessed DataFrame

############################################################################

#######################
### DATA LABELING
#######################
print("DATA LABELING")

sensor = "Output"
label_properties = import_data[sensor][1]
label_raw_data = import_data[sensor][0]
frequency = 1 / 60
for i in range(len(label_properties)):
    label_proc_data = np.empty(
        len(label_raw_data),
        dtype = "object"
        ) # processed label vector
    label_proc_data[:] = "good"
    
    label_name = list(label_properties.keys())[i]
    label_df = label_properties[label_name]
    idx_lp = label_df.Criticality.apply(
        lambda x: any(item for item in variables["critical"] if item in x)
        ) # Logical vector of selected rows
    critical_values = list(label_df[idx_lp].iloc[:,0])
    row_index = label_raw_data[label_name].isin(critical_values)
    label_proc_data[row_index] = "bad"
    label_name = label_name.replace(" ","_") + "_label"
    
    output_vector = [""] * len(preproc_data) # Output Vector
    for ii in range(len(label_proc_data)):
        idx = int(
            (max_frequency / frequency * (ii + 1) ) - 1
            )
        output_vector[idx] = label_proc_data[ii]
    preproc_data[label_name] = output_vector
    
############################################################################

########################
### DATA PLOTTING
########################
print("DATA PLOTTING")

# Plotting raw and preprocessed data
x = preproc_data["time"]
sensor_list = data_properties["Sensor Data"]["Physical quantity"] # Sensor names
for i in range(len(sensor_list)):
    sensor = list(import_data.keys())[i]
    properties_selected = import_data[sensor][1]
    if sensor != "Output":
        # Creating Figure
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        # Plotting
        y = raw_data[sensor]
        y_averaged = preproc_data[sensor]
        ax.plot(
            x, y,
            alpha = 0.25,
            marker = ".",
            markersize = 0.1,
            markerfacecolor = "blue",
            markeredgecolor = "blue",
            label = "Raw data"
            )
        idx = np.isfinite(y_averaged) # logical vector to find not-NaN values
        ax.plot(
            x[idx], y_averaged[idx],
            linestyle = "-",
            linewidth = 1.0,
            color = "red",
            label = "Mov. average [size=" + str(variables["window_size"][sensor]) + "]"
            )
        
        # Settings
        ax.set_xlim(0,max(x))
        ax.xaxis.set_tick_params(labelsize = 8)
        ax.yaxis.set_tick_params(labelsize = 8)
        ax.set_xlabel(
            "Cycles (" + properties_selected["Sampling rate"] + ")",
            fontsize = 8
            )
        ax.set_ylabel(
            sensor + " [" + properties_selected.Unit + "]",
            fontsize = 8
            )
        ax.set_xticks(
            np.linspace(0, max(x), 6).round()
            )
        ax.set_yticks(
            np.linspace(0, helperFunctions.roundup(max(y[idx])), 6).round()
            )
        ax.tick_params(axis='x', labelsize = 8)
        ax.tick_params(axis='y', labelsize = 8)
        ax.grid()
        ax.legend(fontsize = 6)
        
        # Export
        directory = dirpath + "/01_data_plotting/"
        filename = sensor
        file_format = ".jpg"
        f.savefig(
            directory + filename + file_format,
            bbox_inches = "tight",
            dpi = 500
            )

############################################################################
        
#######################
### MISSING DATA
#######################
print("MISSING DATA")

# Linear Interpolation
feature_names = [s for s in preproc_data.columns if "label" not in s]
feature_names = [s for s in feature_names if "time" not in s]
for i in range(len(feature_names)):
    sensor = feature_names[i]
    preproc_data[sensor] = preproc_data[sensor].interpolate(method="linear")

# Filling in labelling columns
label_list = [i for i in list(preproc_data.columns) if "label" in i]
for i in range(len(label_list)):
    label = label_list[i]
    preproc_data[label][preproc_data[label]==""] = np.NaN
    preproc_data[label].fillna(method="bfill", inplace=True) # replace every NaN with the first non-NaN value in the same column below it

# Drop all rows that have any NaN values
rows_deleted = len(preproc_data) - len(preproc_data.dropna())
rows_deleted_pct = rows_deleted/len(preproc_data)*100
preproc_data = preproc_data.dropna()

############################################################################

##########################
### TRAIN-TEST-SPLIT
##########################
print("TRAIN-TEST-SPLIT")

X = preproc_data[feature_names]

y_name_list = helperFunctions.find_true_keys(variables["targets"])
y_name = y_name_list[0].replace(' ','_') + '_label'
y = preproc_data[y_name]

# y = preproc_data[label_list[0]]

# Label balance
label_stats = pd.DataFrame(columns=label_list)
pct_list = np.array([])
for i in range(len(label_list)):
    label = label_list[i]
    pct = ((preproc_data[label] == "bad").sum() / (preproc_data[label] == "good").sum()) * 100
    pct_list = np.append(pct_list, pct)
label_stats.loc[0] = pct_list

#### Stratified K-Folds cross-validation
# Provides train/test indices to split data in train/test sets.
# This cross-validation object is a variation of KFold that returns stratified folds.
# The folds are made by preserving the percentage of samples for each class.
kf = StratifiedKFold(
    n_splits = variables["kFold_splits"],
    shuffle=True,
    random_state=0
    )

train_balance = np.array([])
test_balance = np.array([])
for train_index, test_index in kf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    a = ((y_train == "bad").sum() / (y_train == "good").sum()) * 100
    train_balance = np.append(train_balance, a)
    b = ((y_test == "bad").sum() / (y_test == "good").sum()) * 100
    test_balance = np.append(test_balance, b)

############################################################################
        
#########################
### TRAINING & TEST
#########################
print("TRAINING & TEST")

# Create model
clf = RandomForestClassifier(
    n_estimators = variables["clf_n_estimators"],
    criterion = variables["clf_criterion"],
    max_depth = variables["clf_max_depth"],
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    max_features = "auto",
    max_leaf_nodes = None,
    min_impurity_decrease = 0.0,
    min_impurity_split = None,
    bootstrap = True,
    oob_score = False,
    n_jobs = None,
    random_state = None,
    verbose = 0,
    warm_start = False,
    class_weight = None,
    ccp_alpha = 0.0,
    max_samples = None
    )

# Predict...
# --> ROC curve
# --> AUC score
# --> Confusion Matrices (for each Decision Tree of Random Forest)
def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict,pos_label="good")
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

results = pd.DataFrame(columns=["training_score", "test_score"])
fprs, tprs, scores = [], [], []
conf_matrix_list_of_arrays = [] 

for (train, test), i in zip(kf.split(X, y), range(5)):
    clf.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(train)
    fpr, tpr, auc_score = compute_roc_auc(test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    conf_matrix_list_of_arrays .append(conf_matrix)
 
############################################################################

#######################
### RESULTS
#######################
print("RESULTS")

# ROC curve
fig,ax = helperFunctions.plot_roc_curve(fprs, tprs)
directory = dirpath + "/02_results/"
filename = "ROC" + "_" + y.name
file_format = ".jpg"
fig.savefig(
    directory + filename + file_format,
    bbox_inches = "tight",
    dpi=800
    )

###########################
# Feature Importances in random forest

std = np.std(
    [tree.feature_importances_ for tree in clf.estimators_],
    axis=0
    )
forest_importances = pd.Series(clf.feature_importances_, index = feature_names)

# Figure
fig, ax = plt.subplots(figsize = (15,7.5))
forest_importances.plot.bar(yerr=std, ax = ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# Export
directory = dirpath + "/02_results/"
filename = "Feature_Importances_" + y.name
file_format = ".jpg"
fig.savefig(
    directory + filename + file_format,
    dpi = 800
    )

 
###############################################
# Confusion Matrices of each iteration
os.mkdir(dirpath + "/02_results/confusion_matrices") # Creating Folder

for i in range(len(conf_matrix_list_of_arrays)):
    cm = conf_matrix_list_of_arrays[i]
    
    # Figure
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels = clf.classes_
        )
    disp.plot(cmap = "Blues")
    fig = disp.figure_
    fig.suptitle(
        y.name.replace("_label","").replace("_"," ") + "\n" +
        "Confusion Matrix " + str(i+1),
        fontsize = "10"
        )
    
    # Export
    directory = dirpath + "/02_results/confusion_matrices/"
    filename = "Conf_Mat_" + str(i+1) + "_" + y.name
    file_format = ".jpg"
    fig.savefig(
        directory + filename + file_format,
        bbox_inches = "tight",
        dpi = 800
        )

############################################################################

#######################
### END OF SIMULATION
#######################

# Running time
end_time = time.time() - start_time
print("--- Running Time: %s ---" % str(pd.Timedelta(seconds = end_time)))
