

# # Base RADICAL directory
# RADICAL_data_path = r'D:\OneDrive\Northwestern University\DARPA RADICAL - RADICAL Documents'

# Specimen, experiment
default_experiment = r'(250924) Creep II + Static'

# Run function on some excel file
excel_file_path = r"V:\Data\Specimen Directory\ALL RAD-55 Samples_code_testing.xlsx"
# process_dimple_locations(excel_file_path)
default_specimen = r'RAD-BD-006'

# pressure threshold in psi
pressure_threshold = 100

# check transducer health
check_pressure_transducer_health(r"Data\DABI\HT-DABI\(251020) pressure transducer test\test.csv")

# merge DABI data
# merge_DABI_data(default_specimen, experiment=default_experiment, RADICAL_data_path="V:\\")
# ident_pressure_steps(default_specimen, experiment=default_experiment, thresh=pressure_threshold, RADICAL_data_path=r"V:\\")