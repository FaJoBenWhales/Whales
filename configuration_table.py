# Deep learning lab course final project.
# Kaggle whale classification.

# Create a big (csv) table for analysing simple parameter spaces with
# linear regreesion models.


import os
import sys
import utilities as ut


def create_table(path, min_epochs=0, save_to_file=None):
    """Create a dict / table with hyperparameter settings and final
    validation accuracy. Includes only runs with >= min_epochs
    epochs. Saves csv save_to_file, if specified."""
    
    lines = []
    for run_name in os.listdir(path):
        folder = os.path.join(path, run_name)
        if not os.path.isdir(folder):
            continue
        metric_file = os.path.join(folder, run_name + ".csv")
        config_file = os.path.join(folder, "config.txt")
        metrics = ut.read_csv_dict(filename=metric_file)
        final_val_error = 1.0 - float(metrics['val_acc'][-1])
        epochs = len(metrics['val_acc'])
        with open(config_file, "r") as f:
            config_string = f.readlines()[1]
        config_dict = eval(config_string)
        lines.append((run_name, config_dict, final_val_error, epochs))

    # create csv dict out of (run_name, config_dict) list
    table = dict()
    keys = list(sorted(lines[0][1].keys()))
    for key in keys:
        table[key] = []
    table['final_val_error'] = []
    table['run_name'] = []
    table['epochs'] = []
    for line in lines:
        if not line[3] >= min_epochs:
            continue
        table['run_name'].append(line[0])
        table['final_val_error'].append(line[2])
        table['epochs'].append(line[3])
        for key in keys:
            table[key].append(line[1][key])

    if save_to_file is not None:
        ut.write_csv_dict(table, filename=os.path.join(path, save_to_file))
            
    return table


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python3 configuration_table.py " +
              "<hyperband_directory> <min_epochs>.")
    else:
        min_epochs = 0 if len(sys.argv) == 2 else int(sys.argv[2])
        create_table(sys.argv[1], min_epochs=min_epochs, save_to_file="table.csv")
    
        

