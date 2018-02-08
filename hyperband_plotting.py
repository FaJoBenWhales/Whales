# Deep learning lab course final project.
# Kaggle whale classification.

# Plotting functions for the data created by keras_hyperband.

import sys
import os
import utilities as ut

def plot_hyperband_trajectory(path, title="Hyperband incumbent trajectory"):
    """Create the trajectory plot for the data in hyperband run in folder path."""
    filename = os.path.join(path, "incumbent_trajectory.csv")
    csv = ut.read_csv_dict(filename=filename)
    trajectory = csv['hyperband_incumbent_trajectory']
    x = list(range(len(trajectory)))
    ut.save_plot(x=x,
                 ys=dict([("incumbent", trajectory)]),
                 xlabel="iteration",
                 ylabel="validation error",
                 path=os.path.join(path, "trajectory.png"),
                 title=title)
    

def plot_learning_curves(path, title="Hyperband: evaluated learning curves"):
    """Create plot of all learning curves for the hyperband run in folder path."""
    ys = dict()
    for run_name in os.listdir(path):
        folder = os.path.join(path, run_name)
        if not os.path.isdir(folder):
            continue
        filename = os.path.join(folder, run_name + ".csv")
        csv = ut.read_csv_dict(filename=filename)
        val_error = [1.0 - float(acc) for acc in csv['val_acc']]
        ys[run_name] = val_error

    max_length = max(len(lc) for lc in ys)
    ut.save_plot(x=list(range(max_length)),
                 ys=ys,
                 xlabel="epoch",
                 ylabel="vaidation error",
                 path=os.path.join(path, "learning_curves.png"),
                 title=title,
                 legend=False)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python3 hyperband_plotting.py <path_to_hyperband_run_data>")
        exit()
    else:
        path = sys.argv[1]
        plot_hyperband_trajectory(path)
        plot_learning_curves(path)
