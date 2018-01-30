# Deep learning lab course final project.
# Kaggle whale classification.

# Hyperparameter optimization on the pretrained Keras model using hyperband.

import ConfigSpace as CS
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker

import keras_model

# Model the following hyperparameter space:
#   num_layers: number of dense classifier layers on top (1 to 4?)
#   (num_convs: number of additional convolutional layers before dense layers)
#   num_units_0 ... num_units_3: number of units by layer
#   (cnn parameters for convolutional layers)
#   base_model: InceptionV3, ...?
#   learning_rate
#   optimizer: Adam, SGD, RMSProp
#   (optimizer params (momentum etc.))
#   dropout_0 ... dropout_3: dropout probability by layer
#   (l2_reg_0 ... l2_reg_3: L2 regularization parameter by layer)
#   activation: relu, tanh
#   (learning rate schedules)
#   batch_size
#   (trainable layers vs. locked layers)
#   ... ?

# (parameters in brackets: currently not used by keras_model.py)


def create_config_space():
    cs = CS.ConfigurationSpace()
    # TODO: implement config space as described above, cf. ex. 6 of lab
    # course
    return cs


def objective_function(config, epoch=127, **kwargs):
    """Evaluate success of configuration config."""
    model = keras_model.create_pretrained_model(config)
    loss, runtime, learning_curve = keras_model.train(config)
    return loss, runtime, learning_curve


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, runtime, learning_curve = objective_function(
            cfg.get_dictionary(), epoch=int(budget))
        return {
            'loss': loss,
            'info': {"runtime": runtime,
                     "lc": learning_curve}
        }

# Below the rest of my solution to ex. 6.
# TODO: adapt to our situation.

def save_plot(x, y, label, ymax=None, path="plot.png"):
    import matplotlib.pyplot as pp
    fig = pp.figure()
    pp.plot(x, y, label=label)
    pp.ylim(ymin=0.0)
    if ymax is not None:
        pp.ylim(ymax=ymax)
    pp.legend()
    pp.savefig(path)


def save_lcs(lcs, label, path="lcs.png"):
    import matplotlib.pyplot as pp
    fig = pp.figure()
    for lc in lcs:
        pp.plot(lc)
    pp.ylim(ymin=0.0, ymax=1.0)
    pp.title(label)
    pp.savefig(path)
    

def main():
    nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

    # starting the worker in a separate thread
    w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
    w.run(background=True)

    cs = create_config_space()
    CG = hpbandster.config_generators.RandomSampling(cs)

    # instantiating Hyperband with some minimal configuration
    HB = hpbandster.HB_master.HpBandSter(
        config_generator=CG,
        run_id='0',
        eta=2,  # defines downsampling rate
        min_budget=1,  # minimum number of epochs / minimum budget
        max_budget=127,  # maximum number of epochs / maximum budget
        nameserver=nameserver,
        ns_port=ns_port,
        job_queue_sizes=(0, 1),
    )
    # runs one iteration if at least one worker is available
    res = HB.run(10, min_n_workers=1)

    # shutdown the worker and the dispatcher
    HB.shutdown(shutdown_workers=True)

    # extract incumbent trajectory and all evaluated learning curves
    traj = res.get_incumbent_trajectory()
    wall_clock_time = []
    cum_time = 0

    for c in traj["config_ids"]:
        cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
        wall_clock_time.append(cum_time)

    lc_hyperband = []
    for r in res.get_all_runs():
        c = r["config_id"]
        lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])

    incumbent_performance = traj["losses"]

    # save and plot the wall clock time and the validation of the incumbent
    # after each iteration here
    save_plot(wall_clock_time, incumbent_performance,
              label="validation of incumbent by time",
              ymax=0.08,
              path="incumbent_performance_hb.png")
    save_lcs(lc_hyperband,
             label="evaluated learning curves with HyperBand",
             path="lcs_hyperband.png")
        

