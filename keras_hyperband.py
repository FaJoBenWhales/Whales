# Deep learning lab course final project.
# Kaggle whale classification.

# Hyperparameter optimization on the pretrained Keras model using hyperband.

import ConfigSpace as CS
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker

import keras_model



def configuration_space_from_raw(hpRaw, hpRawConditions):
    cs = CS.ConfigurationSpace()
    # add hyperparameters
    for hp in hpRaw:
        if hp[4] == "float":
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    hp[0],
                    lower=hp[1][0],
                    upper=hp[1][1],
                    default_value=hp[2],
                    log=hp[3]
                )
            )
        elif hp[4] == "int":
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    hp[0],
                    lower=hp[1][0],
                    upper=hp[1][1],
                    default_value=hp[2],
                    log=hp[3]
                )
            )
        elif (hp[4] == "cat"):
            cs.add_hyperparameter(
                CS.CategoricalHyperparameter(
                    hp[0],
                    hp[1]
                )
            )
        else:
            raise Exception("unknown hp type in hpRawList")

    # add conditions
    for cond in hpRawConditions:
        if cond[1] == "eq":
            cs.add_condition(
                CS.EqualsCondition(
                    cs.get_hyperparameter(cond[0]),
                    cs.get_hyperparameter(cond[2]),
                    cond[3]
                )
            )
        elif cond[1] == "geq":
            cs.add_condition(
                CS.GreaterThanCondition(
                    cs.get_hyperparameter(cond[0]),
                    cs.get_hyperparameter(cond[2]),
                    cond[3]
                )
            )
        else:
            raise Exception("unknown condition type in hpRawConditions")
        
    return cs


# Model the following hyperparameter space:
#   base_model: InceptionV3, ...?
#   num_dense_layers: number of dense classifier layers on top (1 to 4?)
#   num_dense_units_0 ... num_dense_units_3: number of units per dense layer
#   activation: relu, tanh
#   dropout_0 ... dropout_3: dropout probability by layer
#   cnn_unlock_epoch: when to unlock parts of the cnn
#   cnn_num_unlocked: number of pretrained layers to unlock
#   optimizer: Adam, SGD, RMSProp
#   learning_rate
#   batch_size
def create_config_space():
    hpRaw = [
        #<    name              >   <  Range       >      <Default>     <Log>   <Type>
        ["base_model",              ["InceptionV3"],    "InceptionV3",  None,   "cat"],
        ["num_dense_layers",        [1, 4],                 2,          False,  "int"],
        ["num_dense_units_0",       [50, 4000],             1024,       False,  "int"],
        ["num_dense_units_1",       [50, 4000],             1024,       False,  "int"],
        ["num_dense_units_2",       [50, 4000],             1024,       False,  "int"],
        ["num_dense_units_3",       [50, 4000],             1024,       False,  "int"],
        ["activation",              ["relu", "tanh"],       "relu",     None,   "cat"],
        ["dropout"],                [True, False],          False,      None,   "cat"],
        ["dropout_0",               [0.0, 1.0],             0.5,        False,  "float"],
        ["dropout_1",               [0.0, 1.0],             0.5,        False,  "float"],
        ["dropout_2",               [0.0, 1.0],             0.5,        False,  "float"],
        ["dropout_3",               [0.0, 1.0],             0.5,        False,  "float"],
        ["optimizer",               ["Adam", "SGD", 
                                     "RMSProp"],            "SGD",      None,   "cat"],
        ["learning_rate",           [0.00001, 0.1],         0.001,      True,   "float"],
        ["cnn_unlock_epoch",        [0, 1000],              200,        False,  "int"],
        ["cnn_num_unlock",          [0, 63],                 0,         False,  "int"],
        ["batch_size",              [16, 64],               32,         True,   "int"],
    ]
    hpRawConditions = [
        #< conditional hp name      >   <cond. Type>    <cond. variable>        <cond. value>
        ["num_dense_units_1",           "geq",          "num_dense_layers",     2],
        ["num_dense_units_2",           "geq",          "num_dense_layers",     3],
        ["num_dense_units_3",           "eq",           "num_dense_layers",     4],
        ["dropout_0",                   "eq",           "dropout",              True],
        ["dropout_1",                   "eq",           "dropout",              True],
        ["dropout_2",                   "eq",           "dropout",              True],
        ["dropout_3",                   "eq",           "dropout",              True],
        ["dropout_1",                   "geq",          "num_dense_layers",     2],
        ["dropout_2",                   "geq",          "num_dense_layers",     3],
        ["dropout_3",                   "eq",           "num_dense_layers",     4],
    ]
    return configuration_space_from_raw(hpRaw, hpRawConditions)


def objective_function(config, epoch=127, **kwargs):
    """Evaluate success of configuration config."""
    model = keras_model.create_pretrained_model(config)
    loss, runtime, learning_curve = keras_model.train(model, config)
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
        

