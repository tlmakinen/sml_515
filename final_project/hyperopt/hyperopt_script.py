"""Start a hyperoptimization from a single node"""
import sys
import numpy as np
import pickle as pkl
import hyperopt
from hyperopt import hp, fmin, tpe, Trials

from skim import main


#Change the following code to your file
################################################################################
# TODO: Declare a folder to hold all trials objects
TRIALS_FOLDER = 'trials'
NUMBER_TRIALS_PER_RUN = 1

def run_trial(args):
    """Evaluate the model loss using the hyperparams in args
    :args: A dictionary containing all hyperparameters
    :returns: Dict with status and loss from cross-validation
    """
    loss = main(args)


    return {
        'status': 'ok', # or 'fail' if nan loss
        'loss': loss
    }


#TODO: Declare your hyperparameter priors here:
space = {
    'alpha1': hp.loguniform('alpha1', np.log(0.1), np.log(75)),
    'beta1': hp.loguniform('beta1', np.log(0.1), np.log(30)),
    'alpha2': hp.loguniform('alpha2', np.log(0.1), np.log(75)),
    'beta2': hp.loguniform('beta2', np.log(0.1), np.log(30)),
    'alpha3': hp.loguniform('alpha3', np.log(0.1), np.log(75)),
    'c': hp.loguniform('c', np.log(0.1), np.log(30)),
    'alpha_obs': hp.loguniform('alpha_obs', np.log(0.1), np.log(75)),
    'beta_obs': hp.loguniform('beta_obs', np.log(0.1), np.log(30))
}

################################################################################



def merge_trials(trials1, trials2_slice):
    """Merge two hyperopt trials objects
    :trials1: The primary trials object
    :trials2_slice: A slice of the trials object to be merged,
        obtained with, e.g., trials2.trials[:10]
    :returns: The merged trials object
    """
    max_tid = 0
    if len(trials1.trials) > 0:
        max_tid = max([trial['tid'] for trial in trials1.trials])

    for trial in trials2_slice:
        tid = trial['tid'] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial
        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials1.insert_trial_docs(hyperopt_trial) 
        trials1.refresh()
    return trials1

loaded_fnames = []
trials = None
# Run new hyperparameter trials until killed
while True:
    np.random.seed()

    # Load up all runs:
    import glob
    path = TRIALS_FOLDER + '/*.pkl'
    for fname in glob.glob(path):
        if fname in loaded_fnames:
            continue

        trials_obj = pkl.load(open(fname, 'rb'))
        n_trials = trials_obj['n']
        trials_obj = trials_obj['trials']
        if len(loaded_fnames) == 0: 
            trials = trials_obj
        else:
            print("Merging trials")
            trials = merge_trials(trials, trials_obj.trials[-n_trials:])

        loaded_fnames.append(fname)

    print("Loaded trials", len(loaded_fnames))
    if len(loaded_fnames) == 0:
        trials = Trials()

    n = NUMBER_TRIALS_PER_RUN
    try:
        best = fmin(run_trial,
            space=space,
            algo=tpe.suggest,
            max_evals=n + len(trials.trials),
            trials=trials,
            verbose=1,
            rstate=np.random.RandomState(np.random.randint(1,10**6))
            )
    except hyperopt.exceptions.AllTrialsFailed:
        continue

    print('current best', best)
    hyperopt_trial = Trials()

    # Merge with empty trials dataset:
    save_trials = merge_trials(hyperopt_trial, trials.trials[-n:])
    new_fname = TRIALS_FOLDER + '/' + str(np.random.randint(0, sys.maxsize)) + '.pkl'
    pkl.dump({'trials': save_trials, 'n': n}, open(new_fname, 'wb'))
    loaded_fnames.append(new_fname)