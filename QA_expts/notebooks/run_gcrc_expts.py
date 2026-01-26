def main():
    print("Hello from conformal-safety!")


if __name__ == "__main__":
    main()

    # %load_ext autoreload
    # %autoreload 2
    # %matplotlib inline
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    import pickle
    import json
    import os
    import sys
    
    from tqdm import tqdm
    src_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'src')
    sys.path.append(src_dir)
    
    import traceback

    def remove_specific_leading_chars(input_string):
        import re
        # Remove leading commas
        input_string = re.sub(r'^,+', '', input_string)
        # Remove numbers followed by a comma
        return re.sub(r'^\d+,+', '', input_string)

    ## LTT util functions
    import numpy as np
    from scipy.stats import binom, norm
    from scipy.optimize import brentq
    # from confseq import betting
    import pdb
    
    def h1(y, mu):
        return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))
    
    def h2(y):
        return (1+y)*np.log(1+y) - y
    
    ### Log tail inequalities of mean
    def hoeffding_plus(mu, x, n):
        return -n * h1(np.maximum(mu,x),mu)
    
    def hoeffding_minus(mu, x, n):
        return -n * h1(np.minimum(mu,x),mu)
    
    def bentkus_plus(mu, x, n):
        return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1
    
    def bentkus_minus(mu, x, n):
        return np.log(max(binom.cdf(np.ceil(n*x),n,mu),1e-10))+1
    
    def binom_p_value(r_hat,n,alpha):
        return binom.cdf(np.ceil(n*r_hat),n,alpha)
    
    def hb_p_value(r_hat,n,alpha):
        bentkus_p_value = np.e * binom.cdf(np.ceil(n*r_hat),n,alpha)
        def h1(y,mu):
            with np.errstate(divide='ignore'):
                return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
        hoeffding_p_value = np.exp(-n*h1(min(r_hat,alpha),alpha))
        return min(bentkus_p_value,hoeffding_p_value)


    from typing import List
    
    
    def score_func(
        claim_scores : List[np.ndarray],
        annotations : List[np.ndarray],
        method : str = "max",
        epsilon : float = 0.1, ## if method == "fraction", largest fraction of allowable unfactual claims
        tau : float = 0.1 ## if method == "fraction_nonmonotone", pre-selected threshold
    ):
        if method == "max":
            min_score = -1
            scores = np.zeros((len(claim_scores),))
            for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
                scores[i] = np.max(cs[~a]) if np.sum(~a) >= 1 else min_score
        if isinstance(method, int):
            ## Think that 'method' here is \lambda in the Cherian et al paper (but tau in that paper is the lambda in loss calibration)
            min_score = -1
            scores = np.zeros((len(claim_scores),))
            for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
                ## if (number of untrue claims) > method:
                    ## scores[i] = (largest sub-claim score of any False sub-claim)
                ## else:
                    ## scores[i] = min_score
                # print((cs, a))
                # if np.sum(~a) > method:
                #     print(f"np.sort(cs[~a])[::-1] : {np.sort(cs[~a])[::-1]}")
                #     print(f"np.sort(cs[~a])[::-1][method] : {np.sort(cs[~a])[::-1][method]}")
    
                scores[i] = np.sort(cs[~a])[::-1][method] if np.sum(~a) > method else min_score
    
        if method == "fraction_monotized":
            min_score = -1
            scores = np.zeros((len(claim_scores),))
            
            for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
                sorted_pairs = sorted(zip(cs, a))[::-1]
                # sorted_pairs = np.array(cs_sorted), np.array(a_sorted)
                cs_sorted, a_sorted = zip(*sorted_pairs)
                cs_sorted, a_sorted = np.array(cs_sorted), np.array(a_sorted)
                mono_fraction_arr = np.array([np.mean(a_sorted[:(j+1)]) for j in range(len(a_sorted))]) ## monotized fraction factual
                # print(f"cs_sorted : {cs_sorted}")
                # print(f"a_sorted : {a_sorted}")
                # print(f"mono_fraction_arr : {mono_fraction_arr}")
                # print(f"(1 - mono_fraction_arr) > epsilon : {(1 - mono_fraction_arr) > epsilon}") 
                scores[i] = cs_sorted[(1 - mono_fraction_arr) > epsilon][0] if np.sum((1 - mono_fraction_arr) > epsilon) > 0 else min_score
                # print(f"scores[i] : {scores[i]}")
        
        if method == "fraction_nonmonotone":
            min_score = -1
            scores = np.zeros((len(claim_scores),))
            
            for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
                scores[i] = np.mean(~a[cs >= tau]) if np.sum(cs >= tau) > 0 else min_score
    
        
        return scores
    
    
    
    def loss_factuality(
        claim_scores : List[np.ndarray], ## Float point scores
        annotations : List[np.ndarray], ## Boolean annotations
        tau : float,
        min_score: int = 0
    ):
        ## Returns 1 if there is some included claim (with score >= tau) that is False
        annotations_included = annotations[claim_scores>=tau]
        return int(max(~annotations_included)) if len(annotations_included) > 0 else min_score
    
    
    
    def loss_factuality_fraction(
        claim_scores : List[np.ndarray], ## Float point scores
        annotations : List[np.ndarray], ## Boolean annotations
        tau : float,
        epsilon : float = None,
        min_loss: int = 0
    ):
        ## Returns 1 if there is some included claim (with score >= tau) that is False
        annotations_included = annotations[claim_scores>=tau]
        if epsilon is None:
            return np.mean(~annotations_included) if len(annotations_included) > 0 else min_loss
        else:
            return int(np.mean(~annotations_included) > epsilon) if len(annotations_included) > 0 else min_loss
            
    
    def loss_factuality_assymetric(
        claim_scores : List[np.ndarray], ## Float point scores
        annotations : List[np.ndarray], ## Boolean annotations
        tau : float,
        epsilon : float = None,
        min_loss: int = 0,
        max_num_subclaims: int = 65
    ):
        ## Returns 1 if there is some included claim (with score >= tau) that is False
        annotations_included = annotations[claim_scores>=tau]
        if epsilon is None:
            # if len(annotations_included) == 0 or np.sum(~annotations_included) == 0:
            #     ## If null set or all subclaims are true, then return minimum loss
            #     return min_loss
            # elif np.sum(~annotations_included) > 0:
            #     ## If there is at least one untrue subclaim, then loss is number of untrue subclaims
            #     return np.sum(~annotations_included)
    
            ## Loss is 10*(Num untrue subclaims) - (Num true subclaims)
            return (np.sum(~annotations_included) - np.sum(annotations_included) / 10) / len(annotations_included) if len(annotations_included) > 0 else min_loss
            
        else:
            raise Exception("Threshold epsilon not yet implemented for this loss function")
            # return int(np.mean(~annotations_included) > epsilon) if len(annotations_included) > 0 else min_loss
    
    
    def loss_factuality_length(
        claim_scores : List[np.ndarray], ## Float point scores
        annotations : List[np.ndarray], ## Boolean annotations
        tau : float,
        epsilon : float = None,
        min_loss: int = 0,
        # max_num_subclaims: int = 
    ):
        ## Returns 1 if there is some included claim (with score >= tau) that is False
        annotations_included = annotations[claim_scores>=tau]
        if epsilon is None:
            # if len(annotations_included) == 0 or np.sum(~annotations_included) == 0:
            #     ## If null set or all subclaims are true, then return minimum loss
            #     return min_loss
            # elif np.sum(~annotations_included) > 0:
            #     ## If there is at least one untrue subclaim, then loss is number of untrue subclaims
            #     return np.sum(~annotations_included)
    
            ## Scaling fraction by number of total claims
            if len(annotations_included) == 0:
                return min_loss
    
            if len(annotations_included) <= 40:
                return np.mean(~annotations_included)
    
            else:
                return min(1, (np.sum(~annotations_included[:40]) + max(0, len(annotations_included) - 40)) / len(annotations_included))
            
            # return np.mean(~annotations_included) * len(annotations_included) if len(annotations_included) > 0 else min_loss
            
        else:
            raise Exception("Threshold epsilon not yet implemented for this loss function")
            # return int(np.mean(~annotations_included) > epsilon) if len(annotations_included) > 0 else min_loss
            
    
    # def loss_factuality_fraction_tol(
    #     claim_scores : List[np.ndarray], ## Float point scores
    #     annotations : List[np.ndarray], ## Boolean annotations
    #     tau : float,
    #     epsilon : float = 0.1,
    #     min_loss: int = 0
    # ):
    #     ## Returns 1 if there is some included claim (with score >= tau) that is False
    #     annotations_included = annotations[claim_scores>=tau]
    #     return int(np.mean(~annotations_included) > epsilon) if len(annotations_included) > 0 else min_loss
        
    
    
    n = 100
    n_subclaims = 100
    alpha = 0.15
    epsilon = 0.15
    model_probs = [np.random.uniform(size=n_subclaims) for i in range(n)]
    subclaim_annotations = [np.random.binomial(n=1, p=model_probs_curr) for model_probs_curr in model_probs]
    
    # print(f"subclaim_annotations : \n {subclaim_annotations}")
    # print(f"model_probs : \n {model_probs}")
    
    for i in range(n):
        # Zip the lists together, sort, and unzip
        model_probs_curr, subclaims_curr = model_probs[i], subclaim_annotations[i]
        zipped_pairs = zip(model_probs_curr, subclaims_curr)
        sorted_pairs = sorted(zipped_pairs)[::-1]
    
        model_probs_curr, subclaims_curr = zip(*sorted_pairs)
        model_probs[i], subclaim_annotations[i] = np.array(model_probs_curr), np.array(subclaims_curr).astype(bool)
    
    
    # print(f"subclaim_annotations sorted : \n {subclaim_annotations}")
    
    # print(f"model_probs sorted : \n {model_probs}")
    
    
    nonconformity_scores = score_func(model_probs, subclaim_annotations, method=0)
    # print(f"nonconformity_scores : {nonconformity_scores}")
    nonconformity_scores_sorted = np.array(sorted(nonconformity_scores))
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    # print(f"ind_q : {ind_q}")
    # print(f"nonconformity_scores_sorted : {nonconformity_scores_sorted}")
    print(f"fitted tau (strict) : {nonconformity_scores_sorted[ind_q - 1]}")
    
    ##
    nc_scores_frac_mono = score_func(model_probs, subclaim_annotations, method="fraction_monotized", epsilon = epsilon)
    nc_scores_frac_mono_sorted = np.array(sorted(nc_scores_frac_mono))
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    # print(f"nc_scores_frac_mono : {nc_scores_frac_mono}")
    print(f"fitted tau (monotized fraction) {nc_scores_frac_mono_sorted[ind_q - 1]}")
    
    
    import sys
    
    def rc_frac_factuality(claim_scores : List[np.ndarray],
                             annotations : List[np.ndarray],
                             taus_to_search,
                             epsilon,
                             alpha,
                             method_name = "gcrc", ## "gcrc", "monotized_losses_crc", "standard_crc", "ltt"
                             small_num_adjust = 1e-10,
                             n_grid = 500, ## Number of threshold to search, ## 1000
                             B = 1, ## Maximum loss
                             loss_name = "loss_factuality_fraction"
                            ):
    
        if (len(claim_scores) != len(annotations)):
            raise Exception(f"len(claim_scores) = {len(claim_scores)} != {len(annotations)} = len(annotations)")
    
        n = len(claim_scores)
    
    
        taus_to_search = np.unique(taus_to_search)[::-1] + small_num_adjust ## Sort descending (safest to most aggressive)
        num_taus_unique = len(taus_to_search)
        k = max(int(num_taus_unique / n_grid), 1)
        # print(f"len all taus : {len(taus_to_search)}")
        
        taus_to_search = taus_to_search[::k]
        # print(f"len subset taus : {len(taus_to_search)}")
    
        # print(f"taus_to_search : {taus_to_search}")
             
        risk_prev = 0.0
        tau_prev = 1.0 + 1e-10
    
        if taus_to_search[0] < 1.0:
            ## Make sure to include safe tau value
            taus_to_search = np.concatenate(([1.0], taus_to_search))
    
        if method_name == "standard_crc":
            ## Standard CRC implemented by searching from most-aggressive (smallest) hyperparameter to safest (largest)
            taus_to_search = taus_to_search[::-1]
        
        losses = np.zeros(n+1)
        losses[n] = B ## Conservative loss for test point
        risks = [risk_prev]
        
        for t, tau in enumerate((taus_to_search)):
            
            for i in range(n):
                if method_name in ["gcrc", "standard_crc", "ltt"] or method_name == "monotized_losses_crc" and t == 0:
                    losses[i] = eval(loss_name)(claim_scores[i], annotations[i], tau, epsilon = epsilon)
                
                elif method_name == "monotized_losses_crc" and t > 0:
                    ## If running monotized-losses CRC (Corollary 1 in Angelopoulos et al. (2024), loss is maximum seen so far for that point)
                    losses[i] = max(losses[i], eval(loss_name)(claim_scores[i], annotations[i], tau, epsilon = epsilon))
    
                else:
                    raise Exception(f"method_name '{method_name}' not recognized!")
                    
    
            if method_name == "ltt":
                ## For LTT, no conservative "test point loss"
                risk_curr = losses[:n].mean()
                p_val_curr = hb_p_value(risk_curr, n, alpha)
                
            else:
                risk_curr = losses.mean()
    
            risks.append(risk_curr)
    
            if method_name in ["gcrc", "monotized_losses_crc"] and risk_curr > alpha:
                ## Stopping condition for GCRC and monotized-losses CRC methods, return most recent tau prior to current
                break
    
            elif method_name == "ltt" and p_val_curr > alpha:
                ## Stopping condition for LTT is based on calculated p-value
                break
                
            elif method_name == "standard_crc" and risk_curr <= alpha:
                ## Stopping condition for standard CRC: return first identified safe hyperparam
                return tau, losses.mean(), risks
            
            else:
                risk_prev = risk_curr
                tau_prev = tau
    
        # print(f"selected tau : {tau_prev}")
        return tau_prev, risk_prev, risks
    
    
    # tau_strict, risk_strict = rc_frac_factuality(claim_scores=model_probs, annotations=subclaim_annotations, taus_to_search=nonconformity_scores, epsilon = 0.0, alpha = alpha)
    tau_frac_gcrc, risk_frac_gcrc, risks_gcrc = rc_frac_factuality(claim_scores=model_probs, annotations=subclaim_annotations, taus_to_search=nonconformity_scores, epsilon = epsilon, alpha = alpha, method_name = "gcrc")
    tau_frac_mono, risk_frac_mono, risks_mono = rc_frac_factuality(claim_scores=model_probs, annotations=subclaim_annotations, taus_to_search=nonconformity_scores, epsilon = epsilon, alpha = alpha, method_name = "monotized_losses_crc")
    
    
    # print(f" GCRC tau strict factuality   : \n tau_strict={tau_strict}, risk_strict={risk_strict} \n")
    print(f" Monotized-CRC tau fraction factuality : \n tau_frac={tau_frac_mono}, risk_frac={risk_frac_mono}\n")
    print(f" GCRC tau fraction factuality          : \n tau_frac={tau_frac_gcrc}, risk_frac={risk_frac_gcrc}\n")
    
    
    # nc_scores_nonmono_frac = score_func(claim_scores=model_probs, annotations=subclaim_annotations, method="fraction_nonmonotone", epsilon=epsilon, tau=tau_frac)
    # nc_scores_nonmono_frac = np.array(sorted(nc_scores_nonmono_frac))
    # ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    # print(f"nonmonotone scores : {nc_scores_nonmono_frac}")
    # print(f"nc_scores_nonmono_frac : {nc_scores_nonmono_frac[ind_q - 1]}")



    remove_easy_data = False
    
    ## Loading datasets
    orig_datasets = {}
    suffix = '.jsonl'
    dataset_dir = "/home/drewprinster/conformal-safety/data/MedLFQAv2" #"/Users/cherian/Projects/conformal-safety/data/MedLFQAv2"
    for path in os.listdir(dataset_dir):
        dataset_name = path[:-len(suffix)]
        if not path.startswith('.'):
            with open(os.path.join(dataset_dir, path), 'r') as fp:
                orig_datasets[dataset_name] = [json.loads(line) for line in fp.readlines()]
    
    ## Dictionary where can search using quesiton and get name of original dataset it came from
    dataset_lookup = {}
    for name, data in orig_datasets.items():
        for dat in data:
            dataset_lookup[dat['Question']] = name
    
    ## Getting scores
    data_path = "/home/drewprinster/conformal-safety/data" #"/Users/cherian/Projects/conformal-safety/data"
    dataset_path = os.path.join(data_path, "medlfqa_dataset.pkl")
    freq_path = os.path.join(data_path, "medlfqa_frequencies.npz")
    logprob_path = os.path.join(data_path, "medlfqa_logprobs.npz")
    selfeval_path = os.path.join(data_path, "medlfqa_selfevals.npz")
    
    with open(dataset_path, 'rb') as fp:
        dataset = pickle.load(fp)
    
    
    ## HARD CODED FIX FOR REDUNDANT PROMPT...which has atomic_facts assigned to the wrong redundancy
    dataset[1132]['atomic_facts'] = dataset[1048]['atomic_facts']
    
    
    frequencies = np.load(freq_path)
    logprobs = np.load(logprob_path)
    selfevals = np.load(selfeval_path)
    
    drop_prompts = []
    for k in frequencies:
        if frequencies[k].ndim != 1:
            drop_prompts.append(k)
        elif np.allclose(selfevals[k], -1):
            drop_prompts.append(k)
        elif k not in logprobs:
            drop_prompts.append(k)
        elif remove_specific_leading_chars(k).strip() not in dataset_lookup:
            drop_prompts.append(k)
    
    # drop and match ordering of dataset
    dataset = [dat for dat in dataset if dat['prompt'] not in drop_prompts] ## List where each entry is a dictionary with 'prompt', 'response', and 'atomic_facts'
    full_dataset = dataset
    prompts_to_keep = [dat['prompt'] for dat in dataset] ## List where each entry is the full prompt
    names_to_keep = [p.split('about')[-1].strip()[:-1] for p in prompts_to_keep] ## List where each entry is an abbreviated prompt for a name
    
    ## Lists where each entry in the list is an array of scores for subclaims
    frequencies_arr = [frequencies[p] for p in prompts_to_keep] ## Frequency scoring
    selfevals_arr = [selfevals[p] for p in prompts_to_keep] ## Self-evaluation scoring
    logprobs_arr = [logprobs[p] for p in prompts_to_keep] ## Log-probability scoring
    annotations_arr = [np.asarray([af["is_supported"] for af in dat["atomic_facts"]]) for dat in dataset] ## Oracle (annotation) scoring
    ordinal_arr = [np.arange(len(f)) for f in frequencies_arr] ## Ordinal scoring (baseline)
    
    
    ## Record maximum subclaim scores
    
    # If removing data where all subclaims are true (for harder benchmark)
    if remove_easy_data:
        indicators_all_subclaims_true = []
        for i in range(len(dataset)):
            indicators_all_subclaims_true.append(bool(min([dataset[i]['atomic_facts'][j]['is_supported'] for j in range(len(dataset[i]['atomic_facts']))])))
    
        dataset = [d for i, d in enumerate(dataset) if indicators_all_subclaims_true[i]]
        full_dataset = dataset
        prompts_to_keep = [dat['prompt'] for dat in dataset] ## List where each entry is the full prompt
        names_to_keep = [p.split('about')[-1].strip()[:-1] for p in prompts_to_keep] ## List where each entry is an abbreviated prompt for a name
    
        frequencies_arr = [d for i, d in enumerate(frequencies_arr) if indicators_all_subclaims_true[i]] ## Frequency scoring
        selfevals_arr = [d for i, d in enumerate(selfevals_arr) if indicators_all_subclaims_true[i]] ## Self-evaluation scoring
        logprobs_arr = [d for i, d in enumerate(logprobs_arr) if indicators_all_subclaims_true[i]] ## Log-probability scoring
        annotations_arr = [d for i, d in enumerate(annotations_arr) if indicators_all_subclaims_true[i]] ## Oracle (annotation) scoring
        ordinal_arr = [d for i, d in enumerate(ordinal_arr) if indicators_all_subclaims_true[i]] ## Ordinal scoring (baseline)
    
    
    print(len(frequencies_arr), len(selfevals_arr), len(dataset), len(annotations_arr))





    # get prompt-level features
    
    # names of datasets
    dataset_arr = [dataset_lookup[remove_specific_leading_chars(p).strip()] for p in prompts_to_keep] ## List of original dataset name for each question
    dataset_dummies = pd.get_dummies(dataset_arr) ## Pandas dataframe with dummies indicating dataset origin name for each question
    dataset_names = [name[:-3] if name.endswith("_qa") else name for name in dataset_dummies.columns] ## Unique list of (five) dataset names
    
    # length of response
    response_len_arr = [len(dat['response']) for dat in dataset]
    response_len_arr = np.asarray(response_len_arr).reshape(-1,1) ## Array where each entry is the string length of the response to each question
    
    # length of prompt
    prompt_len_arr = [len(remove_specific_leading_chars(dat['prompt']).strip()) for dat in dataset]
    prompt_len_arr = np.asarray(prompt_len_arr).reshape(-1,1) ## Array where each entry is the string length of the prompt/question
    
    # mean (exponentiated) logprob 
    logprobs_mean_arr = [np.mean(arr) for arr in logprobs_arr]
    logprobs_mean_arr = np.asarray(logprobs_mean_arr).reshape(-1,1) ## Array where each entry is the mean log-probability (over subclaims) for each response
    
    # std (exponentiated) logprob
    logprobs_std_arr = [np.std(arr) for arr in logprobs_arr]
    logprobs_std_arr = np.asarray(logprobs_std_arr).reshape(-1,1) ## Array where each entry is the std log-probability (over subclaims) for each response
    
    z_ones = np.ones((len(frequencies_arr), 1)) ## Array of ones equal in length to dataset
    z_arr = np.concatenate((z_ones, response_len_arr, prompt_len_arr, logprobs_mean_arr, logprobs_std_arr), axis=1)
    z_dummies = dataset_dummies.to_numpy()
    
    ## Array of features computed using prompt and response (X_i) in paper
    ## Cols: (0) ones, (1) response len, (2) prompt len, (3) mean log-prob over subclaims, (4) std log-prob over subclaims
    ## (5, 6, 7, 8) indicators for 4 of 5 datasets (last one is indicated by first 4 being false)
    z_arr_dummies = np.concatenate((z_arr, z_dummies[:,:-1]), axis=1) 
    
    print(len(dataset_arr), len(response_len_arr), len(prompt_len_arr), len(logprobs_mean_arr), len(logprobs_std_arr))
    dataset_dummies




    from typing import List
    # from conditionalconformal import CondConf
    from condconf import CondConf
    
    def split_dataset(dataset, rng, train_frac = 0.8, train_num = None):
        x, y = dataset
        ind = np.arange(len(x))
        rng.shuffle(ind)
        if train_num is None:
            ## Providing train_num as argument overrules train_frac (else train_frac determines train_num)
            train_num = int(train_frac * len(x))
        train_ind = ind[0:train_num]
        calib_ind = ind[train_num:]
    
        x_train = [x[i] for i in train_ind]
        y_train = [y[i] for i in train_ind]
    
        x_calib = [x[i] for i in calib_ind]
        y_calib = [y[i] for i in calib_ind]
    
        return (x_train, y_train), (x_calib, y_calib), train_ind, calib_ind
        
    def score_func(
        claim_scores : List[np.ndarray],
        annotations : List[np.ndarray],
        method : str = "max"
    ):
        if method == "max":
            min_score = -1
            scores = np.zeros((len(claim_scores),))
            for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
                scores[i] = np.max(cs[~a]) if np.sum(~a) >= 1 else min_score
        if isinstance(method, int):
            ## Think that 'method' here is \lambda in the Cherian et al paper (but tau in that paper is the lambda in loss calibration)
            min_score = -1
            scores = np.zeros((len(claim_scores),))
            for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
                ## if (number of untrue claims) > method:
                    ## scores[i] = (largest sub-claim score of any False sub-claim)
                ## else:
                    ## scores[i] = min_score
                # print((cs, a))
                # if np.sum(~a) > method:
                #     print(f"np.sort(cs[~a])[::-1] : {np.sort(cs[~a])[::-1]}")
                #     print(f"np.sort(cs[~a])[::-1][method] : {np.sort(cs[~a])[::-1][method]}")
    
                scores[i] = np.sort(cs[~a])[::-1][method] if np.sum(~a) > method else min_score
        return scores
    
    def split_threshold(
        conf_scores : np.ndarray,
        quantile
    ):
        n = len(conf_scores)
        threshold = np.sort(conf_scores)[int(np.ceil(quantile * (n + 1)))]
        return threshold
    def get_frac_true_claims_retained(claim_scores, annotations, thresholds):
        recall = []
        for cs, a, t in zip(claim_scores, annotations, thresholds):
            frac = np.sum((cs > t) & a) / np.sum(a) if np.sum(a) > 0 else 0
            recall.append(frac)
        return recall
    
    def get_retained_claims(claim_scores, thresholds):
        claims_retained = []
        for cs, t in zip(claim_scores, thresholds):
            claims_retained.append(np.mean(cs > t))
        return claims_retained
    
    def get_retained_claim_indices(claim_scores, thresholds):
        claims_retained = []
        for cs, t in zip(claim_scores, thresholds):
            claims_retained.append(np.where(cs > t)[0])
        return claims_retained
    
    def get_validity(claim_scores, annotations, threshold, method):
        conf_scores = score_func(claim_scores, annotations, method)
        validity = conf_scores <= threshold
        return validity
    
    def run_split_conformal(x_arr, y_arr, method, quantile):
        conf_scores = score_func(x_arr, y_arr, method=k)
        threshold = split_threshold(conf_scores, quantile)
        return conf_scores, threshold





    BLUE = '#2166ac'           # β̂ vertical line
    TEAL = '#5ab4ac'           # Calibration data (complements blue/red)
    RED = '#d73027'            # Test point (conservative/warning)
    
    def empirical_cdf(
        values, ## numpy array with values
        x ## value to evaluate empirical CDF at
    ) -> float:
        return len(values[values <= x])/len(values)
    
    
    def get_taus_grid_from_data(claim_scores : List ## List of arrays of subclaim scores
                               ):
        taus_set = []
        for i, cs in enumerate(claim_scores):
            taus_set.extend(cs)
            
        return np.array(taus_set)
    
    
    def run_crc_trial(x_arr, ## List where each entry is an array of sub-claim scores for a response
                      y_arr, ## List where each entry is an array of sub-claim "oracle scores" or annotations
                      z_arr, ## List of features for the prompt and response used for conditional calibration (all ones for marginal cp)
                      rng, method_name, alpha, epsilon, loss_name = "loss_factuality_fraction", cal_frac = 0.7):
    
        data_calib, data_test, idx_calib, idx_test = split_dataset((x_arr, y_arr), rng, train_frac=cal_frac) ## here "train_frac" is actually for cal
        # print(data_calib[0])
        # print(f"num cal : {len(data_calib[0])}")
        # scores_calib = score_func(*data_calib, method=method) ## Calibration set nonconformity scores
        # scores_test = score_func(*data_test, method=method) ## Test set nonconformity scores
    
        # print(data_calib)
        taus_to_search = get_taus_grid_from_data(data_calib[0])
        
        
        threshold, _, risks = rc_frac_factuality(*data_calib, taus_to_search=taus_to_search, epsilon=epsilon, alpha=alpha, \
                                                 method_name=method_name, loss_name = loss_name) ## "gcrc", "monotized_losses_crc",
        # valid_inds = []
        constraint_violations = []
        claim_perc = []
    
        
    
        for i, j in enumerate(idx_test):
            loss = eval(loss_name)(data_test[0][i], data_test[1][i], tau=threshold, epsilon=epsilon)
            
            # valid_inds.append(threshold >= scores_test[i])
            constraint_violations.append(loss)
            
            claim_perc.append(get_frac_true_claims_retained([data_test[0][i]], [data_test[1][i]], [threshold])[0])
                              
        # valid_inds = np.asarray(valid_inds).flatten()
        constraint_violations = np.asarray(constraint_violations).flatten()
        claim_perc = np.asarray(claim_perc).flatten()
    
        return np.mean(constraint_violations), np.mean(claim_perc), risks
    
        # all_covs = np.zeros((1,))
        # marginal_cov = np.mean(constraint_violations)
        # all_covs[0] = marginal_cov
    
        # all_claims = np.zeros((1,))
        # all_claims[0] = np.mean(claim_perc)
    
        # dataset_covs = np.zeros(len(dataset_names))
        # dataset_claims = np.zeros(len(dataset_names))
    
        # for d_idx in range(len(dataset_names)):
        #     dataset_cov = np.sum(dataset_dummies.to_numpy()[idx_test,d_idx] * constraint_violations) / np.sum(dataset_dummies.to_numpy()[idx_test,d_idx])
        #     dataset_covs[d_idx] = dataset_cov
        #     dataset_perc = np.sum(dataset_dummies.to_numpy()[idx_test,d_idx] * claim_perc) / np.sum(dataset_dummies.to_numpy()[idx_test,d_idx])
        #     dataset_claims[d_idx] = dataset_perc
    
        # return np.concatenate((all_covs, dataset_covs)), np.concatenate((all_claims, dataset_claims))
    
    
    def run_coverage_trial(x_arr, ## List where each entry is an array of sub-claim scores for a response
                           y_arr, ## List where each entry is an array of sub-claim "oracle scores" or annotations
                           z_arr, ## List of features for the prompt and response used for conditional calibration (all ones for marginal cp)
                           rng, method, quantile):
        data_calib, data_test, idx_calib, idx_test = split_dataset((x_arr, y_arr), rng, train_frac=0.7)
        print(f"data_calib : {data_calib[0][0]}, {data_calib[1][0]}")
        scores_calib = score_func(*data_calib, method=method) ## Calibration set nonconformity scores
        scores_test = score_func(*data_test, method=method) ## Test set nonconformity scores
        
        print(f"cal_test : {len(scores_calib)}")
    
        print(f"scores_test : {len(scores_test)}")
        
        condconf = CondConf(lambda x,y: y, lambda x: x)
        condconf.setup_problem(z_arr[idx_calib], scores_calib)
        # print(condconf.x_calib)
        # print(condconf.y_calib)
        # print(condconf.scores_calib)
        # print(condconf.cvx_problem)
    
        
        # scores, threshold = run_split_conformal(*data_calib, method=method, quantile=quantile)
        valid_inds = []
        claim_perc = []
        # print(f"data_test[0] len : {len(data_test[0])}, {data_test[0][0]}")
        # print(f"data_test[1] len : {len(data_test[1])}, {data_test[1][0]}")
        # for i in range(10):
        #     data_concat = np.c_[data_test[0][i], data_test[1][i]]
            
        #     sort_indices = data_concat[:, 0].argsort()
        #     sorted_data = data_concat[sort_indices]
        #     print(f"sorted_data : {sorted_data}")
    
    
        # print(f"idx_test : {idx_test}")
        for i, j in enumerate(idx_test):
            # print(z_arr[j].reshape(1,-1))
            try:
                threshold = condconf.predict(quantile, z_arr[j].reshape(1,-1), lambda c, x: c, randomize=True) # S_min=-1, S_max=1,
            except Exception as e:
                # print(f"An exception occurred: {e}") 
                # traceback.print_exc()
                threshold = [np.inf]
                # threshold = [1]
            # print(f"threshold : {threshold}, scores_test[i] : {scores_test[i]}")
            valid_inds.append(threshold >= scores_test[i])
            claim_perc.append(get_retained_claims([data_test[0][i]], threshold)[0])
                              
        valid_inds = np.asarray(valid_inds).flatten()
        claim_perc = np.asarray(claim_perc).flatten()
    
        # valid_inds = get_validity(*data_calib, threshold, method)
        all_covs = np.zeros((1,))
        marginal_cov = np.mean(valid_inds)
        all_covs[0] = marginal_cov
    
        all_claims = np.zeros((1,))
        all_claims[0] = np.mean(claim_perc)
    
        dataset_covs = np.zeros(len(dataset_names))
        dataset_claims = np.zeros(len(dataset_names))
    
        for d_idx in range(len(dataset_names)):
            dataset_cov = np.sum(dataset_dummies.to_numpy()[idx_test,d_idx] * valid_inds) / np.sum(dataset_dummies.to_numpy()[idx_test,d_idx])
            dataset_covs[d_idx] = dataset_cov
            dataset_perc = np.sum(dataset_dummies.to_numpy()[idx_test,d_idx] * claim_perc) / np.sum(dataset_dummies.to_numpy()[idx_test,d_idx])
            dataset_claims[d_idx] = dataset_perc
        
        return np.concatenate((all_covs, dataset_covs)), np.concatenate((all_claims, dataset_claims))





    n_trials = 10
    # all_covs = np.zeros((n_trials, 1 + len(dataset_names)))
    # all_claims = np.zeros((n_trials, 1 + len(dataset_names)))
    
    
    rng = np.random.default_rng(seed=0)
    k = 0
    # quantile = 0.9
    alphas = np.arange(0.005, 0.105, 0.005)
    # alphas = np.arange(0.005, 0.006, 0.005)

    print(alphas)
    
    
    epsilon = None #0.1
    cal_frac = 0.7
    
    risk_dict = {}
    claims_dict = {}
    
    # frequencies_arr_jitter = [freq + rng.uniform(low=0, high=1e-3, size=freq.shape) for freq in frequencies_arr]
    # rng = np.random.default_rng(seed=1)
    # for trial in tqdm(range(n_trials)):
    #     cov_result, claims_result = run_coverage_trial(frequencies_arr_jitter, annotations_arr, z_arr_dummies, rng, k, quantile)
    #     all_covs[trial] = cov_result
    #     all_claims[trial] = claims_result
    
    score_names = ["logprobs", "frequency", "selfevals"] #"frequency", "selfevals", 
    score_arr_dict = {"logprobs" : logprobs_arr, "frequency" : frequencies_arr, "selfevals" : selfevals_arr}
    method_names = ["monotized_losses_crc", "ltt", "gcrc"] # "standard_crc", "gcrc", "monotized_losses_crc", "monotized_losses_crc", "ltt",
    loss_name = "loss_factuality_fraction" #"loss_factuality_fraction"
    
    for s_i, score_name in enumerate(["selfevals"]): 
        subclaim_scores_arr = score_arr_dict[score_name]
        
        print(f"Running experiments for {score_name} scoring...")
        for method_name in method_names: #"gcrc", "monotized_losses_crc", 
            print(method_name)
            risk_dict[method_name] = pd.DataFrame(np.c_[alphas, np.zeros(len(alphas)), np.zeros(len(alphas))], columns=["alphas", "risk_mean", "risk_std"])
            claims_dict[method_name] = pd.DataFrame(np.c_[alphas, np.zeros(len(alphas)), np.zeros(len(alphas))], columns=["alphas", "claims_mean", "claims_std"])
        
            
            for a, alpha in enumerate(alphas):
        
                risks = np.zeros(n_trials)
                claims = np.zeros(n_trials)
    
                subclaim_scores_arr_jitter = [np.minimum(1, subclaim_scores + rng.uniform(low=0, high=1e-8, size=subclaim_scores.shape)) for subclaim_scores in subclaim_scores_arr]
    
                # subclaim_scores_arr_jitter = [subclaim_scores + rng.uniform(low=0, high=1e-3, size=subclaim_scores.shape) for subclaim_scores in subclaim_scores_arr]
                rng = np.random.default_rng(seed=1)
                
                for trial in tqdm(range(n_trials)):
                    risk, claim_perc, _ = run_crc_trial(subclaim_scores_arr_jitter, annotations_arr, z_arr_dummies, rng, \
                                                     method_name=method_name, alpha=alpha, epsilon=epsilon, loss_name = loss_name,\
                                                     cal_frac=cal_frac)
                    # print(f"constraint_violations : {constraint_violations}")
                    # print(f"claim_perc : {claim_perc}")
                    risks[trial] = risk
                    claims[trial] = claim_perc
        
                risk_dict[method_name].iloc[a, 1] = np.mean(risks)
                risk_dict[method_name].iloc[a, 2] = np.std(risks)
                claims_dict[method_name].iloc[a, 1] = np.mean(claims)
                claims_dict[method_name].iloc[a, 2] = np.std(claims)
            

        
        method_name_map = {'gcrc' : 'GCRC (proposed)', 
                           'monotized_losses_crc' : 'Monotized-losses CRC \n(Mohri & Hashimoto, 2024; \nAngelopoulos et al., 2024)',
                           'standard_crc' : 'standard CRC (Angelopoulos, et al., 2024)',
                           'ltt' : 'LTT (Angelopoulos, et al., 2025)'}
        
        colors_dict = {'gcrc' : BLUE, 'monotized_losses_crc' : 'C1', 'ltt' : 'C2', 'standard_crc' : 'C6'}
        markers_dict = {'gcrc' : 'o', 'monotized_losses_crc' : 's', 'ltt' : '^', 'standard_crc' : 'X'}
        
        results_df = pd.DataFrame()
        results_df['alphas'] = alphas
                       
        for metric in ['risk', 'claims']:
            metric_dict = eval(f'{metric}_dict')
            for m, method_name in enumerate(method_names):
                alphas = metric_dict[method_name]['alphas']
                metric_mean = metric_dict[method_name][f'{metric}_mean']
                metric_std = metric_dict[method_name][f'{metric}_std']
                results_df[f'{method_name}_{metric}_mean'] = metric_mean
                results_df[f'{method_name}_{metric}_std'] = metric_std

    
        results_df.to_csv(f'{loss_name}Loss_{score_name}Scoring_{n_trials}trials_200ngrid_v2.csv')
