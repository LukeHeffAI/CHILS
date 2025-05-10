import argparse
import os
import sys
import glob
import pickle as pk
import openai
import json
from models.clip_models import *
from src.zshot_utils import *
from src.constants import *
from src.get_inputs import *
from dotenv import load_dotenv

# Add the project root to the path to ensure imports work correctly
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import dataset config from run.py
from run import DATASET_CONFIG

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Detect available models and datasets from existing feature files
def detect_available_features():
    """Find existing feature files and determine available models and datasets"""
    available = {"models": set(), "datasets": {}}
    
    # Look for all conf_*.npz files
    for feature_file in glob.glob(os.path.join(project_root, "*/conf_*.npz")):
        model_dir = os.path.basename(os.path.dirname(feature_file))
        dataset = os.path.basename(feature_file).replace("conf_", "").replace(".npz", "")
        
        # Add to available models
        available["models"].add(model_dir)
        
        # Map model to its available datasets
        if model_dir not in available["datasets"]:
            available["datasets"][model_dir] = []
        available["datasets"][model_dir].append(dataset)
    
    # Default selections based on available files
    default_model = next(iter(available["models"])) if available["models"] else "ClipViTB32"
    default_dataset = (available["datasets"].get(default_model, ["cub"]) or ["cub"])[0]
    
    return {
        "model": default_model,
        "dataset": default_dataset,
        "available": available,
    }

# Get available features and set defaults based on what's already been generated
available_features = detect_available_features()
default_model = available_features["model"]
default_dataset = available_features["dataset"]
data_dir = os.path.expanduser("~/Documents/GitHub/data")

label_set_size = DATASET_CONFIG[default_dataset]['num_classes'] if default_dataset in DATASET_CONFIG else 10

print(f"Available models: {available_features['available']['models']}")
for model in available_features['available']['datasets']:
    print(f"Available datasets for {model}: {available_features['available']['datasets'][model]}")
print(f"Using defaults - Model: {default_model}, Dataset: {default_dataset}")

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default=default_dataset)
parser.add_argument("--domain", type=str, default=default_dataset)
parser.add_argument("--model", type=str, default=default_model)
parser.add_argument("--experiment", type=str, default='gpt')
parser.add_argument("--data-dir", type=str, default=data_dir)
parser.add_argument("--mod-out-dir", type=str, default='')
parser.add_argument("--out_dir", type=str, default='./zshot_outputs')
parser.add_argument("--label-set-size", type=int, default=label_set_size)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--rerun-gpt", action='store_true')
parser.add_argument("--use-gpt", action='store_true')
parser.add_argument("--best-poss", action='store_true')
parser.add_argument("--reweighter",type=str, default='normal')
parser.add_argument("--opt-templates", action='store_true')
parser.add_argument("--superclass-set-ens", action='store_true')


def get_dataset_path(dataset_name):
    """Determine the correct dataset path based on dataset name"""
    if dataset_name == 'cub':
        return data_dir
    elif dataset_name == 'dtd':
        return data_dir + '/DTD'
    elif dataset_name == 'pets':
        return data_dir + '/Oxford_Pets'
    elif dataset_name == 'places365':
        return data_dir + '/places_devkit/torch_download/'
    elif dataset_name == 'food-101':
        return data_dir + '/FOOD_101'
    elif dataset_name == 'eurosat':
        return data_dir
    else:
        # Default path structure
        return data_dir + '/' + dataset_name


def check_args(args):
    # Check if the feature file exists for the specified model and dataset
    feature_file = f"{args.model}/conf_{args.dataset}.npz"
    if not os.path.exists(feature_file):
        print(f"WARNING: Feature file '{feature_file}' does not exist!")
        print(f"You need to run run.py first with model={args.model} and dataset={args.dataset}")
        print(f"Or choose one of the available model/dataset combinations:")
        for model in available_features['available']['datasets']:
            for dataset in available_features['available']['datasets'][model]:
                print(f"  --model {model} --dataset {dataset}")
        sys.exit(1)
    
    if args.dataset not in DATASETS:
        raise NotImplementedError(f"Invalid dataset: {args.dataset}")
    if args.domain != "all" and args.dataset in DOMAINS.keys() and args.domain not in DOMAINS[args.dataset]:
        # For datasets not specified in DOMAINS, use the dataset name as the domain
        if args.dataset not in DOMAINS.keys():
            args.domain = args.dataset
        else:
            raise NotImplementedError(f"Invalid domain: {args.domain}")
    if args.model not in MODELS:
        raise NotImplementedError(f"Invalid model: {args.model}")
    if args.experiment not in EXPERIMENTS:
        raise NotImplementedError(f"Invalid experiment : {args.experiment}")
    if args.dataset not in TRUESETS and args.experiment == 'true':
        raise NotImplementedError(f"No ground-truth subsets for {args.dataset}")
        
    # Set default data_dir path based on the dataset
    if args.data_dir == data_dir:  # If using default value
        args.data_dir = get_dataset_path(args.dataset)


if __name__ == '__main__':
    args = parser.parse_args()
    check_args(args)
    print(args)
    mod = eval(args.model)()
    
    # Get the appropriate run function based on template requirements
    if args.opt_templates:
        run_func = runBestTemplate
    else:
        # Use the correct function name that exists in src.zshot_utils
        run_func = run
    
    if args.domain == 'all' and args.dataset in DOMAINS.keys():
        domains = DOMAINS[args.dataset]
        out = {}
        for domain in domains:
            args.domain = domain
            features, labels, sub2super, super_classes, indices = get_inputs(args)
            out_dom, _ = run_func(mod, features, labels, sub2super, super_classes, indices, args)
            args.rerun_gpt = False
            if args.opt_templates:
                for temp, d in out_dom.items():
                    if temp not in out:
                        out[temp] = {}
                    for k, v in d.items():
                        if k in out[temp].keys():
                            out[temp][k] += d[k] / len(domains)
                        else:
                            out[temp][k] = d[k] / len(domains)
            else:
                for k, v in out_dom.items():
                    if k in out.keys():
                        out[k] += out_dom[k] / len(domains)
                    else:
                        out[k] = out_dom[k] / len(domains)
        print(out)
    else:
        features, labels, sub2super, super_classes, indices = get_inputs(args)
        out, preds_d = run_func(mod, features, labels, sub2super, super_classes, indices, args)
        print(out)
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    pref = f"/{args.dataset}-{args.model}-{args.experiment}-{args.label_set_size}-{args.reweighter}"
    if args.superclass_set_ens:
        pref = f"/{args.dataset}-{args.model}-{args.experiment}-{args.label_set_size}-{args.reweighter}-sse"
    with open(args.out_dir + pref + f".json", 'w') as f:
        json.dump(out, f, indent=2)
    if args.domain != 'all':
        with open(args.out_dir + pref + f"-outputs.pkl", 'wb') as f:
            pk.dump(preds_d, f)



