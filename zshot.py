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

#######################################################################################
# USER CONFIGURATION - Edit these variables to set your models and datasets
#######################################################################################
# Set to None to use command-line arguments, or specify values to hardcode them
USER_MODEL = "ClipViTB32"  # Options include: ClipViTB32, ClipViTB16, ClipViTL14, etc.
USER_DATASET = "dtd"       # Options include: cub, food-101, pets, dtd, eurosat, etc.
USER_DOMAIN = None         # Domain to use (if applicable), set to None to use dataset name
USER_EXPERIMENT = 'gpt'    # Default experiment type
USER_DATA_DIR = os.path.expanduser("~/Documents/GitHub/data")  # Base data directory

# Set to True to run all these models and datasets in sequence
# This overrides USER_MODEL and USER_DATASET if set to True
RUN_MULTIPLE = True
MODELS_TO_RUN = ["ClipViTB32"]
DATASETS_TO_RUN = ["cub", "dtd", "eurosat", "food-101", "imagenet", "pets", "places365"]         # List of datasets to run
#######################################################################################

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
    
    return available

data_dir = USER_DATA_DIR

# Parse command line arguments first to allow for user selection
parser = argparse.ArgumentParser(description='Zero-shot Learning with CHiLS')

# Will populate with suggested defaults after checking what's available
parser.add_argument("--dataset", type=str, default=USER_DATASET, help="Dataset to use")
parser.add_argument("--domain", type=str, default=USER_DOMAIN, help="Domain to use (if applicable)")
parser.add_argument("--model", type=str, default=USER_MODEL, help="Model to use")
parser.add_argument("--experiment", type=str, default=USER_EXPERIMENT)
parser.add_argument("--data-dir", type=str, default=data_dir)
parser.add_argument("--mod-out-dir", type=str, default='')
parser.add_argument("--out_dir", type=str, default='./zshot_outputs')
parser.add_argument("--label-set-size", type=int, help="Number of classes in the label set")
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--rerun-gpt", action='store_true')
parser.add_argument("--use-gpt", action='store_true')
parser.add_argument("--best-poss", action='store_true')
parser.add_argument("--reweighter",type=str, default='normal')
parser.add_argument("--opt-templates", action='store_true')
parser.add_argument("--superclass-set-ens", action='store_true')
parser.add_argument("--list-available", action='store_true', help="List all available models and datasets and exit")

# Get available features to inform the user of choices and set defaults
available_features = detect_available_features()


def get_dataset_path(dataset_name):
    """Determine the correct dataset path based on dataset name"""
    if dataset_name == 'cub':
        return data_dir
    elif dataset_name == 'dtd':
        return data_dir + '/DTD'
    elif dataset_name == 'pets':
        return data_dir
    elif dataset_name == 'places365':
        return data_dir
    elif dataset_name == 'food-101':
        return data_dir + '/FOOD_101'
    elif dataset_name == 'eurosat':
        return data_dir
    elif dataset_name == 'imagenet':
        return data_dir
    else:
        # Default path structure
        return data_dir + '/' + dataset_name


def check_args(args):
    # Check if the model is specified or available
    if args.model is None:
        if available_features["models"]:
            print("Error: Model not specified. Please choose one of the available models:")
            for model in available_features["models"]:
                print(f"  --model {model}")
        else:
            print("Error: No models available. Please run run.py first to generate feature files.")
        sys.exit(1)
        
    # Check if the model exists in the available models
    if args.model not in available_features["models"]:
        print(f"Error: Model '{args.model}' not found. Available models are:")
        for model in available_features["models"]:
            print(f"  --model {model}")
        sys.exit(1)
        
    # Check if the dataset is specified or available
    if args.dataset is None:
        if available_features["datasets"].get(args.model, []):
            print(f"Error: Dataset not specified. Please choose one of the available datasets for model '{args.model}':")
            for dataset in available_features["datasets"][args.model]:
                print(f"  --dataset {dataset}")
        else:
            print(f"Error: No datasets available for model '{args.model}'. Please run run.py first to generate feature files.")
        sys.exit(1)
        
    # Check if the dataset exists for the selected model
    if args.dataset not in available_features["datasets"].get(args.model, []):
        print(f"Error: Dataset '{args.dataset}' not found for model '{args.model}'. Available datasets for this model are:")
        for dataset in available_features["datasets"].get(args.model, []):
            print(f"  --dataset {dataset}")
        sys.exit(1)
    
    # Check if the feature file exists for the specified model and dataset
    feature_file = f"{args.model}/conf_{args.dataset}.npz"
    if not os.path.exists(feature_file):
        print(f"Error: Feature file '{feature_file}' does not exist!")
        print(f"You need to run run.py first with model={args.model} and dataset={args.dataset}")
        sys.exit(1)
    
    # Check for valid dataset
    if args.dataset not in DATASETS:
        raise NotImplementedError(f"Invalid dataset: {args.dataset}")
    
    # Check for valid domain
    if args.domain is None:
        # Set domain to dataset name by default
        args.domain = args.dataset
    elif args.domain != "all" and args.dataset in DOMAINS.keys() and args.domain not in DOMAINS[args.dataset]:
        # For datasets not specified in DOMAINS, use the dataset name as the domain
        if args.dataset not in DOMAINS.keys():
            args.domain = args.dataset
        else:
            raise NotImplementedError(f"Invalid domain: {args.domain}")
    
    # Other argument checks
    if args.model not in MODELS:
        raise NotImplementedError(f"Invalid model: {args.model}")
    if args.experiment not in EXPERIMENTS:
        raise NotImplementedError(f"Invalid experiment : {args.experiment}")
    if args.dataset not in TRUESETS and args.experiment == 'true':
        raise NotImplementedError(f"No ground-truth subsets for {args.dataset}")
        
    # Set default label_set_size if not provided
    if args.label_set_size is None:
        args.label_set_size = DATASET_CONFIG[args.dataset]['num_classes'] if args.dataset in DATASET_CONFIG else 10
        print(f"Using default label set size: {args.label_set_size} for dataset: {args.dataset}")
        
    # Set default data_dir path based on the dataset
    if args.data_dir == data_dir:  # If using default value
        args.data_dir = get_dataset_path(args.dataset)


if __name__ == '__main__':
    # Parse arguments first
    args, unknown_args = parser.parse_known_args()
    
    # Print available models and datasets if requested
    if args.list_available:
        print("\nAvailable models and datasets:")
        print("============================")
        for model in sorted(available_features["models"]):
            print(f"Model: {model}")
            datasets = sorted(available_features["datasets"].get(model, []))
            for dataset in datasets:
                num_classes = DATASET_CONFIG[dataset]['num_classes'] if dataset in DATASET_CONFIG else "unknown"
                print(f"  - Dataset: {dataset} ({num_classes} classes)")
        print("\nUsage example: python zshot.py --model ClipViTB32 --dataset cub")
        print("\nYou can also set USER_MODEL and USER_DATASET variables at the top of this file.")
        sys.exit(0)
    
    # If no command line args were provided, use the hardcoded settings
    if args.model is None and USER_MODEL is not None:
        args.model = USER_MODEL
        print(f"Using model from file configuration: {args.model}")
    
    if args.dataset is None and USER_DATASET is not None:
        args.dataset = USER_DATASET
        print(f"Using dataset from file configuration: {args.dataset}")
    
    if args.domain is None and USER_DOMAIN is not None:
        args.domain = USER_DOMAIN
        print(f"Using domain from file configuration: {args.domain}")
    
    # Run multiple models and datasets if specified
    if RUN_MULTIPLE:
        all_results = {}
        print("\nRunning multiple models and datasets as specified in the configuration")
        
        for model in MODELS_TO_RUN:
            if model not in available_features["models"]:
                print(f"Warning: Skipping model {model} - not available")
                continue
                
            for dataset in DATASETS_TO_RUN:
                if dataset not in available_features["datasets"].get(model, []):
                    print(f"Warning: Skipping dataset {dataset} for model {model} - combination not available")
                    continue
                    
                # Create a new args object with these settings
                current_args = argparse.Namespace(**vars(args))
                current_args.model = model
                current_args.dataset = dataset
                if current_args.domain is None:
                    current_args.domain = dataset
                
                print(f"\n=== Running {model} on {dataset} ===")
                
                # Check these specific arguments
                try:
                    check_args(current_args)
                    
                    # Make sure we have complete args
                    current_args = parser.parse_args(namespace=current_args)
                    
                    print(f"\nSelected configuration:")
                    print(f"  Model: {current_args.model}")
                    print(f"  Dataset: {current_args.dataset}")
                    print(f"  Domain: {current_args.domain}")
                    print(f"  Data directory: {current_args.data_dir}\n")
                    
                    # Run the experiment
                    mod = eval(current_args.model)()
                    
                    # Get the appropriate run function
                    if current_args.opt_templates:
                        run_func = runBestTemplate
                    else:
                        run_func = run
                    
                    if current_args.domain == 'all' and current_args.dataset in DOMAINS.keys():
                        # Handle all domains case
                        domains = DOMAINS[current_args.dataset]
                        out = {}
                        for domain in domains:
                            current_args.domain = domain
                            features, labels, sub2super, super_classes, indices = get_inputs(current_args)
                            out_dom, _ = run_func(mod, features, labels, sub2super, super_classes, indices, current_args)
                            current_args.rerun_gpt = False
                            if current_args.opt_templates:
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
                    else:
                        features, labels, sub2super, super_classes, indices = get_inputs(current_args)
                        out, preds_d = run_func(mod, features, labels, sub2super, super_classes, indices, current_args)
                    
                    # Store results
                    key = f"{model}_{dataset}"
                    all_results[key] = out
                    
                    # Save individual result
                    if not os.path.exists(current_args.out_dir):
                        os.mkdir(current_args.out_dir)
                    pref = f"/{current_args.dataset}-{current_args.model}-{current_args.experiment}-{current_args.label_set_size}-{current_args.reweighter}"
                    if current_args.superclass_set_ens:
                        pref = f"/{current_args.dataset}-{current_args.model}-{current_args.experiment}-{current_args.label_set_size}-{current_args.reweighter}-sse"
                    with open(current_args.out_dir + pref + f".json", 'w') as f:
                        json.dump(out, f, indent=2)
                    if current_args.domain != 'all':
                        with open(current_args.out_dir + pref + f"-outputs.pkl", 'wb') as f:
                            pk.dump(preds_d, f)
                    
                    print(f"Results for {model} on {dataset}:")
                    print(out)
                    
                except Exception as e:
                    print(f"Error running {model} on {dataset}: {e}")
        
        # Save combined results
        combined_path = os.path.join(args.out_dir, "combined_results.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {combined_path}")
        
    else:
        # Regular execution with single model and dataset
        # Check arguments and provide helpful messages
        check_args(args)
        
        # Parse all arguments now that we've set defaults
        args = parser.parse_args()

        print(f"\nSelected configuration:")
        print(f"  Model: {args.model}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Domain: {args.domain}")
        print(f"  Label set size: {args.label_set_size}")
        print(f"  Data directory: {args.data_dir}\n")
        
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
            check_args(args)
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



