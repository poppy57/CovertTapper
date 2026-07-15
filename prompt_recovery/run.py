#!/usr/bin/env python3
"""
Prompt Inversion via Hidden State Matching

Main entry point for a single experiment

Usage examples:
    # Use the default configuration
    python run.py

    # Specify the model and dataset
    python run.py --model llama2-7b --dataset skytrax

    # Use a configuration file
    python run.py --config config/default.yaml

    # Quick test
    python run.py --max_samples 10 --verbose
    
    # Change the target layer
    python run.py --target_layer 16
"""

import os
import json
import argparse
import logging
from datetime import datetime

# HuggingFace authentication is read from the HF_TOKEN environment variable.

from config import Config, ModelConfig, SearchConfig, DataConfig, ExperimentConfig
from models import ModelLoader, setup_hf_auth, HiddenStateExtractor
from methods import ColdstartSearch
from data import get_dataset_loader


def setup_logging(log_dir: str = "logs", name: str = "prompt_inversion") -> logging.Logger:
    """Set up logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(name)
    logger.info(f"Log file: {log_file}")
    return logger


def run_experiment(
    config: Config,
    logger: logging.Logger,
    verbose: bool = True,
) -> dict:
    """
    Run a single experiment
    
    Args:
        config: configuration object
        logger: logger
        verbose: whether to print detailed logs
        
    Returns:
        experiment result dictionary
    """
    logger.info("=" * 70)
    logger.info("Prompt Inversion Experiment")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model.name} (layer: {config.model.target_layer})")
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Search configuration: K_prior={config.search.K_prior}, K_embed={config.search.K_embed}, beam_size={config.search.beam_size}")
    
    # Set up HuggingFace authentication
    setup_hf_auth()
    
    # Load model
    model_loader = ModelLoader(
        model_path=config.model.model_path,
        torch_dtype=config.model.torch_dtype,
        device_map=config.model.device_map,
    )
    model, tokenizer = model_loader.load()
    
    # Create the hidden-state extractor
    target_layer = config.model.get_target_layer(model_loader.num_layers)
    extractor = HiddenStateExtractor(
        model=model,
        tokenizer=tokenizer,
        target_layer=target_layer,
    )
    
    # Create the searcher
    searcher = ColdstartSearch(
        extractor=extractor,
        config=config.search,
    )
    
    # Load dataset
    dataset = get_dataset_loader(
        dataset_name=config.data.dataset,
        data_path=config.data.data_path,
        text_column=config.data.text_column,
        max_samples=config.data.max_samples,
        data_dir="data_cache",
    ).load()
    
    logger.info(f"Dataset samples: {len(dataset)}")
    
    # Store results
    all_results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(dataset),
        'samples': [],
        'summary': {}
    }
    
    total_tokens = 0
    total_correct = 0
    total_oracle_correct = 0  # Oracle accuracyStatistics
    start_time = datetime.now()
    
    # Process samples one by one
    for sample in dataset:
        logger.info(f"\n{'='*70}")
        logger.info(f"[Sample {sample.idx}]")
        if 'airline_name' in sample.metadata:
            logger.info(f"Airline: {sample.metadata['airline_name']}")
        logger.info(f"{'='*70}")
        
        try:
            # Tokenize
            tokens = tokenizer.encode(sample.text, add_special_tokens=False)
            if config.data.max_length is not None:
                tokens = tokens[:config.data.max_length]
            
            # Search and return a SearchOutput object
            search_output = searcher.search(tokens, verbose=verbose)
            
            # Get the best beam result
            if search_output.beam_results:
                best_beam = search_output.beam_results[0]
                
                # Record all beam results
                beam_results_data = [
                    {
                        'rank': br.rank,
                        'tokens': br.tokens,
                        'text': br.text,
                        'score': br.score,
                        'accuracy': br.token_accuracy,
                        'avg_distance': br.avg_distance,
                        'avg_log_prob': br.avg_log_prob,
                    }
                    for br in search_output.beam_results
                ]
                
                sample_result = {
                    'sample_idx': int(sample.idx),
                    'metadata': sample.metadata,
                    'original_text': sample.text,
                    'recovered_text': best_beam.text,
                    'accuracy': best_beam.token_accuracy,
                    'oracle_accuracy': search_output.oracle_accuracy,
                    'total_tokens': len(tokens),
                    'correct_tokens': int(best_beam.token_accuracy * len(tokens)),
                    'beam_results': beam_results_data,
                    'token_details': best_beam.token_details if config.experiment.save_detailed_results else None,
                }
            else:
                sample_result = {
                    'sample_idx': int(sample.idx),
                    'metadata': sample.metadata,
                    'original_text': sample.text,
                    'recovered_text': '',
                    'accuracy': 0.0,
                    'oracle_accuracy': 0.0,
                    'total_tokens': len(tokens),
                    'correct_tokens': 0,
                    'beam_results': [],
                }
            
            all_results['samples'].append(sample_result)
            total_tokens += sample_result['total_tokens']
            total_correct += sample_result['correct_tokens']
            total_oracle_correct += int(sample_result['oracle_accuracy'] * sample_result['total_tokens'])
            
            logger.info(f"[Sample {sample.idx}] completed: Best={sample_result['accuracy']*100:.1f}%, Oracle={sample_result['oracle_accuracy']*100:.1f}%")
            
        except Exception as e:
            logger.error(f"[Sample {sample.idx}] failed: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results['samples'].append({
                'sample_idx': int(sample.idx),
                'error': str(e)
            })
    
    # Summary statistics
    elapsed_time = (datetime.now() - start_time).total_seconds()
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    oracle_accuracy = total_oracle_correct / total_tokens if total_tokens > 0 else 0
    
    all_results['summary'] = {
        'total_samples_processed': len([s for s in all_results['samples'] if 'accuracy' in s]),
        'total_tokens': total_tokens,
        'total_correct': total_correct,
        'overall_accuracy': overall_accuracy,
        'oracle_accuracy': oracle_accuracy,
        'beam_size': config.search.beam_size,
        'elapsed_time_seconds': elapsed_time,
        'tokens_per_second': total_tokens / elapsed_time if elapsed_time > 0 else 0
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("completed")
    logger.info("=" * 70)
    logger.info(f"Sample: {all_results['summary']['total_samples_processed']}")
    logger.info(f"Beam Size: {config.search.beam_size}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Best Beam accuracy: {overall_accuracy*100:.2f}%")
    logger.info(f"Oracle accuracy (any beam): {oracle_accuracy*100:.2f}%")
    logger.info(f"Elapsed time: {elapsed_time:.1f}s")
    logger.info(f"Throughput: {all_results['summary']['tokens_per_second']:.2f} tokens/s")
    
    # Save results
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        config.experiment.output_dir,
        f"results_{config.model.name}_{timestamp}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Prompt Inversion via Hidden State Matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run.py --model llama2-7b --dataset skytrax
  
  # Quick test
  python run.py --max_samples 5 --verbose
  
  # Use a configuration file
  python run.py --config config/default.yaml
  
  # Change the target layer
  python run.py --target_layer 16
        """
    )
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                       help='Config (YAML)')
    
    # ModelConfig
    parser.add_argument('--model', type=str, default='llama2-7b',
                       choices=list(ModelConfig.SUPPORTED_MODELS.keys()),
                       help='Model')
    parser.add_argument('--target_layer', type=int, default=-1,
                       help='Target layer (-1=final layer)')
    
    # Search configuration
    parser.add_argument('--K_prior', type=int, default=1000,
                       help='language-prior candidate pool size')
    parser.add_argument('--K_embed', type=int, default=0,
                       help='embedding-similarity candidate pool size (0=disabled)')
    parser.add_argument('--beam_size', type=int, default=1,
                       help='Beam Size (1=greedy search)')
    parser.add_argument('--lambda_prior', type=float, default=0,
                       help='language-prior weight')
    
    # Data configuration
    parser.add_argument('--dataset', type=str, default='skytrax',
                       help='Dataset')
    parser.add_argument('--data_path', type=str, default=None,
                       help='custom data path')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples')
    parser.add_argument('--max_length', type=int, default=None,
                       help='maximum token length')
    
    # Experiment configuration
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(log_dir=args.log_dir)
    
    # Config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config(
            model=ModelConfig(
                name=args.model,
                target_layer=args.target_layer,
            ),
            search=SearchConfig(
                K_prior=args.K_prior,
                K_embed=args.K_embed,
                beam_size=args.beam_size,
                lambda_prior=args.lambda_prior,
            ),
            data=DataConfig(
                dataset=args.dataset,
                data_path=args.data_path,
                max_samples=args.max_samples,
                max_length=args.max_length,
            ),
            experiment=ExperimentConfig(
                output_dir=args.output_dir,
                log_dir=args.log_dir,
            ),
        )
    
    # Run experiment
    run_experiment(config, logger, verbose=args.verbose)


if __name__ == "__main__":
    main()
