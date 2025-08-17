"""
DVC Experiment Management Script
Demonstrates how to run and track experiments using DVC.
"""

import os
import argparse
from experiments.experiment_tracker import DVCExperimentTracker, create_hyperparameter_sweep, run_model_comparison
from src.utils.metrics_logger import create_metrics_logger_from_config

def run_single_experiment():
    """Run a single experiment with custom configuration."""
    tracker = DVCExperimentTracker()
    
    # Create experiment with custom parameters
    config_overrides = {
        'training': {
            'episodes': 1500,
            'learning_rate': 0.0005,
            'batch_size': 64
        },
        'multi_agent': {
            'agent_types': ['sac', 'sac', 'dqn', 'dqn']  # Different agent mix
        }
    }
    
    exp_id = tracker.create_experiment(
        name="custom_multi_agent",
        description="Multi-agent system with mixed SAC/DQN agents",
        tags=["multi_agent", "custom_config", "mixed_agents"],
        config_overrides=config_overrides
    )
    
    print(f"Running experiment: {exp_id}")
    results = tracker.run_experiment(exp_id)
    
    return exp_id, results

def run_hyperparameter_sweep():
    """Run hyperparameter sweep experiments."""
    print("🔍 Starting Hyperparameter Sweep")
    print("=" * 50)
    
    experiment_ids = create_hyperparameter_sweep()
    tracker = DVCExperimentTracker()
    
    results = {}
    for exp_id in experiment_ids:
        print(f"\nRunning experiment: {exp_id}")
        result = tracker.run_experiment(exp_id)
        results[exp_id] = result
    
    # Compare results
    print("\n Comparing Hyperparameter Sweep Results")
    comparison_df = tracker.compare_experiments(
        experiment_ids, 
        metrics=['portfolio_total_return', 'portfolio_sharpe_ratio', 'training_avg_reward']
    )
    
    print(comparison_df.to_string())
    
    # Find best experiment
    best_exp = tracker.get_best_experiment('portfolio_total_return', minimize=False)
    if best_exp:
        print(f"\n🏆 Best experiment: {best_exp}")
    
    return results

def run_architecture_comparison():
    """Compare different model architectures."""
    print("🏗️ Starting Architecture Comparison")
    print("=" * 50)
    
    experiment_ids = run_model_comparison()
    tracker = DVCExperimentTracker()
    
    results = {}
    for exp_id in experiment_ids:
        print(f"\nRunning experiment: {exp_id}")
        result = tracker.run_experiment(exp_id)
        results[exp_id] = result
    
    # Compare architectures
    print("\n Architecture Comparison Results")
    comparison_df = tracker.compare_experiments(
        experiment_ids,
        metrics=['portfolio_total_return', 'portfolio_sharpe_ratio', 'training_final_reward']
    )
    
    print(comparison_df.to_string())
    
    return results

def run_agent_type_comparison():
    """Compare different agent types (DQN vs SAC vs Multi-Agent)."""
    tracker = DVCExperimentTracker()
    
    # DQN-only experiment
    dqn_exp = tracker.create_experiment(
        name="dqn_only",
        description="All agents using DQN",
        tags=["agent_comparison", "dqn_only"],
        config_overrides={
            'multi_agent': {
                'agent_types': ['dqn', 'dqn', 'dqn', 'dqn']
            }
        }
    )
    
    # SAC-only experiment
    sac_exp = tracker.create_experiment(
        name="sac_only", 
        description="All agents using SAC",
        tags=["agent_comparison", "sac_only"],
        config_overrides={
            'multi_agent': {
                'agent_types': ['sac', 'sac', 'sac', 'sac']
            }
        }
    )
    
    # Mixed agents experiment
    mixed_exp = tracker.create_experiment(
        name="mixed_agents",
        description="Mixed DQN and SAC agents",
        tags=["agent_comparison", "mixed_agents"],
        config_overrides={
            'multi_agent': {
                'agent_types': ['dqn', 'sac', 'dqn', 'sac']
            }
        }
    )
    
    experiment_ids = [dqn_exp, sac_exp, mixed_exp]
    
    # Run experiments
    results = {}
    for exp_id in experiment_ids:
        print(f"\nRunning {exp_id}...")
        result = tracker.run_experiment(exp_id)
        results[exp_id] = result
    
    # Compare results
    print("\n📊 Agent Type Comparison")
    comparison_df = tracker.compare_experiments(
        experiment_ids,
        metrics=['portfolio_total_return', 'portfolio_sharpe_ratio', 'allocation_entropy']
    )
    
    print(comparison_df.to_string())
    
    return results

def analyze_experiments():
    """Analyze all completed experiments."""
    tracker = DVCExperimentTracker()
    all_experiments = tracker._get_all_experiments()
    
    if not all_experiments:
        print("No experiments found.")
        return
    
    print(f"📊 Analyzing {len(all_experiments)} experiments")
    print("=" * 50)
    
    # Get comparison of all experiments
    comparison_df = tracker.compare_experiments(all_experiments)
    
    # Display summary statistics
    print("\nExperiment Summary:")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Completed: {len(comparison_df[comparison_df['status'] == 'completed'])}")
    print(f"Failed: {len(comparison_df[comparison_df['status'] == 'failed'])}")
    print(f"Running: {len(comparison_df[comparison_df['status'] == 'running'])}")
    
    # Show top performers
    numeric_cols = comparison_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print(f"\nTop 5 experiments by portfolio return:")
        if 'portfolio_total_return' in comparison_df.columns:
            top_experiments = comparison_df.nlargest(5, 'portfolio_total_return')
            print(top_experiments[['experiment_id', 'name', 'portfolio_total_return', 'portfolio_sharpe_ratio']].to_string())
    
    # Save analysis
    analysis_file = "experiments/experiment_analysis.csv"
    comparison_df.to_csv(analysis_file, index=False)
    print(f"\n💾 Analysis saved to: {analysis_file}")
    
    return comparison_df

def cleanup_experiments():
    """Archive old or failed experiments."""
    tracker = DVCExperimentTracker()
    all_experiments = tracker._get_all_experiments()
    
    archived_count = 0
    for exp_id in all_experiments:
        exp_dir = os.path.join(tracker.experiments_dir, exp_id)
        metadata_file = os.path.join(exp_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Archive failed experiments older than 7 days
            if metadata.get('status') == 'failed':
                from datetime import datetime, timedelta
                created_at = datetime.fromisoformat(metadata['created_at'])
                if datetime.now() - created_at > timedelta(days=7):
                    tracker.archive_experiment(exp_id)
                    archived_count += 1
    
    print(f"📦 Archived {archived_count} old/failed experiments")

def main():
    parser = argparse.ArgumentParser(description="DVC Experiment Management")
    parser.add_argument('--action', choices=[
        'single', 'sweep', 'architecture', 'agent_comparison', 
        'analyze', 'cleanup'
    ], required=True, help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'single':
        run_single_experiment()
    elif args.action == 'sweep':
        run_hyperparameter_sweep()
    elif args.action == 'architecture':
        run_architecture_comparison()
    elif args.action == 'agent_comparison':
        run_agent_type_comparison()
    elif args.action == 'analyze':
        analyze_experiments()
    elif args.action == 'cleanup':
        cleanup_experiments()

if __name__ == "__main__":
    # Example usage without command line args
    print("🚀 DVC Experiment Management Demo")
    print("=" * 50)
    
    print("\nAvailable actions:")
    print("1. Run single experiment")
    print("2. Run hyperparameter sweep")
    print("3. Compare architectures")
    print("4. Compare agent types")
    print("5. Analyze all experiments")
    print("6. Cleanup old experiments")
    
    choice = input("\nSelect action (1-6): ").strip()
    
    if choice == '1':
        run_single_experiment()
    elif choice == '2':
        run_hyperparameter_sweep()
    elif choice == '3':
        run_architecture_comparison()
    elif choice == '4':
        run_agent_type_comparison()
    elif choice == '5':
        analyze_experiments()
    elif choice == '6':
        cleanup_experiments()
    else:
        print("Invalid choice. Running single experiment as default.")
        run_single_experiment()
