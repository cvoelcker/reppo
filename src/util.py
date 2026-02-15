import jax
import jax.numpy as jnp


def prefix_dict(prefix: str, metrics: dict, sep: str = "/") -> dict:
    """Add a prefix to all keys in a dictionary."""
    return {f"{prefix}{sep}{key}": value for key, value in metrics.items()}


def log_callback(train_state, metrics):
    for k, v in metrics.items():
        if jnp.any(jnp.isnan(v)):
            print(f"Warning: Metric {k} is NaN")
    metrics = jax.tree.map(lambda x: x.mean(0), metrics)

    # Separate train and eval metrics
    train_metrics = {k.replace("train/", ""): v for k, v in metrics.items() if k.startswith("train/")}
    eval_metrics = {k.replace("eval/", ""): v for k, v in metrics.items() if k.startswith("eval/")}

    # Header
    print("\n" + "=" * 100)
    print(f"📊 Iteration {train_state.iteration[0].item():>6} | Timesteps {train_state.time_steps[0].item():>10,}")
    print("=" * 100)

    # Helper function to group metrics by prefix
    def group_by_prefix(metrics_dict):
        groups = {}
        for key, value in metrics_dict.items():
            # Extract prefix (everything before last slash, or use key itself if no slash)
            if "/" in key:
                prefix = key.rsplit("/", 1)[0]  # Everything before the last /
                metric_name = key.rsplit("/", 1)[1]  # Everything after the last /
            else:
                prefix = "other"
                metric_name = key

            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][metric_name] = value
        return groups

    # Training metrics
    if train_metrics:
        print("\n🏋️  TRAINING METRICS:")
        print("-" * 100)

        grouped = group_by_prefix(train_metrics)

        # Sort groups by name for consistent ordering
        for group_name in sorted(grouped.keys()):
            group_metrics = grouped[group_name]
            print(f"  {group_name.upper()}:")
            for k, v in sorted(group_metrics.items()):
                print(f"    {k:<30} {v.item():>10.4f}")
            print()

    # Evaluation metrics
    if eval_metrics:
        print("🎯 EVALUATION METRICS:")
        print("-" * 100)

        grouped = group_by_prefix(eval_metrics)

        # Sort groups by name for consistent ordering
        for group_name in sorted(grouped.keys()):
            group_metrics = grouped[group_name]
            print(f"  {group_name.upper()}:")
            for k, v in sorted(group_metrics.items()):
                print(f"    {k:<30} {v.item():>10.4f}")
            print()

    print("=" * 100)
