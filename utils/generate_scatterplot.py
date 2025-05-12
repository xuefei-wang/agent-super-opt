import json
import matplotlib.pyplot as plt

def plot_scatterplot(output_json_path, baseline_val, baseline_test, scatter_output_path):
    x_vals = []
    y_vals = []
    with open(output_json_path, 'r') as f:
        functions = json.load(f)

    for func in functions:
        val_adv = func['combined_val'] - baseline_val
        test_adv = func['combined_test'] - baseline_test
        x_vals.append(val_adv)
        y_vals.append(test_adv)
    

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals, alpha=0.7)

    plt.xlabel('Advantage over Validation Baseline')
    plt.ylabel('Advantage over Test Baseline')
    plt.title('Scatterplot: Preprocessing Function Improvements')

    # Add reference lines
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)

    # Diagonal reference line: test = val
    grid_size = max(abs(min(x_vals)), abs(max(x_vals)), abs(min(y_vals)), abs(max(y_vals))) + 0.01
    plt.plot([-grid_size, grid_size], [-grid_size, grid_size], 'r--', label='Val = Test')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # Axis limits
    plt.xlim([-grid_size, grid_size])
    plt.ylim([-grid_size, grid_size])

    # Save the figure
    plt.savefig(scatter_output_path)
    print(f"Plot saved to {scatter_output_path}")
