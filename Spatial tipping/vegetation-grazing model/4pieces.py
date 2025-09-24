import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def f(V, r, V_c, c):
    """Reaction term function"""
    return r * V * (1 - V / V_c) - c * V ** 2 / (V ** 2 + 1)


def f_prime(V, r, V_c, c):
    """Derivative of reaction term"""
    return r * (1 - 2 * V / V_c) - 2 * c * V / (V ** 2 + 1) ** 2


def checkerboard_eq(V, r, V_c, c, D, h):
    """Checkerboard equilibrium equations: V11=V22=Va, V12=V21=Vb"""
    Va, Vb = V
    beta = D / h ** 2
    eq1 = f(Va, r, V_c, c) + 2 * beta * (Vb - Va)
    eq2 = f(Vb, r, V_c, c) + 2 * beta * (Va - Vb)
    return [eq1, eq2]


def row_sync_eq(V, r, V_c, c, D, h):
    """Row-synchronized equilibrium equations: V11=V12=Vr, V21=V22=Vc"""
    Vr, Vc = V
    beta = D / h ** 2
    eq1 = f(Vr, r, V_c, c) + beta * (Vc - Vr)  # For V11 and V12
    eq2 = f(Vc, r, V_c, c) + beta * (Vr - Vc)  # For V21 and V22
    return [eq1, eq2]


# Parameter settings
r, V_c, c, D, h = 1.0, 10.0, 2.0, 0.1, 1.0

# Create figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. First find synchronous equilibria (no diffusion)
V_range = np.linspace(0, 10, 1000)
f_values = f(V_range, r, V_c, c)

# Find synchronous equilibria by solving f(V)=0
sync_equilibria = []
zero_crossings = np.where(np.diff(np.sign(f_values)))[0]

for i in zero_crossings:
    V_guess = V_range[i]
    V_eq = fsolve(lambda V: f(V, r, V_c, c), V_guess)[0]
    if 0 <= V_eq <= 10:
        sync_equilibria.append(V_eq)

sync_equilibria = sorted(sync_equilibria)
print("Synchronous equilibria:", sync_equilibria)

# 2. Use synchronous equilibria as starting points for other patterns
# For checkerboard and row-sync patterns, we expect pairs of values
# that are close to the synchronous equilibria

# Find checkerboard equilibria using synchronous equilibria as starting points
checker_equilibria = []
for V_low in sync_equilibria:
    for V_high in sync_equilibria:
        if V_low < V_high:  # Only consider pairs where low < high
            guess = (V_low, V_high)
            try:
                sol = fsolve(lambda V: checkerboard_eq(V, r, V_c, c, D, h), guess, full_output=True)
                if sol[2] == 1:  # Solution found
                    Va, Vb = sol[0]
                    residual = checkerboard_eq([Va, Vb], r, V_c, c, D, h)
                    if np.allclose(residual, [0, 0], atol=1e-6):
                        checker_equilibria.append((Va, Vb))
            except:
                continue

print("Checkerboard equilibria:", checker_equilibria)

# Find row-synchronized equilibria
row_sync_equilibria = []
for V_low in sync_equilibria:
    for V_high in sync_equilibria:
        if V_low < V_high:  # Only consider pairs where low < high
            guess = (V_low, V_high)
            try:
                sol = fsolve(lambda V: row_sync_eq(V, r, V_c, c, D, h), guess, full_output=True)
                if sol[2] == 1:  # Solution found
                    Vr, Vc = sol[0]
                    residual = row_sync_eq([Vr, Vc], r, V_c, c, D, h)
                    if np.allclose(residual, [0, 0], atol=1e-6):
                        row_sync_equilibria.append((Vr, Vc))
            except:
                continue

print("Row-synchronized equilibria:", row_sync_equilibria)

# 3. Plot f(V)-V for all patterns with consistent coloring
V_range = np.linspace(0, 10, 1000)
f_values = f(V_range, r, V_c, c)

colors = ['red', 'green', 'blue', 'purple', 'orange']
pattern_colors = {'sync': 'red', 'checker': 'green', 'row': 'blue'}

# Plot f(V) curve for all subplots
for i in range(3):
    axes[0, i].plot(V_range, f_values, 'k-', linewidth=2, label='f(V)')
    axes[0, i].axhline(0, color='gray', linestyle='--', alpha=0.7)
    axes[0, i].set_xlabel('V')
    axes[0, i].set_ylabel('f(V)')
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].legend()

# 4. Plot fully synchronized pattern
axes[0, 0].set_title('Fully Synchronized Pattern: f(V) vs V')
for i, V_eq in enumerate(sync_equilibria):
    axes[0, 0].plot(V_eq, 0, 'o', color=colors[i % len(colors)], markersize=8,
                    label=f'V_{i + 1} = {V_eq:.3f}')
axes[0, 0].legend()

# 5. Plot checkerboard pattern
axes[0, 1].set_title('Checkerboard Pattern: f(V) vs V')
if checker_equilibria:
    for i, (Va, Vb) in enumerate(checker_equilibria):
        # Plot the actual f(V) values at these points
        axes[0, 1].plot(Va, f(Va, r, V_c, c), 's', color=colors[i % len(colors)],
                        markersize=8, label=f'f(V_a={Va:.3f})')
        axes[0, 1].plot(Vb, f(Vb, r, V_c, c), '^', color=colors[i % len(colors)],
                        markersize=8, label=f'f(V_b={Vb:.3f})')

        # Plot the modified values (should be zero at equilibrium)
        beta = D / h ** 2
        mod_fa = f(Va, r, V_c, c) + 2 * beta * (Vb - Va)
        mod_fb = f(Vb, r, V_c, c) + 2 * beta * (Va - Vb)

        axes[0, 1].plot(Va, mod_fa, 'x', color=colors[i % len(colors)],
                        markersize=10, markeredgewidth=2, label=f'mod f(V_a)={mod_fa:.3f}')
        axes[0, 1].plot(Vb, mod_fb, 'x', color=colors[i % len(colors)],
                        markersize=10, markeredgewidth=2, label=f'mod f(V_b)={mod_fb:.3f}')

        # Add vertical lines
        axes[0, 1].axvline(Va, color=colors[i % len(colors)], linestyle=':', alpha=0.5)
        axes[0, 1].axvline(Vb, color=colors[i % len(colors)], linestyle=':', alpha=0.5)

    axes[0, 1].legend(fontsize=8)
else:
    axes[0, 1].text(0.5, 0.5, 'No checkerboard equilibria found',
                    ha='center', va='center', transform=axes[0, 1].transAxes)

# 6. Plot row-synchronized pattern
axes[0, 2].set_title('Row-Synchronized Pattern: f(V) vs V')
if row_sync_equilibria:
    for i, (Vr, Vc) in enumerate(row_sync_equilibria):
        # Plot the actual f(V) values at these points
        axes[0, 2].plot(Vr, f(Vr, r, V_c, c), 's', color=colors[i % len(colors)],
                        markersize=8, label=f'f(V_r={Vr:.3f})')
        axes[0, 2].plot(Vc, f(Vc, r, V_c, c), '^', color=colors[i % len(colors)],
                        markersize=8, label=f'f(V_c={Vc:.3f})')

        # Plot the modified values (should be zero at equilibrium)
        beta = D / h ** 2
        mod_fr = f(Vr, r, V_c, c) + beta * (Vc - Vr)
        mod_fc = f(Vc, r, V_c, c) + beta * (Vr - Vc)

        axes[0, 2].plot(Vr, mod_fr, 'x', color=colors[i % len(colors)],
                        markersize=10, markeredgewidth=2, label=f'mod f(V_r)={mod_fr:.3f}')
        axes[0, 2].plot(Vc, mod_fc, 'x', color=colors[i % len(colors)],
                        markersize=10, markeredgewidth=2, label=f'mod f(V_c)={mod_fc:.3f}')

        # Add vertical lines
        axes[0, 2].axvline(Vr, color=colors[i % len(colors)], linestyle=':', alpha=0.5)
        axes[0, 2].axvline(Vc, color=colors[i % len(colors)], linestyle=':', alpha=0.5)

    axes[0, 2].legend(fontsize=8)
else:
    axes[0, 2].text(0.5, 0.5, 'No row-synchronized equilibria found',
                    ha='center', va='center', transform=axes[0, 2].transAxes)

# 7. Plot pattern visualizations with consistent coloring
patterns_data = [
    ("Fully Synchronized", [(sync_equilibria[0], sync_equilibria[0])] if sync_equilibria else [], 'viridis'),
    ("Checkerboard", checker_equilibria, 'plasma'),
    ("Row-Synchronized", row_sync_equilibria, 'inferno')
]

for i, (pattern_name, pattern_list, cmap_name) in enumerate(patterns_data):
    if not pattern_list:
        axes[1, i].text(0.5, 0.5, f'No {pattern_name.lower()} equilibria found',
                        ha='center', va='center', transform=axes[1, i].transAxes)
        axes[1, i].set_title(f'{pattern_name} Pattern')
        continue

    # Use the first solution for visualization
    if pattern_name == "Fully Synchronized":
        V_val, _ = pattern_list[0]
        grid = np.array([[V_val, V_val], [V_val, V_val]])
        title = f'{pattern_name}: V = {V_val:.3f}'
    elif pattern_name == "Checkerboard":
        Va, Vb = pattern_list[0]
        grid = np.array([[Va, Vb], [Vb, Va]])
        title = f'{pattern_name}: V_a={Va:.3f}, V_b={Vb:.3f}'
    else:
        Vr, Vc = pattern_list[0]
        grid = np.array([[Vr, Vr], [Vc, Vc]])
        title = f'{pattern_name}: V_r={Vr:.3f}, V_c={Vc:.3f}'

    # Create consistent color mapping
    vmin, vmax = 0, max(10, np.max(grid) * 1.1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    im = axes[1, i].imshow(grid, cmap=cmap, norm=norm, origin='upper')

    # Add value labels with appropriate color contrast
    for row in range(2):
        for col in range(2):
            text_color = 'white' if grid[row, col] > (vmin + vmax) / 2 else 'black'
            axes[1, i].text(col, row, f'{grid[row, col]:.3f}',
                            ha="center", va="center", color=text_color, fontweight='bold')

    axes[1, i].set_title(title)
    axes[1, i].set_xticks([0, 1])
    axes[1, i].set_yticks([0, 1])
    axes[1, i].set_xticklabels(['1', '2'])
    axes[1, i].set_yticklabels(['1', '2'])
    axes[1, i].set_xlabel('x')
    axes[1, i].set_ylabel('y')

    plt.colorbar(im, ax=axes[1, i], label='V value')

# 8. Plot stability analysis
for i, pattern_name in enumerate(["Fully Synchronized", "Checkerboard", "Row-Synchronized"]):
    axes[2, i].plot(V_range, f_values, 'k-', linewidth=2, label='f(V)')
    axes[2, i].axhline(0, color='gray', linestyle='--', alpha=0.7)
    axes[2, i].set_xlabel('V')
    axes[2, i].set_ylabel('f(V)')
    axes[2, i].set_title(f'{pattern_name}: Stability Overview')
    axes[2, i].grid(True, alpha=0.3)

    # Mark all found equilibria
    if pattern_name == "Fully Synchronized" and sync_equilibria:
        for j, V_eq in enumerate(sync_equilibria):
            stability = "stable" if f_prime(V_eq, r, V_c, c) < 0 else "unstable"
            axes[2, i].plot(V_eq, 0, 'o', color=colors[j % len(colors)], markersize=8,
                            label=f'V={V_eq:.3f} ({stability})')

    elif pattern_name == "Checkerboard" and checker_equilibria:
        for j, (Va, Vb) in enumerate(checker_equilibria):
            axes[2, i].plot(Va, f(Va, r, V_c, c), 's', color=colors[j % len(colors)],
                            markersize=8, label=f'V_a={Va:.3f}')
            axes[2, i].plot(Vb, f(Vb, r, V_c, c), '^', color=colors[j % len(colors)],
                            markersize=8, label=f'V_b={Vb:.3f}')

    elif pattern_name == "Row-Synchronized" and row_sync_equilibria:
        for j, (Vr, Vc) in enumerate(row_sync_equilibria):
            axes[2, i].plot(Vr, f(Vr, r, V_c, c), 's', color=colors[j % len(colors)],
                            markersize=8, label=f'V_r={Vr:.3f}')
            axes[2, i].plot(Vc, f(Vc, r, V_c, c), '^', color=colors[j % len(colors)],
                            markersize=8, label=f'V_c={Vc:.3f}')

    axes[2, i].legend(fontsize=8)

plt.suptitle('Corrected Analysis of 2×2 Grid Reaction-Diffusion System', fontsize=16)
plt.savefig('corrected_reaction_diffusion_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Output detailed results
print("\n=== CORRECTED ANALYSIS RESULTS ===")
print(f"Parameters: r={r}, V_c={V_c}, c={c}, D={D}, h={h}")

print("\nSynchronous equilibria (f(V)=0):")
for i, V_eq in enumerate(sync_equilibria):
    stability = "stable" if f_prime(V_eq, r, V_c, c) < 0 else "unstable"
    print(f"  V_{i + 1} = {V_eq:.4f} ({stability}), f'(V) = {f_prime(V_eq, r, V_c, c):.4f}")

print("\nCheckerboard equilibria (f(V) + diffusion = 0):")
for i, (Va, Vb) in enumerate(checker_equilibria):
    residual = checkerboard_eq([Va, Vb], r, V_c, c, D, h)
    print(f"  Solution {i + 1}: V_a = {Va:.4f}, V_b = {Vb:.4f}")
    print(f"    Residual: [{residual[0]:.6f}, {residual[1]:.6f}]")
    print(f"    f(V_a) = {f(Va, r, V_c, c):.4f}, f(V_b) = {f(Vb, r, V_c, c):.4f}")

print("\nRow-synchronized equilibria (f(V) + diffusion = 0):")
for i, (Vr, Vc) in enumerate(row_sync_equilibria):
    residual = row_sync_eq([Vr, Vc], r, V_c, c, D, h)
    print(f"  Solution {i + 1}: V_r = {Vr:.4f}, V_c = {Vc:.4f}")
    print(f"    Residual: [{residual[0]:.6f}, {residual[1]:.6f}]")
    print(f"    f(V_r) = {f(Vr, r, V_c, c):.4f}, f(V_c) = {f(Vc, r, V_c, c):.4f}")