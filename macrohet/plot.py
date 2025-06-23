import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

# Colour scheme
expanded_piyg = ['#1a9641', '#a6d96a', '#978897', '#d1d1ca', '#f1b6da', '#d02c91']

def single_cell_growth(df, ID):
    """
    Plot the intracellular Mtb growth dynamics for a single cell using LOWESS fit and annotated doubling times.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns: 'ID', 'Time Model (hours)', 'Mtb Area Model (µm)',
        'Mtb Area Processed (µm)', 'Strain', 'Compound', 'Concentration'.

    ID : str
        Unique ID of the single cell to be plotted (e.g., "1.3.5.PS0000")
    """

    sc_df = df[df['ID'] == ID].dropna(subset=[
        'Time Model (hours)', 'Mtb Area Model (µm)', 'Mtb Area Processed (µm)'])
    sc_df = sc_df.sort_values(by='Time Model (hours)')

    if sc_df.empty:
        print(f"No data found for ID {ID}")
        return

    strain, compound, concentration = sc_df[['Strain', 'Compound', 'Concentration']].iloc[0]
    r2 = round(r2_score(sc_df['Mtb Area Processed (µm)'], sc_df['Mtb Area Model (µm)']), 2)

    min_value = max(sc_df['Mtb Area Model (µm)'].round(1).min(), 1.92)
    max_value = sc_df['Mtb Area Model (µm)'].round(1).max()

    min_idx = sc_df['Mtb Area Model (µm)'].idxmin()
    max_idx = sc_df['Mtb Area Model (µm)'].idxmax()

    N_series = []
    if max_idx > min_idx:
        N_i = min_value
        while N_i <= max_value:
            N_series.append(N_i)
            N_i *= 2
    else:
        N_i = max_value
        while N_i >= min_value:
            N_series.append(N_i)
            N_i /= 2

    if len(N_series) < 2:
        print(f"No population doubling for ID {ID}")
        return

    doubling_time_indices = np.clip(
        np.searchsorted(sc_df['Mtb Area Model (µm)'], N_series), 0, len(sc_df) - 1)
    doubling_time_points = sc_df['Time Model (hours)'].iloc[doubling_time_indices]
    doubling_times = doubling_time_points.diff().dropna().values.tolist()

    plt.figure(figsize=(8, 6))
    plt.plot(sc_df['Time Model (hours)'], sc_df['Mtb Area Model (µm)'], label='LOWESS Fit', color=expanded_piyg[-2])
    plt.scatter(sc_df['Time Model (hours)'], sc_df['Mtb Area Processed (µm)'], label='Processed Mtb Signal', color=expanded_piyg[0], s=5)

    for i, (time, population) in enumerate(zip(doubling_time_points, N_series)):
        time = abs(time)
        population = abs(population)
        color = expanded_piyg[i % len(expanded_piyg)]
        plt.axvline(x=time, color=color, linestyle='--', linewidth=1, alpha=0.5)
        plt.axhline(y=population, color=color, linestyle='--', linewidth=1, alpha=0.5)

        if i > 0:
            previous_time = doubling_time_points.iloc[i - 1]
            gap = time - previous_time
            fontsize = max(8, min(12, gap))
            plt.text(time, population + population * 0.01, f'{abs(doubling_times[i - 1]):.2f}h',
                     ha='right', va='bottom', fontsize=fontsize, color=color, alpha=0.5)
            deltaMtb = population - N_series[i - 1]
            plt.text(time + 0.1, population - max_value * 0.01, f'Δ {abs(deltaMtb):.1f}µm²',
                     fontsize=fontsize, color=color, ha='left', va='top', rotation=270, alpha=0.5)

    plt.xlabel('Time (Hours)')
    plt.ylabel('Mtb Area µm$^2$')
    plt.title(f'Cell ID {ID} | Strain: {strain} | Compound: {compound} | Concentration: {concentration} | R² = {r2}', fontsize=12)
    plt.suptitle('Single-Macrophage Intracellular Mtb Dynamics', weight='bold', fontsize=16)
    plt.legend()
    sns.despine(offset=10)
    plt.tight_layout()
    plt.show()


def population_dynamics_coloured(df, strain='WT', compound='CTRL', origin_filter='Junk'):
    """
    Plot all intracellular single-cell Mtb growth dynamics as a series of line graphs
    color-coded by final Mtb burden.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'ID', 'Strain', 'Compound', 'mtb_origin',
        'Time Model (hours)', and 'Mtb Area Model (µm)'.

    strain : str
        Strain filter (default 'WT').

    compound : str
        Compound filter (default 'CTRL').

    origin_filter : str
        Exclude rows with this mtb_origin label (default 'Junk').
    """

    # Constants for figure size
    fig_width_mm = 122.365
    fig_height_mm = 42
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_height_mm / 25.4

    # Define the colormap using 'PiYG_r'
    cmap = plt.get_cmap('PiYG_r')

    # Subset DataFrame
    subset_df = df[(df['Compound'] == compound)
                   & (df['Strain'] == strain)
                   & (df['mtb_origin'] != origin_filter)]

    # Create the plot
    plt.figure(figsize=(fig_width_inch, fig_height_inch))

    for ID in tqdm(subset_df['ID'].unique(), total=subset_df['ID'].nunique()):
        sc_df = subset_df[subset_df['ID'] == ID]
        time_model = sc_df['Time Model (hours)'].dropna().values
        population_model = sc_df['Mtb Area Model (µm)'].dropna().values

        if len(time_model) == 0 or len(population_model) == 0:
            continue

        max_value = np.nanmax(population_model)
        if max_value < 4 or max_value > 300:
            continue

        colour_value = population_model[-1]  # Final value
        norm = mpl.colors.Normalize(vmin=0, vmax=300)
        color = cmap(norm(colour_value))

        plt.plot(time_model, population_model, color=color, alpha=0.2)

    # Final styling
    plt.xlabel('Time (hours)')
    plt.ylabel('Mtb Area (µm$^2$)')
    plt.xlim(5, 74)
    plt.ylim(bottom=0)
    sns.despine(offset=5)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.02, aspect=10, shrink=0.9)
    cbar.set_label('Final Mtb Load (µm$^2$)')
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.show()



def doubling_times_boxplot(subset_df, expanded_piyg, width_mm=280, height_mm=100):
    """
    Plot a boxplot of doubling times and a horizontal bar chart of growth category proportions per strain.

    Parameters
    ----------
    subset_df : pd.DataFrame
        Must contain 'Doubling Times', 'Strain', and any filters already applied.
    expanded_piyg : list
        List of hex color codes to use for plotting.
    width_mm : float
        Width of the figure in millimetres. Default is 280mm.
    height_mm : float
        Height of the figure in millimetres. Default is 100mm.
    """

    # Convert mm to inches
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    # Bin doubling times
    bins = [0, 16, 24, float('inf')]
    bin_labels = ['Fast', 'Normal', 'Slow']
    subset_df['Growth Category'] = pd.cut(subset_df['Doubling Times'], bins=bins, labels=bin_labels, right=False)

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in / 1.25),
                             gridspec_kw={'width_ratios': [5, 1]})
    ax_boxplot, ax_barchart = axes

    # Boxplot + stripplot
    sns.boxplot(
        data=subset_df, x="Doubling Times", y="Strain", hue="Strain",
        whis=[0, 100], width=.6, palette=[expanded_piyg[-2], expanded_piyg[1]],
        ax=ax_boxplot, legend=False
    )
    sns.stripplot(
        data=subset_df, x="Doubling Times", y="Strain", size=4, hue="Strain",
        palette=[expanded_piyg[-1], expanded_piyg[0]], alpha=0.5,
        edgecolor='gray', linewidth=1, jitter=0.3,
        ax=ax_boxplot
    )

    # Shaded regions and annotations
    ax_boxplot.axvspan(0, 16, color=expanded_piyg[3], alpha=0.0)
    ax_boxplot.axvspan(16, 24, color=expanded_piyg[3], alpha=0.8)
    ax_boxplot.axvspan(24, 70, color=expanded_piyg[3], alpha=0.0)

    ax_boxplot.text(0, -0.35, 'Fast', fontsize=48, color='gray', alpha=0.2, ha='left', rotation=90)
    ax_boxplot.text(16, -0.35, 'Normal', fontsize=48, color='white', alpha=0.4, ha='left', rotation=90)
    ax_boxplot.text(70.5, -0.35, 'Slow', fontsize=48, color='gray', alpha=0.2, ha='right', rotation=90)

    ax_boxplot.set_xlim(-2, 75)
    ax_boxplot.set_ylim(-0.5, 1.5)
    ax_boxplot.set_xlabel('Doubling Times (hours)')
    sns.despine(ax=ax_boxplot, offset=10, left=True)

    # Calculate proportions
    total_counts = subset_df.groupby(['Strain', 'Growth Category'], observed=True).size()
    strain_totals = subset_df.groupby('Strain', observed=True).size()

    percentage_df = []
    for strain_val in subset_df['Strain'].unique():
        for category_val in bin_labels:
            count = total_counts.get((strain_val, category_val), 0)
            total = strain_totals.get(strain_val, 1)
            percentage_df.append({
                'Strain': strain_val,
                'Growth Category': category_val,
                'Percentage': (count / total) * 100
            })
    percentage_df = pd.DataFrame(percentage_df)
    pivot = percentage_df.pivot(index='Strain', columns='Growth Category', values='Percentage').fillna(0)
    pivot = pivot[bin_labels]

    # Align order with boxplot if possible
    boxplot_strains = [label.get_text() for label in ax_boxplot.get_yticklabels()]
    ordered_strains = [s for s in boxplot_strains if s in pivot.index]
    if set(ordered_strains) == set(pivot.index):
        pivot = pivot.reindex(ordered_strains)

    # Bar chart
    bar_colors = [expanded_piyg[1], expanded_piyg[3], expanded_piyg[-2]]
    pivot.plot(kind='barh', stacked=True, ax=ax_barchart, color=bar_colors, width=0.8)

    ax_barchart.set_xlim(0, 100)
    ax_barchart.set_xlabel('\nGrowth Phenotype\nProportions')
    ax_barchart.set_xticks([])
    ax_barchart.set_ylabel('')
    ax_barchart.set_yticks([])

    # Annotate bars with percentages
    for p in ax_barchart.patches:
        width = p.get_width()
        if width > 1.5:
            x = p.get_x() + width / 2
            y = p.get_y() + p.get_height() / 2
            ax_barchart.text(x, y, f'{width:.0f}%',
                             ha='center', va='center', fontsize=9,
                             color='black',
                             bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

    ax_barchart.legend(title='Growth\nPhenotype\nKey', loc='center left',
                       bbox_to_anchor=(1.05, 0.5), frameon=False)
    sns.despine(ax=ax_barchart, offset=10)
    ax_barchart.spines['left'].set_visible(False)

    plt.tight_layout(w_pad=3.0)
    plt.show()



def growth_dynamics_by_origin(subset_df, expanded_piyg, width_mm=280, height_mm=100):
    """
    Plot doubling time distributions segregated by Mtb origin (intracellular growth vs. extracellular jump),
    with a bar chart showing counts of each origin per category.

    Parameters
    ----------
    subset_df : pd.DataFrame
        Must contain columns: 'Doubling Times', 'Strain', 'mtb_origin'.

    expanded_piyg : list
        List of hex color codes.

    width_mm : float
        Width of the figure in millimetres.

    height_mm : float
        Height of the figure in millimetres.
    """

    # Convert mm to inches
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    # Bin doubling times into categories
    bins = [0, 20, 28, float('inf')]
    labels = ['Fast', 'Normal', 'Slow']
    subset_df['Growth Category'] = pd.cut(subset_df['Doubling Times'], bins=bins, labels=labels, right=False)

    # Label rows based on Mtb origin
    subset_df['Growth'] = subset_df.apply(
        lambda row: f"{row['Strain']} - Intracellular Growth" if row['mtb_origin'] == 'Growth'
                    else f"{row['Strain']} - Extracellular Jump", axis=1
    )

    # Define order of groups
    order = ['RD1 - Extracellular Jump', 'RD1 - Intracellular Growth',
             'WT - Extracellular Jump', 'WT - Intracellular Growth']

    # Set up figure and gridspec layout
    plt.figure(figsize=(width_in, height_in / 1.25))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    # --- Left plot: Boxplot and Stripplot ---
    ax1 = plt.subplot(gs[0])
    sns.boxplot(
        data=subset_df, x="Doubling Times", y="Growth", hue="Strain",
        whis=[0, 100], width=.6, palette=[expanded_piyg[1], expanded_piyg[-2]],
        order=order, ax=ax1
    )
    sns.stripplot(
        data=subset_df, x="Doubling Times", y="Growth", size=4, hue="Strain",
        palette=[expanded_piyg[0], expanded_piyg[-1]], alpha=0.5,
        edgecolor='gray', linewidth=1, jitter=0.3,
        order=order, ax=ax1
    )

    # Shaded categories
    ax1.axvspan(0, 20, color=expanded_piyg[3], alpha=0)
    ax1.axvspan(20, 28, color=expanded_piyg[3], alpha=0.8)
    ax1.axvspan(28, 70, color=expanded_piyg[3], alpha=0)

    # Text annotations
    ax1.text(0.5, 3, 'Fast', fontsize=50, color='gray', alpha=0.2, ha='left', rotation=90)
    ax1.text(20.5, 3, 'Normal', fontsize=50, color='white', alpha=0.4, ha='left', rotation=90)
    ax1.text(70.5, 3, 'Slow', fontsize=50, color='gray', alpha=0.2, ha='right', rotation=90)

    ax1.set_xlabel('Doubling Time (hours)')
    ax1.xaxis.grid(True)
    sns.despine(ax=ax1, offset=10, left=True)
    ax1.legend().remove()

    # --- Right plot: Bar chart of Mtb origin counts ---
    ax2 = plt.subplot(gs[1])

    origin_counts = subset_df.groupby(['Growth', 'mtb_origin']).size().unstack(fill_value=0)
    for drop_cat in ['Junk', 'Unknown']:
        if drop_cat in origin_counts.columns:
            del origin_counts[drop_cat]
    origin_counts = origin_counts.reindex(reversed(order))
    origin_counts.plot(kind='barh', stacked=True, ax=ax2,
                       color=[expanded_piyg[1], expanded_piyg[-2], expanded_piyg[-1]])

    ax2.set_xlabel('Count')
    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    sns.despine(ax=ax2, offset=10, left=True)
    ax2.spines['left'].set_visible(False)
    ax2.legend(title='Mtb origin', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # Final layout
    plt.tight_layout(w_pad=3.0)
    plt.show()


def doubling_times_with_piecharts(subset_df, expanded_piyg, width_in=10, height_in=6):
    """
    Generate a multi-panel figure showing boxplots of doubling times across antibiotic conditions,
    and side-panel pie charts illustrating growth category distributions (Fast, Normal, Slow)
    for each strain/compound/concentration combination.

    Parameters
    ----------
    subset_df : pd.DataFrame
        Filtered DataFrame with columns 'Doubling Times', 'Compound', 'Strain',
        'Concentration', and 'mtb_origin'.

    expanded_piyg : list
        List of hex color values to be used as the palette.

    width_in : float
        Width of the full figure in inches.

    height_in : float
        Height of the full figure in inches.
    """


    # Bin doubling times
    bins = [0, 20, 28, float('inf')]
    labels = ['Fast', 'Normal', 'Slow']
    subset_df['Growth Category'] = pd.cut(subset_df['Doubling Times'], bins=bins, labels=labels, right=False)

    # Normalize labels
    subset_df['Mtb Strain or Drug Compound'] = subset_df.apply(
        lambda row: 'CTRL' if row['Compound'] == 'CTRL' else row['Compound'], axis=1
    )
    subset_df['CTRL Type'] = subset_df.apply(
        lambda row: row['Strain'] if row['Compound'] == 'CTRL' else None, axis=1
    )

    strain_order = ['CTRL', 'PZA', 'INH', 'RIF']

    fig = plt.figure(figsize=(width_in, height_in))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax = plt.subplot(gs[0])

    for compound in strain_order:
        subset = subset_df[subset_df['Mtb Strain or Drug Compound'] == compound]
        available_concentrations = subset['Concentration'].unique()

        if compound == 'CTRL':
            sns.boxplot(
                data=subset, x="Doubling Times", y="Mtb Strain or Drug Compound", hue="CTRL Type",
                whis=[0, 100], width=.75, palette=[expanded_piyg[0], expanded_piyg[1]], ax=ax, order=strain_order
            )
            sns.stripplot(
                data=subset, x="Doubling Times", y="Mtb Strain or Drug Compound", hue="CTRL Type", size=4,
                edgecolor='gray', linewidth=1, jitter=0.3,
                palette=[expanded_piyg[0], expanded_piyg[1]], dodge=True, alpha=0.5, ax=ax, order=strain_order
            )
        else:
            palette = [expanded_piyg[4] if concentration == 'EC50' else expanded_piyg[5] for concentration in available_concentrations]
            sns.boxplot(
                data=subset, x="Doubling Times", y="Mtb Strain or Drug Compound", hue="Concentration",
                whis=[0, 100], width=.75, palette=palette, hue_order=available_concentrations, ax=ax, order=strain_order
            )
            sns.stripplot(
                data=subset, x="Doubling Times", y="Mtb Strain or Drug Compound", hue="Concentration", size=4,
                edgecolor='gray', linewidth=1, jitter=0.3,
                palette=palette, hue_order=available_concentrations, dodge=True, alpha=0.5, ax=ax, order=strain_order
            )

    # Add visual elements
    ax.axvspan(0, 20, color=expanded_piyg[3], alpha=0)
    ax.axvspan(20, 28, color=expanded_piyg[3], alpha=0.8)
    ax.axvspan(28, 70, color=expanded_piyg[3], alpha=0.)

    ax.text(0, 3.35, 'Fast', fontsize=50, color='gray', alpha=0.2, ha='left', rotation=90)
    ax.text(20, 3.35, 'Normal', fontsize=50, color='white', alpha=0.4, ha='left', rotation=90)
    ax.text(70.5, 3.35, 'Slow', fontsize=50, color='gray', alpha=0.2, ha='right', rotation=90)

    ax.set_xlabel("Doubling Times (hours)")
    ax.xaxis.grid(True)
    sns.despine(offset=10, left=True)

    legend_handles = [
        mpatches.Patch(color=expanded_piyg[0], label='Wild-type control'),
        mpatches.Patch(color=expanded_piyg[1], label='∆RD1 control'),
        mpatches.Patch(color=expanded_piyg[4], label='EC50'),
        mpatches.Patch(color=expanded_piyg[5], label='EC99')
    ]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False)

    # Pie chart data
    compound_order = ['WT (CTRL)', 'RD1 (CTRL)', 'PZA', 'INH', 'RIF']
    concentration_order = ['EC50', 'EC0', 'EC99']
    growth_labels = ['Fast', 'Normal', 'Slow']
    pie_colormap = [expanded_piyg[1], expanded_piyg[3], expanded_piyg[4]]

    grouped_counts = subset_df.groupby(['Mtb Strain or Drug Compound', 'Concentration', 'Growth Category']).size()
    pi_x_start_pos = 0.75
    pi_y_start_pos = 0.93
    y_gap = 0.2

    for i, compound in enumerate(compound_order):
        for j, concentration in enumerate(concentration_order):
            fast_count = grouped_counts.get((compound, concentration, 'Fast'), 0)
            normal_count = grouped_counts.get((compound, concentration, 'Normal'), 0)
            slow_count = grouped_counts.get((compound, concentration, 'Slow'), 0)

            if fast_count + normal_count + slow_count > 0:
                counts = [fast_count, normal_count, slow_count]
                filtered_counts = [c for c in counts if c > 0]
                filtered_labels = [growth_labels[k] for k, c in enumerate(counts) if c > 0]
                filtered_colors = [pie_colormap[k] for k, c in enumerate(counts) if c > 0]

                x_pos = pi_x_start_pos + 0.15 * (1 if concentration == 'EC99' or compound == 'RD1 (CTRL)' else 0)
                y_pos = pi_y_start_pos - (i * y_gap)

                ax_pie = fig.add_axes([x_pos, y_pos, 0.15, 0.15], aspect=1)
                ax_pie.pie(filtered_counts, colors=filtered_colors, labels=filtered_labels,
                           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})

                ax_label = fig.add_axes([x_pos, y_pos - 0.02, 0.1, 0.03])
                ax_label.text(0.75, 5.75, f'{compound} - {concentration}', ha='center', va='center',
                              fontsize=9, fontweight='bold')
                ax_label.set_axis_off()

    plt.tight_layout()
    plt.show()
