import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import pandas as pd
import matplotlib
import xml.etree.ElementTree as ET
import os

def elements_balance(kinetic_folder, results_folders, elements_list, threshold=0.05):
    """Plot, for each element in `elements_list`, a stackplot of which species carry
    it across an OpenSMOKE temperature sweep

    kinetic_folder   -- folder containing kinetics.xml (element composition per species)
    results_folders  -- folder containing Case*/Output.xml (full species mass fractions per T)
    elements_list    -- element symbols to plot, ["C", "H", "O"]; any case is accepted
                         ("he", "HE", "He" ) elements absent from every species in
                         the mechanism are skipped with a warning rather than raising
    threshold        -- a species gets its own band on an element's plot if it holds more
                         than this fraction of that element's total at some point in the sweep
                         everything else is lumped into "Others"
    """
    O2_break_threshold = 0.30    # trigger a broken y-axis on the O plot if O2's peak share of total O exceeds this
    O2_break_margin = 0.03       # gap (as a fraction of total O) left on each side of the cut
    O2_break_bottom_fraction = 0.05  # bottom (0-margin) panel's share of the row's vertical space

    # matplotlib's tab20 hues, reordered as the 10 saturated colors then
    # 10 light variants (instead of tab20's default saturated/light interleaving), so
    # the first 10 species assigned stay maximally distinct from each other
    # Species keep the same slot across all element plots (see assign_colors)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    others_colour = "silver"  # for the "Others" lumped bucket, silver so it doesn't blend into a real species gray

    title_fontsize = 18
    legend_fontsize = 12

    def normalize_element(symbol):
        """'HE' / 'he' / 'He' -> 'He', so kinetics.xml's all-caps element names
        (and however the caller capitalized elements_list) always match up."""
        return symbol[0].upper() + symbol[1:].lower() if len(symbol) > 1 else symbol.upper()

    def parse_kinetics_elements(kinetics_dir):
        """Return a dic {SPECIES_NAME: {Element: atom_count}} from kinetics.xml in
        `kinetics_folder`, <AtomicComposition> has one row per species, in the same
        order as <NamesOfSpecies> all matched by position, matched also by <NamesOfElements>"""
        path = os.path.join(kinetics_dir, "kinetics.xml")
        root = ET.parse(path).getroot()
        element_names = [normalize_element(e) for e in root.find('NamesOfElements').text.split()]
        species_names = root.find('NamesOfSpecies').text.split()
        rows = root.find('AtomicComposition').text.strip().split('\n')
        if len(rows) != len(species_names):
            raise ValueError(f"{path}: {len(species_names)} species but {len(rows)} "
                              f"<AtomicComposition> rows — can't match by position")

        species_elements = {}
        for name, row in zip(species_names, rows):
            counts = [float(v) for v in row.split()]
            elems = {el: int(c) for el, c in zip(element_names, counts) if c != 0}
            species_elements[name.upper()] = elems

        print(f"[kinetics] parsed {len(species_elements)} species from {path}", file=sys.stderr)
        return species_elements

    def parse_case_xml(path):
        """Return (T, {species_name: (mass_fraction, MW)}) for each Case steady state
        (last row of <profiles>) in an OpenSMOKE Output.xml.
        Mass fraction and molecular weight are used as OS++ output.xml doesnt report mol%"""
        text = open(path).read()

        add_block = re.search(r'<additional>\n(.*?)</additional>', text, re.S).group(1).strip().split('\n')
        add_count = int(add_block[0])
        add_index = {}
        for line in add_block[1:1 + add_count]:
            *name_parts, idx = line.rsplit(maxsplit=1)
            add_index[' '.join(name_parts)] = int(idx)
        t_idx = next(v for k, v in add_index.items() if k.startswith('temperature'))

        mf_block = re.search(r'<mass-fractions>\n(.*?)</mass-fractions>', text, re.S).group(1).strip().split('\n')
        n_species = int(mf_block[0])
        species_mw = {}
        species_col = {}
        for line in mf_block[1:1 + n_species]:
            sname, mw, col = line.split()
            species_mw[sname] = float(mw)
            species_col[sname] = int(col)

        prof_block = re.search(r'<profiles>\n(.*?)</profiles>', text, re.S).group(1).strip().split('\n')
        last_row = [float(v) for v in prof_block[-1].split()]

        min_col = min(add_index.values())  # column-index-in-file of the first "additional" quantity
        T = last_row[t_idx - min_col]

        result = {}
        for sname, col in species_col.items():
            w = last_row[col - min_col]
            result[sname] = (w, species_mw[sname])
        return T, result

    def load_cases(out_dir):
        """Stitch every out/CaseN/Output.xml into one DataFrame of mass fractions vs T, sorted by T."""
        paths = glob.glob(f"{out_dir}/Case*/Output.xml")
        if not paths:
            raise SystemExit(f"No out/Case*/Output.xml found under {out_dir}")

        rows = {}
        mw = {}
        for path in paths:
            T, species = parse_case_xml(path)
            rows[T] = {s: w for s, (w, _) in species.items()}
            for s, (_, m) in species.items():
                mw.setdefault(s, m)

        df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
        df.index.name = 'T'
        return df, mw

    def element_moles(mass_fractions, mw, species_elements, element):
        """DataFrame (T x species) of moles-of-`element` contributed by each species,
        per unit mass of mixture. Species absent from species_elements or without
        `element` are dropped."""
        cols = {}
        missing = []
        for sname in mass_fractions.columns:
            comp = species_elements.get(sname.upper())
            if comp is None:
                missing.append(sname)
                continue
            n_atoms = comp.get(element, 0)
            if n_atoms == 0:
                continue
            moles_species = mass_fractions[sname] / mw[sname]
            cols[sname] = moles_species * n_atoms
        if missing:
            print(f"[warn] {len(missing)} species not found in kinetics data, skipped: {missing[:10]}"
                  + (" ..." if len(missing) > 10 else ""), file=sys.stderr)
        return pd.DataFrame(cols, index=mass_fractions.index)

    def lump_minor_species(elem_df, threshold):
        """Keep species whose peak fraction of the element total exceeds `threshold`
        sum the rest into an 'Others' column. Returns DataFrame sorted by mean share, descending"""
        total = elem_df.sum(axis=1)
        share = elem_df.div(total.replace(0, np.nan), axis=0).fillna(0)
        keep = share.columns[share.max(axis=0) >= threshold]
        kept = elem_df[keep]
        others = elem_df.drop(columns=keep).sum(axis=1)
        out = kept.copy()
        if others.abs().sum() > 0:
            out['Others'] = others
        return out[out.mean(axis=0).sort_values(ascending=False).index]

    def assign_colors(lumped_by_element, elements):
        """Map each species name to a fixed palette slot, shared across every element
        plot so the same species always reads as the same color always in first-seen order,
        scanning elements in `elements` order, 'Others' always gets others_colour"""
        color_of = {}
        next_slot = 0
        for el in elements:
            for sname in lumped_by_element[el].columns:
                if sname == 'Others' or sname in color_of:
                    continue
                color_of[sname] = palette[next_slot % len(palette)]
                next_slot += 1
        if next_slot > len(palette):
            print(f"[warn] {next_slot} distinct species across plots, only {len(palette)} "
                  f"palette colors — some species share a color", file=sys.stderr)
        color_of['Others'] = others_colour
        return color_of

    def add_percent_axis(ax, total_ref, label=False):
        secax = ax.secondary_yaxis(
            'right',
            functions=(lambda y: y / total_ref * 100, lambda p: p / 100 * total_ref)
        )
        if label:
            secax.set_ylabel("% of total")
        return secax

    def draw_break_marks(ax_top, ax_bottom):
        """specific function to split plots in two for easier reading in case of a O2 
        very high concentration so skip it using threshold and marging defined earlier"""
        ax_top.spines.bottom.set_visible(False)
        ax_bottom.spines.top.set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        d = 0.5
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none",
                      color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    def draw_composition(ax, T, lumped, colors, with_labels=False):
        """Draw the species breakdown on `ax`: a stackplot vs T when there are
        >=2 points, or a single stacked bar when there's only one (e.g. a lone
        Case) — a stackplot has no width to fill with just one x-value   """
        if len(T) < 2:
            bottom = 0.0
            for c, color in zip(lumped.columns, colors):
                v = lumped[c].values[0]
                ax.bar(0, v, bottom=bottom, width=0.6, color=color, label=c if with_labels else None)
                bottom += v
            ax.set_xlim(-1, 1)
            ax.set_xticks([0])
            ax.set_xticklabels([f"{T[0]:.0f}"])
        else:
            stack_kwargs = {'labels': lumped.columns} if with_labels else {}
            ax.stackplot(T, [lumped[c].values for c in lumped.columns], colors=colors, alpha=0.9, **stack_kwargs)


    def plot_element_speciation(mass_fractions, mw, species_elements, elements, threshold):
        """all plotting"""
        T = mass_fractions.index.values
        elem_dfs = {el: element_moles(mass_fractions, mw, species_elements, el) for el in elements}
        lumped_by_element = {el: lump_minor_species(elem_dfs[el], threshold) for el in elements}

        # an element with nothing to plot — either absent from every species in the
        # mechanism (asking for Cl in a mech with no chlorine), or structurally
        # present but zero throughout this run (e.g. a bath-gas species like
        # He that's declared in the mechanism but never actually fed in), has
        # an empty lumped table either way; skip it instead of crashing on an empty
        # stackplot / divide-by-zero percent axis
        present = [el for el in elements if lumped_by_element[el].shape[1] > 0]
        absent = [el for el in elements if lumped_by_element[el].shape[1] == 0]
        if absent:
            print(f"[warn] element(s) with nothing to plot (absent, or zero throughout "
                  f"this run), skipped: {absent}", file=sys.stderr)
        if not present:
            raise ValueError(f"None of the requested elements {elements} have any nonzero content")
        elements = present
        elem_dfs = {el: elem_dfs[el] for el in elements}
        lumped_by_element = {el: lumped_by_element[el] for el in elements}
        color_of = assign_colors(lumped_by_element, elements)

        # Decide which elements need a broken axis: an O2 band that eats most of the plot.
        breaks = {}
        for el in elements:
            edf = elem_dfs[el]
            if 'O2' not in edf.columns:
                continue
            total = edf.sum(axis=1)
            share_o2 = (edf['O2'] / total.replace(0, np.nan)).fillna(0)
            if share_o2.max() <= O2_break_threshold:
                continue
            total_ref = total.mean()
            low = O2_break_margin * total_ref
            high = (share_o2.min() - O2_break_margin) * total_ref
            if high > low:
                breaks[el] = (low, high, total.max(), total_ref)

        fig = plt.figure(figsize=(12.5, 3.6 * len(elements)))
        # Outer grid: one loose-spaced row per element, so titles never crowd the plot
        # above. Elements needing a broken axis get a tight nested 2-row grid of their own.
        outer = fig.add_gridspec(len(elements), 1, hspace=0.55, left=0.08, right=0.78, top=0.95, bottom=0.07)

        bottom_ax = None
        for i, el in enumerate(elements):
            lumped = lumped_by_element[el]
            colors = [color_of[c] for c in lumped.columns]

            if el in breaks:
                low, high, top, total_ref = breaks[el]
                inner = outer[i].subgridspec(2, 1, height_ratios=(1.0 - O2_break_bottom_fraction,
                                                                    O2_break_bottom_fraction), hspace=0.08)
                ax_top = fig.add_subplot(inner[0])
                ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)
                draw_composition(ax_top, T, lumped, colors, with_labels=True) 
                draw_composition(ax_bottom, T, lumped, colors, with_labels=False)                
                ax_top.set_ylim(high, top * 1.02)
                ax_bottom.set_ylim(0, low)
                draw_break_marks(ax_top, ax_bottom)
                add_percent_axis(ax_top, total_ref, label=True)
                secax_bottom = add_percent_axis(ax_bottom, total_ref)
                # bottom panel is a thin sliver (just the 0-margin strip) — too little
                # room for readable moles tick labels there, so the left axis just gets
                # tick marks and the graduation (0, 3%) lives on the right (%) axis
                margin_pct = O2_break_margin * 100
                ax_bottom.set_yticks([0, low])
                ax_bottom.set_yticklabels([])
                secax_bottom.set_yticks([0, margin_pct])
                ax_top.set_title(f"{el} speciation vs T  (species >{threshold:.0%} of {el} total; "
                                  f"axis broken {low / total_ref:.0%}-{high / total_ref:.0%})",
                                  fontsize=title_fontsize, pad=10)
                ax_top.set_ylabel(f"{el} moles")
                # stackplot draws the first column at the bottom of the stack; reverse
                # the legend so it reads top-to-bottom in the same order as the stack
                handles, labels = ax_top.get_legend_handles_labels()
                ax_top.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.08, 0.5),
                              fontsize=legend_fontsize)
                bottom_ax = ax_bottom
            else:
                ax = fig.add_subplot(outer[i], sharex=bottom_ax)
                draw_composition(ax, T, lumped, colors, with_labels=True)
                total_ref = elem_dfs[el].sum(axis=1).mean()
                add_percent_axis(ax, total_ref, label=True)
                ax.set_ylabel(f"{el} moles")
                ax.set_title(f"{el} speciation vs T  (species >{threshold:.0%} of {el} total)",
                             fontsize=title_fontsize, pad=10)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.08, 0.5),
                          fontsize=legend_fontsize)
                bottom_ax = ax

        bottom_ax.set_xlabel("T [K]")

    elements = [normalize_element(e) for e in elements_list]
    species_elements = parse_kinetics_elements(kinetic_folder)
    mass_fractions, mw = load_cases(results_folders)

    plot_element_speciation(mass_fractions, mw, species_elements, elements, threshold)
    plt.show()
