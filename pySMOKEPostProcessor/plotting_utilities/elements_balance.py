import re
import sys
import glob
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def elements_balance(kinetic_folder, results_folder, elements_list, threshold=0.05):
    """Plot, for each element in `elements_list`, a stackplot of which species carries
    it across an OpenSMOKE run.

    The x-axis and the underlying Output.xml layout are picked automatically from
    each Output.xml's <Type> tag:
      - Flame1D: `results_folder` is a single Output.xml holding one full spatial
        profile (every <profiles> row used) -> plotted vs axial coordinate.
      - anything else (PerfectlyStirredReactor, ShockTube, PlugFlow, ...):
        `results_folder` is a folder of Case*/Output.xml, one steady-state point
        (last <profiles> row) per Case -> plotted vs temperature.

    kinetic_folder   -- folder containing kinetics.xml (element composition per species)
    results_folder   -- folder containing Case*/Output.xml (T sweep), or the path to a
                         single Flame1D run's Output.xml
    elements_list    -- element symbols to plot, e.g. ["C", "H", "O"]; any case is accepted
                         ("he", "HE", "He" all match); elements absent from every species in
                         the mechanism are skipped with a warning rather than raising.
    threshold        -- a species gets its own band on an element's plot if it holds more
                         than this fraction of that element's total at some point in the
                         run; everything else is lumped into "Others"
    """
    O2_break_threshold = 0.30    # trigger a broken y-axis on the O plot if O2's peak share of total O exceeds this
    O2_break_margin = 0.03       # gap (as a fraction of total O) left on each side of the cut
    O2_break_bottom_fraction = 0.03  # bottom (0-margin) panel's share of the row's vertical space

    # matplotlib's tab20 hues, reordered as the 10 saturated colors followed by their
    # 10 light variants (instead of tab20's default saturated/light interleaving), so
    # the first 10 species assigned stay maximally distinct from each other.
    # Species keep the same slot across all element plots (see assign_colors).
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    others_colour = "silver"  # reserved for the "Others" lumped bucket to make it different from standard grey

    title_fontsize = 14
    legend_fontsize = 9

    def normalize_element(symbol):
        """'HE' / 'he' / 'He' => 'He', so kinetics.xml's all-caps element names
        (and however the caller capitalized elements_list) always match up."""
        return symbol[0].upper() + symbol[1:].lower() if len(symbol) > 1 else symbol.upper()

    def parse_kinetics_elements(kinetics_dir):
        """Return {SPECIES_NAME (upper): {Element: atom_count}} from kinetics.xml in
        `kinetics_dir`. <AtomicComposition> has one row per species, in the same
        order as <NamesOfSpecies> and matched by position, both keyed by <NamesOfElements>."""
        path = os.path.join(kinetics_dir, "kinetics.xml")
        root = ET.parse(path).getroot()
        element_names = [normalize_element(e) for e in root.find('NamesOfElements').text.split()]
        #element_names = root.find('NamesOfElements').text.split()  #without normalisation, can break using some externals mechs
        species_names = root.find('NamesOfSpecies').text.split()
        rows = root.find('AtomicComposition').text.strip().split('\n')
        if len(rows) != len(species_names):
            raise ValueError(f"{path}: {len(species_names)} species but {len(rows)} "
                              f"<AtomicComposition> rows — can't match by position")

        species = {}
        for name, row in zip(species_names, rows):
            counts = [float(v) for v in row.split()]
            elems = {el: int(c) for el, c in zip(element_names, counts) if c != 0}
            species[name.upper()] = elems

        print(f"[kinetics] parsed {len(species)} species from {path}", file=sys.stderr)
        return species

    def parse_output_type(path):
        """Return the OpenSMOKE <Type> of an Output.xml (e.g. 'PerfectlyStirredReactor', 'Homogeneous'(PFR and ST), 'Flame1D').
        only Flame1D gets treated differently"""
        text = open(path).read()
        m = re.search(r'<Type>\s*(.*?)\s*</Type>', text)
        if not m:
            raise ValueError(f"{path}: no <Type> tag found")
        return m.group(1)

    def parse_output_xml(path):
        """Return (add_index, species_mw, species_pos, prof_rows) for an OpenSMOKE
        Output.xml. add_index maps each '<additional>' quantity's full name (e.g.
        'temperature [K]') to its 0-based position in a raw <profiles> row;
        species_mw/species_pos map species name to molecular weight / 0-based
        position; prof_rows is every <profiles> row (one per Case steady state
        for a temperature sweep, one per spatial point for a Flame1D run), each
        as a list of floats.

        Positions are derived purely from block order (<additional> quantities
        first, then species in <mass-fractions> declaration order), NOT from the
        column-number each entry is individually tagged with in the XML: those
        tags are contiguous for a PerfectlyStirredReactor/ShockTube/PlugFlow run
        (species start right after <additional> ends) but for a Flame1D run
        OpenSMOKE reuses the <additional> block's last column number as the
        first species' number too, which silently misreads every species one
        column early if trusted directly."""
        text = open(path).read()

        add_block = re.search(r'<additional>\n(.*?)</additional>', text, re.S).group(1).strip().split('\n')
        add_count = int(add_block[0])
        add_index = {}
        for i, line in enumerate(add_block[1:1 + add_count]):
            *name_parts, _tagged_col = line.rsplit(maxsplit=1)
            add_index[' '.join(name_parts)] = i

        mf_block = re.search(r'<mass-fractions>\n(.*?)</mass-fractions>', text, re.S).group(1).strip().split('\n')
        n_species = int(mf_block[0])
        species_mw = {}
        species_pos = {}
        for i, line in enumerate(mf_block[1:1 + n_species]):
            sname, mw, _tagged_col = line.split()
            species_mw[sname] = float(mw)
            species_pos[sname] = add_count + i

        prof_block = re.search(r'<profiles>\n(.*?)</profiles>', text, re.S).group(1).strip().split('\n')
        prof_rows = [[float(v) for v in row.split()] for row in prof_block]

        return add_index, species_mw, species_pos, prof_rows

    def find_additional(add_index, key_prefix):
        """Return (full_key, row_position) of the <additional> entry whose name
        starts with `key_prefix` (e.g. 'temperature' or 'axial-coordinate').
        Done because some specific output don't start directly on the species masses so its skip the rows"""
        return next((k, v) for k, v in add_index.items() if k.startswith(key_prefix))

    def load_cases(paths):
        """Stitch Output.xml files (any non-Flame1D reactor type (e.g.
        PerfectlyStirredReactor, ShockTube, PlugFlow) sweep, one steady-state row
        each) into one DataFrame of mass fractions vs T, sorted by T."""
        rows = {}
        mw = {}
        x_key = None
        for path in paths:
            add_index, species_mw, species_pos, prof_rows = parse_output_xml(path)
            x_key, t_pos = find_additional(add_index, 'temperature')
            last_row = prof_rows[-1]
            T = last_row[t_pos]
            rows[T] = {s: last_row[pos] for s, pos in species_pos.items()}
            mw.update({s: m for s, m in species_mw.items() if s not in mw})

        df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
        df.index.name = x_key
        return df, mw

    def load_flame(path):
        """Load a single Flame1D Output.xml full spatial profile (every
        <profiles> row, one per axial position) into a DataFrame of mass
        fractions vs axial coordinate."""
        add_index, species_mw, species_pos, prof_rows = parse_output_xml(path)
        x_key, x_pos = find_additional(add_index, 'axial-coordinate')

        rows = {}
        for row in prof_rows:
            x = row[x_pos]
            rows[x] = {s: row[pos] for s, pos in species_pos.items()}

        df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
        df.index.name = x_key
        return df, species_mw

    def resolve_output(out_dir):
        """Find the Output.xml file(s) under 'out_dir' and the <Type> of the
        first one. 'out_dir' may be a single Output.xml path, a directory of
        Case*/Output.xml (a sweep), or a directory holding one bare .xml file.
        Returns (type, paths) — paths is every file to load; loading logic
        dispatches on 'type' (Flame1D vs everything else), not on this layout."""
        if os.path.isfile(out_dir):
            return parse_output_type(out_dir), [out_dir]

        case_paths = sorted(glob.glob(f"{out_dir}/Case*/Output.xml"))
        if case_paths:
            return parse_output_type(case_paths[0]), case_paths

        xml_paths = glob.glob(f"{out_dir}/*.xml")
        if len(xml_paths) == 1:
            return parse_output_type(xml_paths[0]), xml_paths
        if len(xml_paths) > 1:
            raise SystemExit(f"Multiple .xml files under {out_dir}, expected exactly one: {xml_paths}")

        raise SystemExit(f"No Output.xml found under {out_dir} (looked for Case*/Output.xml and *.xml)")

    def element_moles(mass_fractions, mw, thermo, element):
        """DataFrame (x x species) of moles-of-"element" contributed by each species,
        per unit mass of mixture. Species absent from thermo or without "element" are dropped."""
        cols = {}
        missing = []
        for sname in mass_fractions.columns:
            comp = thermo.get(sname.upper())
            if comp is None:
                missing.append(sname)
                continue
            n_atoms = comp.get(element, 0)
            if n_atoms == 0:
                continue
            moles_species = mass_fractions[sname] / mw[sname]
            cols[sname] = moles_species * n_atoms
        if missing:
            warnings.warn(f"{len(missing)} species not found in kinetics thermo data, skipped "
                           f"(their {element} content is silently missing from this plot): "
                           f"{missing[:10]}" + (" ..." if len(missing) > 10 else ""))
        return pd.DataFrame(cols, index=mass_fractions.index)

    def lump_minor_species(elem_df, threshold):
        """Keep species whose peak fraction of the element total exceeds "threshold";
        sum the rest into an 'Others' column. Returns DataFrame sorted by mean share, descending."""
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
        """Map each species name to a fixed palette slot, shared across every element's
        plot so the same species always reads as the same color. First seen order,
        scanning elements in 'elements' order; 'Others' always gets others_colour."""
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
                  f"palette colors some species share a color", file=sys.stderr)
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
        """ Its break an axis in case of a species taking too much space (O2 mostly) 
        """
        ax_top.spines.bottom.set_visible(False)
        ax_bottom.spines.top.set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        d = 0.5
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none",
                      color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    def draw_plot(mass_fractions, mw, thermo, elements, threshold):
        x = mass_fractions.index.values
        x_label = mass_fractions.index.name
        x_short = x_label.split('[')[0].strip()

        elem_dfs = {el: element_moles(mass_fractions, mw, thermo, el) for el in elements}
        lumped_by_element = {el: lump_minor_species(elem_dfs[el], threshold) for el in elements}
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

        fig = plt.figure(figsize=(10.5, 3.6 * len(elements)))
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
                ax_top.stackplot(x, [lumped[c].values for c in lumped.columns], colors=colors,
                                  labels=lumped.columns, alpha=0.9)
                ax_bottom.stackplot(x, [lumped[c].values for c in lumped.columns], colors=colors, alpha=0.9)
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
                ax_top.set_title(f"{el} speciation vs {x_short}  (species >{threshold:.0%} of {el} total; "
                                  f"axis broken {low / total_ref:.0%}-{high / total_ref:.0%})",
                                  fontsize=title_fontsize, pad=10)
                ax_top.set_ylabel(f"{el} moles")
                # stackplot draws the first column at the bottom of the stack; reverse
                # the legend so it reads top-to-bottom in the same order as the stack
                handles, labels = ax_top.get_legend_handles_labels()
                ax_top.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.15, 0.5),
                              fontsize=legend_fontsize)
                bottom_ax = ax_bottom
            else:
                ax = fig.add_subplot(outer[i], sharex=bottom_ax)
                ax.stackplot(x, [lumped[c].values for c in lumped.columns], colors=colors, labels=lumped.columns,
                             alpha=0.9)
                total_ref = elem_dfs[el].sum(axis=1).mean()
                add_percent_axis(ax, total_ref, label=True)
                ax.set_ylabel(f"{el} moles")
                ax.set_title(f"{el} speciation vs {x_short}  (species >{threshold:.0%} of {el} total)",
                             fontsize=title_fontsize, pad=10)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.15, 0.5),
                          fontsize=legend_fontsize)
                bottom_ax = ax

        bottom_ax.set_xlabel(x_label)

    thermo = parse_kinetics_elements(kinetic_folder)

    output_type, paths = resolve_output(results_folder)
    if output_type == 'Flame1D':
        mass_fractions, mw = load_flame(paths[0])
    else:
        mass_fractions, mw = load_cases(paths)

    draw_plot(mass_fractions, mw, thermo, elements_list, threshold)
    plt.show()
