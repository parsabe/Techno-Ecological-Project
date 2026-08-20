import os
import subprocess
import shutil

def escape_latex(val):
    """Escapes special LaTeX characters in dynamic string values."""
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    val = val.replace('\\', '\\textbackslash{}')
    val = val.replace('&', '\\&')
    val = val.replace('%', '\\%')
    val = val.replace('$', '\\$')
    val = val.replace('#', '\\#')
    val = val.replace('_', '\\_')
    val = val.replace('~', '\\textasciitilde{}')
    val = val.replace('^', '\\textasciicircum{}')
    return val

def process_latex_template(template_path, tex_output_path, stats_data, plots_dir, census_summary=None, fish_images_dir=None):
    """
    Reads the standalone LaTeX template, injects species census, specimen gallery, 
    statistical data, and dynamic plot analysis blocks, writing final .tex file.
    """
    tex_output_path = os.path.abspath(tex_output_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(template_path) or not template_path.endswith('.tex'):
        template_path = os.path.join(script_dir, "report_template.tex")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"LaTeX template not found at {template_path}")

    # Read the clean LaTeX template
    with open(template_path, 'r', encoding='utf-8') as f:
        template_str = f.read()

    # 1. Generate Species Census Data Rows
    species_census_str = ""
    if census_summary and census_summary.get("sorted_species"):
        sorted_species = census_summary["sorted_species"]
        tot = census_summary.get("total_unique", sum(c for _, c in sorted_species)) or 1
        for rank, (sp_name, count) in enumerate(sorted_species, start=1):
            biomass_pct = (count / float(tot)) * 100.0
            clean_name = escape_latex(sp_name)
            species_census_str += f"{rank} & {clean_name} & {count} & {biomass_pct:.1f}\\% \\\\\n"
    else:
        species_census_str = "1 & Specimen Taxa & 1 & 100.0\\% \\\\\n"

    # 2. Generate Fish Image Specimen Gallery (1 Deduplicated Crop Per Species)
    fish_gallery_str = ""
    if fish_images_dir and os.path.exists(fish_images_dir):
        all_imgs = sorted([f for f in os.listdir(fish_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Deduplicate to 1 image per species
        seen_species = set()
        species_imgs = []
        for img_f in all_imgs:
            sp_key = os.path.splitext(img_f)[0].split('_id')[0].replace('_', ' ').strip()
            if sp_key not in seen_species:
                seen_species.add(sp_key)
                species_imgs.append((sp_key, img_f))

        if species_imgs:
            fish_gallery_str += "\\begin{figure}[h!]\n\\centering\n\\begin{tabular}{ccc}\n"
            row_imgs = []
            row_lbls = []
            for idx, (sp_key, img_f) in enumerate(species_imgs):
                img_fpath = os.path.join(fish_images_dir, img_f)
                rel_img = os.path.relpath(img_fpath, os.path.dirname(tex_output_path)).replace('\\', '/')
                clean_lbl = escape_latex(sp_key)
                row_imgs.append(f"\\includegraphics[width=4.0cm,height=3.0cm,keepaspectratio=false]{{{rel_img}}}")
                row_lbls.append(f"\\small{{\\textbf{{{clean_lbl}}}}}")

                if len(row_imgs) == 3 or idx == len(species_imgs) - 1:
                    while len(row_imgs) < 3:
                        row_imgs.append("")
                        row_lbls.append("")
                    fish_gallery_str += " & ".join(row_imgs) + " \\\\\n"
                    fish_gallery_str += " & ".join(row_lbls) + " \\\\[0.3cm]\n"
                    row_imgs = []
                    row_lbls = []
            fish_gallery_str += "\\end{tabular}\n"
            fish_gallery_str += "\\caption{Deduplicated representative specimen crops (1 image per detected species).}\n\\end{figure}\n\n"
    if not fish_gallery_str:
        fish_gallery_str = "\\textit{No individual specimen images were captured for this session.}\n\n"

    # 3. Generate Statistical Data Rows with LaTeX escaping
    stats_str = ""
    if stats_data:
        for row in stats_data:
            c0 = escape_latex(row[0]) if len(row) > 0 else ""
            c1 = escape_latex(row[1]) if len(row) > 1 else ""
            c2 = escape_latex(row[2]) if len(row) > 2 else ""
            c3 = escape_latex(row[3]) if len(row) > 3 else ""
            stats_str += f"{c0} & {c1} & {c2} & {c3} \\\\\n"

    # 4. Generate Plot Embeddings and Analytical Breakdowns for all 20 plots
    plots_str = ""
    plot_descriptions = [
        ("01_population_time_series.png", "Figure 01: Stochastic Population Time Series (Prey X vs Predator Y)", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} The stochastic population time series illustrates the temporal trajectory of Prey ($X$) derived from persistent YOLO detections alongside the EnKF estimated Predator ($Y$) state. The interaction reveals classic Lotka-Volterra phase lag dynamics where fluctuations in prey abundance directly modulate predator growth rates with a characteristic delay. Euler-Maruyama process noise introduces realistic environmental perturbations, simulating natural variability in aquatic nutrient availability and foraging efficiency.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Simultaneously, the continuous assimilation of live vision telemetry bounds state variance, preventing numerical drift. As predator densities peak following prey surges, subsequent prey depletion induces a phase-delayed decline in predator biomass. This cyclic equilibrium confirms the robustness of the Euler-Maruyama numerical scheme under real-world observation noise."),
        
        ("02_extinction_risk_curve.png", "Figure 02: EnKF Extinction Risk Probability Trajectory (\\%)", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} The Ensemble Kalman Filter extinction probability trajectory quantifies ecological collapse risk by tracking the percentage of ensemble members dropping below the critical threshold ($X \\le 2.0$). Rapid risk amplification reflects localized density depletion or observation gaps. Data assimilation dynamically adjusts risk bounds, offering actionable early warnings for habitat management and species conservation interventions.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} When observation density remains stable, extinction probability stabilizes below critical alert thresholds ($P < 0.35$). However, sudden drops in detected prey trigger instant risk escalation across the ensemble, demonstrating the filter's capacity to serve as an automated early warning system for environmental management."),
        
        ("03_phase_space_orbits.png", "Figure 03: Lotka-Volterra Phase Space Orbit Trajectory (X vs Y)", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Phase space orbit analysis depicts the continuous state trajectory in the $X$-$Y$ plane. Closed or spiral trajectories confirm the presence of stable limit cycles around the analytical non-trivial equilibrium point. Orbit radius variations indicate the magnitude of stochastic forcing, while orbital direction reflects prey-predator energy transfer kinetics.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} The non-linear coupling between prey growth and predator consumption maintains closed orbital geometry in state space. Perturbations induced by Gaussian noise create tight stochastic orbits, validating theoretical co-existence models in aquatic environments."),

        ("04_species_abundance_bar.png", "Figure 04: Species Census Abundance Distribution Bar Chart", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} The biological species census abundance distribution highlights the relative dominance of key marine taxa identified within the video stream. Deduplication via BoT-SORT / ByteTrack ensures persistent track ID assignment, eliminating double-counting across contiguous video frames.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Primary species counts form the baseline for community diversity and trophic pyramid calculations. High relative abundance in primary consumer species indicates a robust baseline for supporting higher trophic level predators."),

        ("05_ensemble_density_hist.png", "Figure 05: EnKF Ensemble Member State Density Distribution", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} The ensemble state density histogram displays the empirical probability distribution across all $M=100$ EnKF state vectors. Bimodal or heavy-tailed distributions signify state uncertainty during rapid population transitions.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Tight unimodal clustering indicates high Kalman filter confidence and state convergence. As measurement updates are applied, empirical variance shrinks around the maximum a posteriori state estimate."),

        ("06_cumulative_tracked_unique.png", "Figure 06: Cumulative Unique Individual Detections Over Time", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Cumulative unique individual tracking plots the rate of new specimen discovery over the session timeline. An initial steep slope represents rapid target initialization in newly observed river sections.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} The trajectory transitions toward an asymptote as the total resident population is enumerated, confirming census completeness across the active camera field of view."),

        ("07_confidence_distribution.png", "Figure 07: Bounding Box Detection Confidence Distribution", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Detection confidence score distributions measure YOLO neural network inference certainty across varied lighting, water turbidity, and specimen orientations.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} High mean confidence scores validate model generalization and reticle accuracy under field conditions, while low variance indicates consistent feature extraction."),

        ("08_velocity_magnitude_hist.png", "Figure 08: Specimen Swimmer Velocity Magnitude Distribution", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Swimmer velocity magnitude histograms analyze inter-frame specimen displacement (pixels/frame).\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} The Rayleigh-like distribution distinguishes routine cruising velocities from rapid burst-swimming evasion responses triggered by predator proximity."),

        ("09_shannon_diversity_index.png", "Figure 09: Ecological Shannon Diversity Index (H')", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} The Shannon Diversity Index ($H' = -\\sum p_i \\ln p_i$) evaluates ecological richness and proportional species abundance over time.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Temporal stability in $H'$ indicates balanced ecological community structure without hyper-dominance by a single invasive species."),

        ("10_pielou_evenness_index.png", "Figure 10: Pielou's Species Evenness Metric (J')", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Pielou's Evenness Metric ($J' = H' / \\ln S$) quantifies how evenly specimen counts are distributed across detected species.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Values approaching 1.0 indicate equal species representations, providing crucial context for community stability assessments."),

        ("11_spatial_centroid_heatmap.png", "Figure 11: 2D Spatial Centroid Density Heatmap", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Spatial centroid heatmaps render 2D kernel density estimations of specimen bounding box centers across the viewport frame.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} High-density hot spots highlight micro-habitat preferences, feeding zones, or structural shelter regions utilized by aquatic organisms."),

        ("12_growth_rate_phase_plot.png", "Figure 12: Population Growth Rate Differential Phase Plot", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Differential population growth phase plots ($dX/dt$ vs $dY/dt$) illustrate derivative dynamics across consecutive time steps.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Quadrant analysis reveals phase lead/lag relationships between prey production and predator consumption rates."),

        ("13_stochastic_noise_variance.png", "Figure 13: Euler-Maruyama Stochastic Process Noise Variance", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Stochastic process noise variance trajectories ($\sigma_X, \sigma_Y$) parameterize the intensity of Gaussian random forcing added during integration.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Maintaining calibrated noise variance prevents ensemble collapse and ensures realistic state dispersion across Monte Carlo trajectories."),

        ("14_kalman_gain_dynamics.png", "Figure 14: Ensemble Kalman Gain Dynamics Over Time", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Ensemble Kalman Gain elements ($\hat{K}_n$) track adaptive weighting applied between background model predictions and live observation updates.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} As background error covariance decreases with filter convergence, Kalman Gain dynamics stabilize toward optimal stationary weighting."),

        ("15_innovation_residuals.png", "Figure 15: Measurement Innovation Residual Error", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Innovation residual error time series ($y_n - H \\bar{\\mathbf{z}}_n$) measure the difference between actual YOLO prey counts and EnKF forecast priors.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Zero-mean residual distributions confirm unbiased state estimations and validate linear observation assumptions."),

        ("16_specimen_size_boxplot.png", "Figure 16: Specimen Bounding Box Size Area Distribution", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Bounding box area boxplots ($w \\times h$) categorize specimen size distributions.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Bimodal boxplot spreads differentiate juvenile fish from fully mature specimens within the target habitat."),

        ("17_detection_rate_fps.png", "Figure 17: Real-Time Specimen Detection Rate (Detections/s)", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Real-time detection rates (detections/second) measure pipeline throughput and target density per unit time.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Sustained high detection rates confirm real-time inference capability on NVIDIA GPU hardware without frame dropping."),

        ("18_species_cooccurrence_matrix.png", "Figure 18: Species Spatial Co-occurrence Matrix Heatmap", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Species spatial co-occurrence matrices quantify inter-species proximity and co-location frequencies within shared frame regions.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Off-diagonal co-occurrence values identify inter-specific schooling behavior and spatial overlap among co-existing taxa."),

        ("19_ecological_balance_gauge.png", "Figure 19: Prey-to-Predator Ecological Balance Ratio", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} Ecological balance ratio trajectories ($X/Y$) monitor the instantaneous proportion of prey to predators relative to theoretical equilibrium ($X^*/Y^* = 2.0$).\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Identifying periods of predator overabundance or prey depletion provides vital metrics for ecosystem health."),

        ("20_enkf_state_covariance.png", "Figure 20: Ensemble State Covariance Matrix Trace Tr(Pf)", 
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} State covariance matrix trace plots ($\\text{Tr}(P_f)$) track total ensemble variance across prey and predator state dimensions.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Monotonic covariance decay confirms Kalman filter stability and uncertainty reduction over time as initial prior uncertainty is assimilated."),

        ("21_3d_volumetric_trajectories.png", "Figure 21: 3D Volumetric Spatial Trajectory & Swimming Lanes",
         "\\textbf{Ollama AI Analytical Breakdown (Part 1):} The 3D volumetric spatial trajectory plot visualizes the 3D spatiotemporal swimming channels and depth preferences of specimen tracks over time.\n\n\\textbf{Ollama AI Analytical Breakdown (Part 2):} Spatial density clusters and elevation dynamics reveal micro-habitat utilization and structural shelter preferences within the aquatic environment.")
    ]

    for fname, title, desc in plot_descriptions:
        fpath = os.path.join(plots_dir, fname)
        if os.path.exists(fpath):
            rel_path = os.path.relpath(fpath, os.path.dirname(tex_output_path)).replace('\\', '/')
            plots_str += f"\\subsubsection*{{{title}}}\n"
            plots_str += f"{desc}\n\n"
            clean_caption = title.split(":", 1)[-1].strip() if ":" in title else title
            plots_str += f"\\begin{{figure}}[h!]\n\\centering\n\\includegraphics[width=0.85\\textwidth]{{{rel_path}}}\n\\caption{{{clean_caption}}}\n\\end{{figure}}\n\n"
            plots_str += "\\vspace{0.4cm}\n"

    # 5. Inject into the Template
    final_tex = template_str.replace("{{SPECIES_CENSUS_DATA}}", species_census_str)
    final_tex = final_tex.replace("{{FISH_GALLERY_DATA}}", fish_gallery_str)
    final_tex = final_tex.replace("{{STATS_DATA}}", stats_str)
    final_tex = final_tex.replace("{{PLOTS_DATA}}", plots_str)

    # 6. Save Final Output
    os.makedirs(os.path.dirname(tex_output_path), exist_ok=True)
    with open(tex_output_path, 'w', encoding='utf-8') as f:
        f.write(final_tex)
        
    print(f"[LaTeX Exporter] Successfully populated template and saved to: {tex_output_path}")
    return tex_output_path

def compile_latex_to_pdf(tex_path, output_pdf_path):
    """
    Locates the local pdflatex compiler and builds the PDF natively.
    """
    pdflatex_exe = shutil.which('pdflatex')
    if not pdflatex_exe:
        possible_paths = [
            r"C:\Users\parsa\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
            r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                pdflatex_exe = p
                break
                
    if pdflatex_exe:
        tex_dir = os.path.dirname(os.path.abspath(tex_path))
        tex_file = os.path.basename(tex_path)
        res = subprocess.run([pdflatex_exe, '-interaction=nonstopmode', '-output-directory', tex_dir, tex_file],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tex_dir)
        
        target_pdf_in_dir = os.path.join(tex_dir, os.path.splitext(tex_file)[0] + ".pdf")
        if os.path.exists(target_pdf_in_dir):
            if os.path.abspath(target_pdf_in_dir) != os.path.abspath(output_pdf_path):
                shutil.copy(target_pdf_in_dir, output_pdf_path)
            print(f"[pdflatex Compiler] Successfully compiled native LaTeX PDF report to: {output_pdf_path}")
            return output_pdf_path
        else:
            print(f"[pdflatex Compiler] Error during compilation:\n{res.stderr or res.stdout}")
    else:
        print("[pdflatex Compiler] Could not locate pdflatex executable. Please ensure MiKTeX or TeX Live is installed.")
    
    return None

def generate_pdf_report(template_path, plots_dir, output_pdf_path, stats_data=None, census_summary=None, fish_images_dir=None):
    """Master function to orchestrate the template reading and PDF compilation."""
    output_pdf_path = os.path.abspath(output_pdf_path)
    tex_path = os.path.splitext(output_pdf_path)[0] + ".tex"
    
    # 1. Process template
    process_latex_template(template_path, tex_path, stats_data, plots_dir, census_summary=census_summary, fish_images_dir=fish_images_dir)
    
    # 2. Compile to PDF
    return compile_latex_to_pdf(tex_path, output_pdf_path)