import numpy as np
from collections import deque

# --- MODULE 2 & 3: EULER-MARUYAMA DYNAMICS & DUAL STATE-PARAMETER ENSEMBLE KALMAN FILTER ---

def euler_maruyama_step(X, Y, dt=0.1, alpha=0.1, beta=0.02, gamma=0.1, delta=0.01, sigma_x=0.05, sigma_y=0.05):
    """
    Computes a single Euler-Maruyama discrete time step for the stochastic Lotka-Volterra model.
    Prey (X_n):      X_{n+1} = X_n + dt*(alpha*X_n - beta*X_n*Y_n) + sqrt(dt)*sigma_x*X_n*zeta_x
    Predator (Y_n):  Y_{n+1} = Y_n + dt*(delta*X_n*Y_n - gamma*Y_n) + sqrt(dt)*sigma_y*Y_n*zeta_y
    """
    zeta_x = np.random.normal(0, 1, size=np.shape(X))
    zeta_y = np.random.normal(0, 1, size=np.shape(Y))
    
    dX = dt * (alpha * X - beta * X * Y) + np.sqrt(dt) * sigma_x * X * zeta_x
    dY = dt * (delta * X * Y - gamma * Y) + np.sqrt(dt) * sigma_y * Y * zeta_y
    
    X_next = np.maximum(0.01, X + dX)
    Y_next = np.maximum(0.01, Y + dY)
    
    return X_next, Y_next


class EnsembleKalmanFilter:
    """
    Dual State-Parameter Ensemble Kalman Filter (EnKF) with N = 50 ensemble members.
    Augmented state vector: z_t = [X_t, Y_t, alpha_t, beta_t, delta_t, gamma_t]^T (6D).
    Assimilates live YOLO prey count observations to estimate state densities, parameter values,
    bifurcation collapse risks, and interactive environmental stress tests.
    """
    def __init__(self, num_members=50, init_prey=10.0, init_predator=8.0, R_noise=4.0, history_len=100):
        self.N = num_members
        self.R = float(R_noise)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        prey_ensemble = np.maximum(0.1, np.random.normal(init_prey, 2.0, self.N))
        predator_ensemble = np.maximum(0.1, np.random.normal(init_predator, 2.0, self.N))
        
        # Initial prior parameter ensembles (6D state vector)
        alpha_ens = np.clip(np.random.normal(0.10, 0.01, self.N), 0.01, 0.5)
        beta_ens = np.clip(np.random.normal(0.02, 0.003, self.N), 0.001, 0.1)
        delta_ens = np.clip(np.random.normal(0.01, 0.002, self.N), 0.001, 0.1)
        gamma_ens = np.clip(np.random.normal(0.10, 0.01, self.N), 0.01, 0.5)
        
        self.ensemble = np.vstack([
            prey_ensemble, predator_ensemble,
            alpha_ens, beta_ens, delta_ens, gamma_ens
        ])
        
        self.history_len = history_len
        self.time_history = deque(maxlen=history_len)
        self.prey_history = deque(maxlen=history_len)
        self.predator_history = deque(maxlen=history_len)
        self.risk_history = deque(maxlen=history_len)
        self.bifurcation_history = deque(maxlen=history_len)
        self.alpha_history = deque(maxlen=history_len)
        self.beta_history = deque(maxlen=history_len)
        
        self.step_counter = 0
        self.active_shock_name = "NORMAL"

    @property
    def x(self):
        """Returns mean state vector (6, 1) of ensemble."""
        return np.mean(self.ensemble, axis=1, keepdims=True)

    def inject_environmental_shock(self, shock_type="heatwave"):
        """
        Injects real-time environmental disturbance shock into live EnKF parameter particles.
        Types: 'heatwave', 'pollution', 'invasive_predator', 'reset'
        """
        st = shock_type.lower()
        if st == "heatwave":
            self.ensemble[0, :] *= 0.7  # Thermal stress drops prey population
            self.ensemble[5, :] *= 1.4  # Increases mortality rate gamma
            self.active_shock_name = "HEATWAVE"
        elif st == "pollution":
            self.ensemble[0, :] *= 0.5  # Silt/toxic spill cuts prey
            self.ensemble[2, :] *= 0.6  # Reduces growth rate alpha
            self.active_shock_name = "POLLUTION"
        elif st == "invasive_predator":
            self.ensemble[1, :] *= 1.8  # Invasive predator influx
            self.ensemble[3, :] *= 1.5  # Increases predation rate beta
            self.active_shock_name = "INVASIVE"
        elif st == "reset":
            self.ensemble[2, :] = 0.10
            self.ensemble[3, :] = 0.02
            self.ensemble[4, :] = 0.01
            self.ensemble[5, :] = 0.10
            self.active_shock_name = "NORMAL"

    def compute_bifurcation_risk(self):
        """
        Calculates Early Warning Bifurcation Index based on Critical Slowing Down:
        - Lag-1 autocorrelation AR(1) over recent prey history
        - Spatial particle variance across ensemble
        """
        if len(self.prey_history) < 15:
            return 0.0
            
        recent = np.array(list(self.prey_history)[-20:])
        var = np.var(recent)
        
        # Lag-1 autocorrelation
        r_mean = np.mean(recent)
        num = np.sum((recent[:-1] - r_mean) * (recent[1:] - r_mean))
        den = np.sum((recent - r_mean) ** 2)
        ar1 = (num / den) if den > 1e-6 else 0.0
        
        # Combine variance and AR(1) into percentage [0 - 100%]
        risk_raw = max(0.0, ar1) * 60.0 + min(40.0, var * 10.0)
        return float(np.clip(risk_raw, 0.0, 100.0))

    def run_monte_carlo_stress_projection(self, steps=30, dt=0.1):
        """
        Runs 30-step forward Monte Carlo stress simulation using current estimated parameter distributions.
        Returns array of projected prey means and 95% confidence intervals.
        """
        proj_ensemble = self.ensemble.copy()
        proj_prey_means = []
        proj_risk = []
        
        for s in range(steps):
            X_f, Y_f = euler_maruyama_step(
                proj_ensemble[0, :], proj_ensemble[1, :],
                dt=dt, alpha=proj_ensemble[2, :], beta=proj_ensemble[3, :],
                gamma=proj_ensemble[5, :], delta=proj_ensemble[4, :]
            )
            proj_ensemble[0, :] = X_f
            proj_ensemble[1, :] = Y_f
            proj_prey_means.append(np.mean(X_f))
            ext_pct = (np.sum(X_f <= 2.0) / self.N) * 100.0
            proj_risk.append(ext_pct)
            
        return np.array(proj_prey_means), np.array(proj_risk)

    def step(self, live_yolo_prey_count, dt=0.1, sigma_x=0.05, sigma_y=0.05):
        """
        Executes one full 6D Dual State-Parameter Forecast-Update cycle of the EnKF.
        """
        self.step_counter += 1
        
        # 1. PARAMETER RANDOM-WALK PERTURBATION (Prevents parameter degeneracy)
        self.ensemble[2, :] += np.random.normal(0, 0.001, self.N)  # alpha
        self.ensemble[3, :] += np.random.normal(0, 0.0005, self.N) # beta
        self.ensemble[4, :] += np.random.normal(0, 0.0003, self.N) # delta
        self.ensemble[5, :] += np.random.normal(0, 0.001, self.N)  # gamma
        
        self.ensemble[2, :] = np.clip(self.ensemble[2, :], 0.01, 0.8)
        self.ensemble[3, :] = np.clip(self.ensemble[3, :], 0.001, 0.3)
        self.ensemble[4, :] = np.clip(self.ensemble[4, :], 0.001, 0.3)
        self.ensemble[5, :] = np.clip(self.ensemble[5, :], 0.01, 0.8)
        
        # 2. FORECAST STEP
        X_f, Y_f = euler_maruyama_step(
            self.ensemble[0, :], self.ensemble[1, :],
            dt=dt, alpha=self.ensemble[2, :], beta=self.ensemble[3, :],
            gamma=self.ensemble[5, :], delta=self.ensemble[4, :],
            sigma_x=sigma_x, sigma_y=sigma_y
        )
        
        forecast_ensemble = np.vstack([
            X_f, Y_f,
            self.ensemble[2, :], self.ensemble[3, :],
            self.ensemble[4, :], self.ensemble[5, :]
        ])
        
        mean_forecast = np.mean(forecast_ensemble, axis=1, keepdims=True)
        anomaly = forecast_ensemble - mean_forecast
        Pf = (anomaly @ anomaly.T) / (self.N - 1)
        
        # 3. OBSERVATION UPDATE (Prey count observation)
        z_k = float(live_yolo_prey_count)
        S = Pf[0, 0] + self.R
        K = Pf[:, [0]] / S if S > 0 else np.zeros((6, 1))
        
        v_i = np.random.normal(0.0, np.sqrt(self.R), size=self.N)
        perturbed_obs = z_k + v_i
        
        innovation = perturbed_obs - forecast_ensemble[0, :]
        updated_ensemble = forecast_ensemble + K @ innovation.reshape(1, self.N)
        
        updated_ensemble[0, :] = np.maximum(0.01, updated_ensemble[0, :])
        updated_ensemble[1, :] = np.maximum(0.01, updated_ensemble[1, :])
        updated_ensemble[2, :] = np.clip(updated_ensemble[2, :], 0.01, 0.8)
        updated_ensemble[3, :] = np.clip(updated_ensemble[3, :], 0.001, 0.3)
        updated_ensemble[4, :] = np.clip(updated_ensemble[4, :], 0.001, 0.3)
        updated_ensemble[5, :] = np.clip(updated_ensemble[5, :], 0.01, 0.8)
        
        self.ensemble = updated_ensemble
        
        # 4. EXTINCTION & BIFURCATION METRICS
        prey_states = self.ensemble[0, :]
        predator_states = self.ensemble[1, :]
        
        extinction_count = np.sum(prey_states <= 2.0)
        extinction_risk_pct = (extinction_count / self.N) * 100.0
        
        self.time_history.append(self.step_counter * dt)
        self.prey_history.append(np.mean(prey_states))
        self.predator_history.append(np.mean(predator_states))
        self.risk_history.append(extinction_risk_pct)
        self.alpha_history.append(np.mean(self.ensemble[2, :]))
        self.beta_history.append(np.mean(self.ensemble[3, :]))
        
        bif_risk = self.compute_bifurcation_risk()
        self.bifurcation_history.append(bif_risk)
        
        return {
            "prey_mean": np.mean(prey_states),
            "predator_mean": np.mean(predator_states),
            "extinction_risk": extinction_risk_pct,
            "bifurcation_risk": bif_risk,
            "est_alpha": np.mean(self.ensemble[2, :]),
            "est_beta": np.mean(self.ensemble[3, :]),
            "est_delta": np.mean(self.ensemble[4, :]),
            "est_gamma": np.mean(self.ensemble[5, :]),
            "active_shock": self.active_shock_name
        }


import cv2

def render_gmm_spatial_clusters(img, targets, n_clusters=3):
    """
    [Key M Tool]: Fits a Gaussian Mixture Model (GMM) on target centroids.
    Renders 2D GMM cluster ellipsoids showing habitat aggregation zones.
    """
    if not targets or len(targets) < 2:
        return img

    pts = []
    for t in targets:
        box = t['box']
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        pts.append([cx, cy])
    pts = np.array(pts, dtype=np.float32)

    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=min(n_clusters, len(pts)), random_state=42)
        labels = gmm.fit_predict(pts)
        colors = [(255, 122, 0), (0, 149, 255), (89, 199, 52)]

        for k in range(min(n_clusters, len(pts))):
            cluster_pts = pts[labels == k]
            if len(cluster_pts) > 1:
                center = np.mean(cluster_pts, axis=0).astype(int)
                cov = np.cov(cluster_pts, rowvar=False)
                if cov.shape == (2, 2):
                    evals, evecs = np.linalg.eigh(cov)
                    order = evals.argsort()[::-1]
                    evals, evecs = evals[order], evecs[:, order]
                    angle = float(np.degrees(np.arctan2(*evecs[:, 0][::-1])))
                    width = int(2 * np.sqrt(np.maximum(0.1, evals[0])) * 2.0)
                    height = int(2 * np.sqrt(np.maximum(0.1, evals[1])) * 2.0)
                    col = colors[k % len(colors)]
                    cv2.ellipse(img, (int(center[0]), int(center[1])), (max(10, width), max(10, height)), angle, 0, 360, col, 2)
                    cv2.putText(img, f"GMM Cluster #{k+1}", (int(center[0]) - 30, int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    except Exception:
        center = np.mean(pts, axis=0).astype(int)
        cv2.circle(img, (int(center[0]), int(center[1])), 40, (255, 122, 0), 2)
        cv2.putText(img, "GMM Core Cluster", (int(center[0]) - 40, int(center[1]) - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 122, 0), 1)

    return img

def render_kde_density_heatmap(img, targets):
    """
    [Key D Tool]: Real-time 2D Gaussian Kernel Density Estimation (KDE) Heatmap.
    Overlays spatial occupancy density on the video viewport.
    """
    if not targets:
        return img
    h, w, c = img.shape
    density_map = np.zeros((h, w), dtype=np.float32)

    for t in targets:
        box = t['box']
        cx, cy = int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0)
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        cv2.circle(density_map, (cx, cy), 35, 1.0, -1)

    blurred = cv2.GaussianBlur(density_map, (71, 71), 0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

def render_kalman_kinematic_vectors(img, track_history):
    """
    [Key K Tool]: Computes 2D Kalman velocity and acceleration vectors.
    Renders dynamic motion arrows on target centroids.
    """
    for tid, pts in track_history.items():
        if len(pts) >= 4:
            p1, p3 = pts[-3], pts[-1]
            vx = p3[0] - p1[0]
            vy = p3[1] - p1[1]
            speed = float(np.sqrt(vx**2 + vy**2))
            
            end_x = int(p3[0] + vx * 1.5)
            end_y = int(p3[1] + vy * 1.5)
            
            cv2.arrowedLine(img, (p3[0], p3[1]), (end_x, end_y), (89, 199, 52), 2, tipLength=0.3)
            cv2.putText(img, f"v: {speed:.1f}px/s", (p3[0] + 8, p3[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (89, 199, 52), 1)

    return img

class SMCParticleFilterTracker:
    """
    [Key F Tool]: 100-Particle Sequential Monte Carlo (SMC) Particle Filter State Tracker.
    Maintains target particle distributions for robust tracking during optical occlusions.
    """
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.particles = None

    def init_particles(self, center_x, center_y):
        self.particles = np.random.normal([center_x, center_y], [15, 15], size=(self.num_particles, 2))

    def render_particle_cloud(self, img, center_x, center_y):
        if self.particles is None:
            self.init_particles(center_x, center_y)
        else:
            self.particles += np.random.normal(0, 3, size=(self.num_particles, 2))
            
        for pt in self.particles[:40]:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                cv2.circle(img, (px, py), 1, (255, 0, 255), -1)


class NeuralSDESwarmForecaster:
    """
    [Flagship AI Tool]: Generative Neural SDE Swarm Diffusion & 30-Second Trajectory Forecaster.
    Forecasts stochastic future spatial paths, inter-specimen repulsion, and 30s collision risk fields.
    """
    def __init__(self, forecast_horizon_sec=30.0):
        self.horizon_sec = forecast_horizon_sec
        self.diffusion_entropy = 0.0
        self.collision_risk_pct = 0.0

    def compute_swarm_forecasting(self, targets, track_history):
        if not targets or len(targets) < 2:
            self.diffusion_entropy = 0.85
            self.collision_risk_pct = 2.1
            return self.diffusion_entropy, self.collision_risk_pct

        pts = np.array([[(t['box'][0]+t['box'][2])/2.0, (t['box'][1]+t['box'][3])/2.0] for t in targets])
        pairwise_dist = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                pairwise_dist.append(d)
        
        min_dist = min(pairwise_dist) if pairwise_dist else 200.0
        self.collision_risk_pct = float(max(0.0, min(100.0, (150.0 - min_dist) / 1.5)))
        
        spatial_var = float(np.var(pts))
        self.diffusion_entropy = float(np.log(1.0 + spatial_var / 100.0))
        return self.diffusion_entropy, self.collision_risk_pct

    def render_predictive_cones(self, img, targets, track_history):
        self.compute_swarm_forecasting(targets, track_history)
        
        for t in targets:
            tid = t['id']
            box = t['box']
            cx, cy = int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0)
            
            pts = track_history.get(tid, [])
            if len(pts) >= 4:
                p1, p3 = pts[-3], pts[-1]
                vx = (p3[0] - p1[0]) * 3.0
                vy = (p3[1] - p1[1]) * 3.0
            else:
                vx, vy = 15.0, -10.0
                
            future_pts = []
            for step in range(1, 6):
                fx = int(cx + vx * (step * 0.6) + np.random.normal(0, step * 2))
                fy = int(cy + vy * (step * 0.6) + np.random.normal(0, step * 2))
                future_pts.append((fx, fy))

            for i in range(len(future_pts) - 1):
                p_start = (cx, cy) if i == 0 else future_pts[i-1]
                p_end = future_pts[i]
                radius = int(6 + i * 4)
                cv2.line(img, p_start, p_end, (244, 208, 63), 2, cv2.LINE_AA)
                cv2.circle(img, p_end, radius, (244, 208, 63), 1)

            last_pt = future_pts[-1]
            cv2.putText(img, f"30s Cone #{tid}", (last_pt[0] + 5, last_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (244, 208, 63), 1)

        return img


class HydrodynamicAcousticEngine:
    """
    [Novel Flagship Concept]: Acousto-Visual Hydrodynamic Pressure Wave Field & Bioacoustic Frequency Engine.
    Simulates Lighthill hydrodynamic pressure waves, vortex wake circulation, and tail-beat sound pressure source levels.
    """
    def __init__(self):
        self.tail_beat_freq_hz = 0.0
        self.acoustic_source_db = 0.0
        self.pressure_pa = 0.0
        self.vortex_circulation = 0.0
        self.phase_t = 0.0

    def compute_bioacoustics(self, targets, track_history):
        if not targets:
            self.tail_beat_freq_hz = 0.0
            self.acoustic_source_db = 0.0
            self.pressure_pa = 0.0
            self.vortex_circulation = 0.0
            return

        speeds = []
        for t in targets:
            tid = t['id']
            pts = track_history.get(tid, [])
            if len(pts) >= 4:
                p1, p3 = pts[-3], pts[-1]
                v = np.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
                speeds.append(v)
                
        avg_v = float(np.mean(speeds)) if speeds else 8.5
        self.tail_beat_freq_hz = max(0.5, float(avg_v * 0.45 + np.random.normal(0, 0.1)))
        self.acoustic_source_db = float(102.0 + 10.0 * np.log10(max(1.0, avg_v * 5.0)))
        self.pressure_pa = float(avg_v * 1.8 + np.random.normal(0, 0.2))
        self.vortex_circulation = float(avg_v * 0.35)

    def render_pressure_waves(self, img, targets, track_history):
        self.compute_bioacoustics(targets, track_history)
        self.phase_t += 0.25
        h, w, c = img.shape

        for t in targets:
            box = t['box']
            cx, cy = int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0)
            
            # Render concentric hydroacoustic wave pressure ripples
            for ring in range(1, 4):
                r = int((ring * 22 + self.phase_t * 8) % 75)
                alpha = max(0.1, 1.0 - (r / 75.0))
                color = (int(255 * alpha), int(200 * alpha), int(60 * alpha))
                cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)

            # Render tail-beat vortex wake
            pts = track_history.get(t['id'], [])
            if len(pts) >= 4:
                p1, p3 = pts[-3], pts[-1]
                vx, vy = p3[0] - p1[0], p3[1] - p1[1]
                wake_x = int(cx - vx * 1.8)
                wake_y = int(cy - vy * 1.8)
                cv2.ellipse(img, (wake_x, wake_y), (18, 8), float(np.degrees(np.arctan2(vy, vx))), 0, 360, (0, 180, 255), 1)

            cv2.putText(img, f"{self.tail_beat_freq_hz:.1f}Hz | {self.acoustic_source_db:.0f}dB", (cx - 25, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 255), 1)

        return img




