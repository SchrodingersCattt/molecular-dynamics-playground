"""
Part 1: Molecular Dynamics — From Newton to Integrators
Manim Community Edition, white background.

Render each scene individually:
    manim -pqh part1/integrators.py Scene0_Mechanics
    manim -pqh part1/integrators.py Scene1_PhaseSpace
    manim -pqh part1/integrators.py Scene2_ManyBody
    manim -pqh part1/integrators.py Scene3_Discretization
    manim -pqh part1/integrators.py Scene4_ExplicitEuler
    manim -pqh part1/integrators.py Scene5_Verlet
    manim -pqh part1/integrators.py Scene6_VelocityVerlet
    manim -pqh part1/integrators.py Scene7_Leapfrog
"""

from manim import *
import numpy as np

# ─────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────
BG   = WHITE
FG   = BLACK
C1   = "#1565C0"   # deep blue
C2   = "#B71C1C"   # deep red
C3   = "#1B5E20"   # deep green
C4   = "#E65100"   # deep orange
GRAY = "#555555"
LG   = "#BBBBBB"

config.background_color = BG

def tex(*args, color=FG, fs=30, **kw):
    return MathTex(*args, color=color, font_size=fs, **kw)

def txt(s, color=FG, fs=26, **kw):
    return Text(s, color=color, font_size=fs, **kw)

def make_spring(x_left, x_right, y=0, n=8, color=C1):
    pts = [np.array([x_left, y, 0])]
    w = (x_right - x_left) / (n + 1)
    for i in range(n):
        cx = x_left + w * (i + 0.5)
        pts.append(np.array([cx, y + 0.22 * (1 if i % 2 == 0 else -1), 0]))
    pts.append(np.array([x_right, y, 0]))
    return VMobject(color=color, stroke_width=2.5).set_points_as_corners(pts)

def phase_axes(xl=3.5, yl=3.5, xr=None, yr=None):
    xr = xr or [-1.6, 1.6, 0.8]
    yr = yr or [-1.6, 1.6, 0.8]
    return Axes(x_range=xr, y_range=yr, x_length=xl, y_length=yl,
                axis_config={"color": FG, "stroke_width": 1.5}, tips=False)

# ─────────────────────────────────────────────
# Numerical helpers
# ─────────────────────────────────────────────
def euler_step(x, v, dt, omega=1.0):
    a = -omega**2 * x
    return x + v*dt, v + a*dt

def verlet_step(x, x_prev, dt, omega=1.0):
    a = -omega**2 * x
    return 2*x - x_prev + a*dt**2

def vv_step(x, v, dt, omega=1.0):
    a = -omega**2 * x
    v_half = v + 0.5*a*dt
    x_new = x + v_half*dt
    a_new = -omega**2 * x_new
    v_new = v_half + 0.5*a_new*dt
    return x_new, v_new

def lf_step(x, v_half, dt, omega=1.0):
    a = -omega**2 * x
    v_half_new = v_half + a*dt
    x_new = x + v_half_new*dt
    return x_new, v_half_new

def total_energy(x, v, omega=1.0):
    return 0.5*v**2 + 0.5*omega**2*x**2


# ─────────────────────────────────────────────
# Live-value display using DecimalNumber (no LaTeX per frame)
# ─────────────────────────────────────────────
def live_row(label_str, tracker_fn, color=C4, fs=22, label_color=GRAY):
    """
    Returns a VGroup with a static Text label and a DecimalNumber that
    updates via add_updater — no LaTeX compilation per frame.
    """
    lbl = Text(label_str, color=label_color, font_size=fs)
    num = DecimalNumber(tracker_fn(), num_decimal_places=3,
                        color=color, font_size=fs)
    num.add_updater(lambda m: m.set_value(tracker_fn()))
    row = VGroup(lbl, num).arrange(RIGHT, buff=0.12)
    return row


# ═══════════════════════════════════════════════════════════════
# Scene 0 — F=ma → Lagrangian → Hamiltonian
# ═══════════════════════════════════════════════════════════════
class Scene0_Mechanics(Scene):
    def construct(self):
        omega, A = 1.5, 1.0
        x0_eq = -2.5

        # ── oscillator ──
        wall = Rectangle(width=0.25, height=2.0, color=FG, fill_color=GRAY, fill_opacity=0.4
                         ).move_to(LEFT*5.8)
        hatch = VGroup(*[
            Line(wall.get_right()+UP*(0.6-0.24*i),
                 wall.get_right()+UP*(0.6-0.24*i)+LEFT*0.22+DOWN*0.22,
                 color=GRAY, stroke_width=1.2) for i in range(6)])
        spring = make_spring(-5.55, x0_eq - 0.32)
        mass   = Square(0.6, color=C2, fill_color=C2, fill_opacity=0.85
                        ).move_to(RIGHT*x0_eq)
        m_lbl  = tex("m", color=WHITE, fs=22).move_to(mass)
        self.play(FadeIn(wall), FadeIn(hatch), Create(spring),
                  FadeIn(mass), Write(m_lbl))

        tracker = ValueTracker(0.0)

        def xval(t): return A*np.cos(omega*t)
        def vval(t): return -A*omega*np.sin(omega*t)
        def aval(t): return -omega**2*A*np.cos(omega*t)

        # live readout — DecimalNumber, no LaTeX per frame
        x_lbl = Text("x =", color=C1, font_size=20).shift(LEFT*2.5 + DOWN*1.1)
        x_num = DecimalNumber(xval(0), num_decimal_places=3, color=C1, font_size=20
                              ).next_to(x_lbl, RIGHT, buff=0.1)
        x_num.add_updater(lambda m: m.set_value(xval(tracker.get_value())))

        v_lbl = Text("v =", color=C3, font_size=20).next_to(x_lbl, DOWN, buff=0.2)
        v_num = DecimalNumber(vval(0), num_decimal_places=3, color=C3, font_size=20
                              ).next_to(v_lbl, RIGHT, buff=0.1)
        v_num.add_updater(lambda m: m.set_value(vval(tracker.get_value())))

        a_lbl = Text("a =", color=C2, font_size=20).next_to(v_lbl, DOWN, buff=0.2)
        a_num = DecimalNumber(aval(0), num_decimal_places=3, color=C2, font_size=20
                              ).next_to(a_lbl, RIGHT, buff=0.1)
        a_num.add_updater(lambda m: m.set_value(aval(tracker.get_value())))

        def spring_upd(mob):
            mob.become(make_spring(-5.55, x0_eq + xval(tracker.get_value()) - 0.32))
        def mass_upd(mob):
            mob.move_to(np.array([x0_eq + xval(tracker.get_value()), 0, 0]))
        def mlbl_upd(mob):
            mob.move_to(mass.get_center())

        spring.add_updater(spring_upd)
        mass.add_updater(mass_upd)
        m_lbl.add_updater(mlbl_upd)
        self.add(x_lbl, x_num, v_lbl, v_num, a_lbl, a_num)

        # ── formula pipeline ──
        f_newton = tex(r"F = ma", color=C1, fs=36).shift(RIGHT*2.5 + UP*2.6)
        f_spring = tex(r"F = -kx", color=C2, fs=32).next_to(f_newton, DOWN, buff=0.3)
        f_ode    = tex(r"m\ddot{x} = -kx", color=FG, fs=32).next_to(f_spring, DOWN, buff=0.3)
        self.play(Write(f_newton))
        self.play(tracker.animate.set_value(2*PI/omega), run_time=3, rate_func=linear)
        self.play(Write(f_spring), Write(f_ode))
        self.wait(0.4)

        # Lagrangian
        f_L  = tex(r"L = T - V", color=C3, fs=32).next_to(f_ode, DOWN, buff=0.5)
        f_EL = tex(r"\frac{d}{dt}\frac{\partial L}{\partial \dot{x}}-\frac{\partial L}{\partial x}=0",
                   color=C3, fs=26).next_to(f_L, DOWN, buff=0.25)
        self.play(Write(f_L), Write(f_EL))
        self.play(tracker.animate.set_value(4*PI/omega), run_time=3, rate_func=linear)
        self.wait(0.3)

        # Hamiltonian
        self.play(FadeOut(f_L), FadeOut(f_EL))
        f_H   = tex(r"H = \frac{p^2}{2m}+\frac{1}{2}kx^2", color=C4, fs=30
                    ).next_to(f_ode, DOWN, buff=0.5)
        f_Heq = tex(r"\dot{x}=\frac{\partial H}{\partial p},\quad \dot{p}=-\frac{\partial H}{\partial x}",
                    color=C4, fs=26).next_to(f_H, DOWN, buff=0.25)
        self.play(Write(f_H), Write(f_Heq))
        self.play(tracker.animate.set_value(6*PI/omega), run_time=3, rate_func=linear)
        self.wait(0.6)

        spring.clear_updaters(); mass.clear_updaters(); m_lbl.clear_updaters()
        x_num.clear_updaters(); v_num.clear_updaters(); a_num.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.wait(0.2)


# ═══════════════════════════════════════════════════════════════
# Scene 1 — Phase space + Liouville theorem
# ═══════════════════════════════════════════════════════════════
class Scene1_PhaseSpace(Scene):
    def construct(self):
        omega, A = 1.0, 1.0
        x0_eq = -2.5

        # ── oscillator ──
        wall = Rectangle(width=0.22, height=1.8, color=FG, fill_color=GRAY, fill_opacity=0.4
                         ).move_to(LEFT*5.8)
        hatch = VGroup(*[
            Line(wall.get_right()+UP*(0.5-0.22*i),
                 wall.get_right()+UP*(0.5-0.22*i)+LEFT*0.2+DOWN*0.2,
                 color=GRAY, stroke_width=1.2) for i in range(5)])
        spring = make_spring(-5.55, x0_eq - 0.32)
        mass   = Square(0.55, color=C2, fill_color=C2, fill_opacity=0.85
                        ).move_to(RIGHT*x0_eq)
        m_lbl  = tex("m", color=WHITE, fs=20).move_to(mass)
        self.play(FadeIn(wall), FadeIn(hatch), Create(spring),
                  FadeIn(mass), Write(m_lbl))

        tracker = ValueTracker(0.0)
        def xv(t): return A*np.cos(omega*t), -A*omega*np.sin(omega*t)

        def spring_upd(mob):
            x, _ = xv(tracker.get_value())
            mob.become(make_spring(-5.55, x0_eq + x - 0.32))
        def mass_upd(mob):
            x, _ = xv(tracker.get_value())
            mob.move_to(np.array([x0_eq + x, 0, 0]))
        def mlbl_upd(mob):
            mob.move_to(mass.get_center())

        spring.add_updater(spring_upd)
        mass.add_updater(mass_upd)
        m_lbl.add_updater(mlbl_upd)

        # ── right-top: equations + live values (DecimalNumber) ──
        eq_xdot = tex(r"\dot{x}=v", color=C1, fs=28).shift(RIGHT*2.0 + UP*2.8)
        eq_vdot = tex(r"\dot{v}=-\omega^2 x", color=C2, fs=28).next_to(eq_xdot, DOWN, buff=0.22)
        eq_E    = tex(r"E=\tfrac{1}{2}v^2+\tfrac{1}{2}\omega^2 x^2", color=C3, fs=24
                      ).next_to(eq_vdot, DOWN, buff=0.22)
        self.play(Write(eq_xdot), Write(eq_vdot), Write(eq_E))

        # live x, v, E using DecimalNumber
        x_row_lbl = Text("x =", color=C1, font_size=22).next_to(eq_E, DOWN, buff=0.35)
        x_row_num = DecimalNumber(A, num_decimal_places=3, color=C1, font_size=22
                                  ).next_to(x_row_lbl, RIGHT, buff=0.1)
        x_row_num.add_updater(lambda m: m.set_value(xv(tracker.get_value())[0]))

        v_row_lbl = Text("v =", color=C2, font_size=22).next_to(x_row_lbl, DOWN, buff=0.18)
        v_row_num = DecimalNumber(0.0, num_decimal_places=3, color=C2, font_size=22
                                  ).next_to(v_row_lbl, RIGHT, buff=0.1)
        v_row_num.add_updater(lambda m: m.set_value(xv(tracker.get_value())[1]))

        E0 = total_energy(A, 0.0, omega)
        E_row_lbl = Text("E =", color=C3, font_size=22).next_to(v_row_lbl, DOWN, buff=0.18)
        E_row_num = DecimalNumber(E0, num_decimal_places=4, color=C3, font_size=22
                                  ).next_to(E_row_lbl, RIGHT, buff=0.1)
        E_row_num.add_updater(lambda m: m.set_value(
            total_energy(*xv(tracker.get_value()), omega)))
        self.add(x_row_lbl, x_row_num, v_row_lbl, v_row_num, E_row_lbl, E_row_num)

        # ── right-bottom: phase portrait ──
        ax = phase_axes(3.2, 3.2)
        ax.shift(RIGHT*2.2 + DOWN*1.5)
        xl = tex("x", fs=18, color=FG).next_to(ax.x_axis, RIGHT, buff=0.08)
        vl = tex("v", fs=18, color=FG).next_to(ax.y_axis, UP, buff=0.08)
        self.play(Create(ax), Write(xl), Write(vl))

        theta = np.linspace(0, 2*np.pi, 300)
        ell_pts = [ax.c2p(A*np.cos(t), -A*omega*np.sin(t)) for t in theta]
        ellipse = VMobject(color=LG, stroke_width=1.5).set_points_smoothly(ell_pts)
        self.play(Create(ellipse))

        phase_dot   = Dot(ax.c2p(A, 0), radius=0.07, color=C2)
        phase_trail = VMobject(color=C2, stroke_width=2.0)
        phase_trail.set_points_as_corners([ax.c2p(A, 0), ax.c2p(A, 0)])
        self.add(phase_trail, phase_dot)

        def phase_dot_upd(mob):
            mob.move_to(ax.c2p(*xv(tracker.get_value())))
        def phase_trail_upd(mob):
            mob.add_points_as_corners([ax.c2p(*xv(tracker.get_value()))])

        phase_dot.add_updater(phase_dot_upd)
        phase_trail.add_updater(phase_trail_upd)

        # ── Liouville cloud ──
        rng = np.random.default_rng(42)
        ic_offsets = rng.uniform(-0.12, 0.12, (25, 2))
        cloud_ics = []
        for dx, dv in ic_offsets:
            xi, vi = A + dx, dv
            R   = np.sqrt(xi**2 + (vi/omega)**2)
            phi = np.arctan2(-vi/omega, xi)
            cloud_ics.append((R, phi))

        cloud_dots = VGroup(*[
            Dot(ax.c2p(R*np.cos(phi), -R*omega*np.sin(phi)),
                radius=0.04, color=C4)
            for R, phi in cloud_ics
        ])
        self.play(FadeIn(cloud_dots))

        def cloud_upd(mob):
            t = tracker.get_value()
            for dot, (R, phi) in zip(mob, cloud_ics):
                dot.move_to(ax.c2p(R*np.cos(omega*t+phi),
                                   -R*omega*np.sin(omega*t+phi)))
        cloud_dots.add_updater(cloud_upd)

        liouville = tex(r"\frac{d\rho}{dt}=0", color=C4, fs=28
                        ).next_to(ax, DOWN, buff=0.18)
        self.play(Write(liouville))

        self.play(tracker.animate.set_value(4*np.pi/omega),
                  run_time=6, rate_func=linear)

        for mob in [spring, mass, m_lbl, phase_dot, phase_trail,
                    cloud_dots, x_row_num, v_row_num, E_row_num]:
            mob.clear_updaters()
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.wait(0.2)


# ═══════════════════════════════════════════════════════════════
# Scene 2 — Many-body: trajectory intractable, statistics useful
# ═══════════════════════════════════════════════════════════════
class Scene2_ManyBody(Scene):
    def construct(self):
        rng = np.random.default_rng(7)
        N   = 18

        box = Rectangle(width=4.0, height=4.0, color=FG, stroke_width=2).shift(LEFT*3.2)
        self.play(Create(box))

        pos    = rng.uniform(-1.7, 1.7, (N, 2))
        vel    = rng.uniform(-0.6, 0.6, (N, 2))
        colors = [C1, C2, C3, C4, GRAY]*4
        dots   = VGroup(*[
            Dot(np.array([pos[i,0]-3.2, pos[i,1], 0]),
                radius=0.14, color=colors[i%5])
            for i in range(N)
        ])
        self.play(FadeIn(dots))

        tracker = ValueTracker(0.0)
        dt_sim  = 0.016
        state   = {"pos": pos.copy(), "vel": vel.copy(), "t_last": 0.0}

        def bounce_update(mob):
            t     = tracker.get_value()
            steps = int((t - state["t_last"]) / dt_sim)
            for _ in range(steps):
                state["pos"] += state["vel"] * dt_sim
                for i in range(N):
                    for d in range(2):
                        lim = 1.7 - 0.14
                        if state["pos"][i,d] >  lim: state["pos"][i,d] =  lim; state["vel"][i,d] *= -1
                        if state["pos"][i,d] < -lim: state["pos"][i,d] = -lim; state["vel"][i,d] *= -1
            state["t_last"] = t - (t - state["t_last"]) % dt_sim
            for i, dot in enumerate(mob):
                dot.move_to(np.array([state["pos"][i,0]-3.2, state["pos"][i,1], 0]))

        dots.add_updater(bounce_update)

        # ── right-top: equation ──
        eq2 = tex(r"m_i\ddot{\mathbf{r}}_i=\sum_{j\neq i}\mathbf{F}_{ij}",
                  color=C1, fs=30).shift(RIGHT*2.5 + UP*2.2)
        dim = tex(r"\text{state dim}=6N", color=GRAY, fs=24
                  ).next_to(eq2, DOWN, buff=0.3)
        self.play(Write(eq2), Write(dim))

        # ── right-bottom: running observable ──
        ax_obs = Axes(x_range=[0,8,2], y_range=[0.1,0.7,0.2],
                      x_length=4.5, y_length=2.2,
                      axis_config={"color": FG, "stroke_width": 1.3}, tips=False
                      ).shift(RIGHT*2.5 + DOWN*1.5)
        obs_xl = tex("t", fs=18, color=FG).next_to(ax_obs.x_axis, RIGHT, buff=0.08)
        obs_yl = tex(r"\langle E_k\rangle", fs=18, color=FG
                     ).next_to(ax_obs.y_axis, UP, buff=0.08)
        self.play(Create(ax_obs), Write(obs_xl), Write(obs_yl))

        t_obs   = np.linspace(0, 8, 200)
        raw_ek  = 0.5 * np.sum(vel**2) / N
        noise   = 0.12*np.exp(-t_obs*0.5)*np.sin(7*t_obs) + 0.015*rng.standard_normal(200)
        obs_vals = np.clip(raw_ek + noise, 0.12, 0.68)

        obs_curve = VMobject(color=C3, stroke_width=2.5)
        obs_curve.set_points_as_corners([ax_obs.c2p(t_obs[0], obs_vals[0]),
                                         ax_obs.c2p(t_obs[0], obs_vals[0])])
        avg_line = ax_obs.plot(lambda t: raw_ek, color=C4, stroke_width=1.5)
        avg_lbl  = tex(r"\langle A\rangle=\frac{1}{\tau}\int_0^\tau A\,dt",
                       color=C4, fs=20).next_to(ax_obs, DOWN, buff=0.15)
        self.play(Create(avg_line), Write(avg_lbl))
        self.add(obs_curve)

        obs_tracker = ValueTracker(0)

        def obs_upd(mob):
            idx = max(2, int(obs_tracker.get_value()))
            pts = [ax_obs.c2p(t_obs[i], obs_vals[i]) for i in range(idx)]
            mob.set_points_as_corners(pts)

        obs_curve.add_updater(obs_upd)

        self.play(
            tracker.animate.set_value(8.0),
            obs_tracker.animate.set_value(199),
            run_time=8, rate_func=linear
        )

        dots.clear_updaters(); obs_curve.clear_updaters()
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.wait(0.2)


# ═══════════════════════════════════════════════════════════════
# Scene 3 — Discretization
# ═══════════════════════════════════════════════════════════════
class Scene3_Discretization(Scene):
    def construct(self):
        omega, A = 1.0, 1.0

        ax = Axes(x_range=[0,10,2], y_range=[-1.4,1.4,0.7],
                  x_length=9.0, y_length=4.5,
                  axis_config={"color": FG, "stroke_width": 1.5}, tips=False
                  ).shift(DOWN*0.3)
        xl    = tex("t", fs=22, color=FG).next_to(ax.x_axis, RIGHT)
        yl    = tex("x(t)", fs=22, color=FG).next_to(ax.y_axis, UP)
        exact = ax.plot(lambda t: A*np.cos(omega*t), color=C3, stroke_width=2.5)
        self.play(Create(ax), Write(xl), Write(yl), Create(exact))

        fd = tex(r"\frac{dx}{dt}\approx\frac{x(t+\Delta t)-x(t)}{\Delta t}",
                 color=C1, fs=30).to_edge(UP, buff=0.35)
        self.play(Write(fd))

        # large dt
        dt_large = 1.5
        t_pts = np.arange(0, 10.01, dt_large)
        dots_l = VGroup(*[Dot(ax.c2p(t, A*np.cos(omega*t)), radius=0.07, color=C2)
                          for t in t_pts])
        lines_l = VGroup(*[
            Line(ax.c2p(t_pts[i], A*np.cos(omega*t_pts[i])),
                 ax.c2p(t_pts[i+1], A*np.cos(omega*t_pts[i+1])),
                 color=C2, stroke_width=2.0)
            for i in range(len(t_pts)-1)
        ])
        dt_lbl = tex(r"\Delta t=1.5", color=C2, fs=26
                     ).to_edge(UP, buff=0.35).shift(RIGHT*4.5)
        self.play(FadeIn(dots_l), Create(lines_l), Write(dt_lbl))
        self.wait(0.8)

        # small dt
        dt_small = 0.5
        t_pts2 = np.arange(0, 10.01, dt_small)
        dots_s = VGroup(*[Dot(ax.c2p(t, A*np.cos(omega*t)), radius=0.05, color=C1)
                          for t in t_pts2])
        lines_s = VGroup(*[
            Line(ax.c2p(t_pts2[i], A*np.cos(omega*t_pts2[i])),
                 ax.c2p(t_pts2[i+1], A*np.cos(omega*t_pts2[i+1])),
                 color=C1, stroke_width=1.5)
            for i in range(len(t_pts2)-1)
        ])
        dt_lbl2 = tex(r"\Delta t=0.5", color=C1, fs=26
                      ).to_edge(UP, buff=0.35).shift(RIGHT*4.5)
        self.play(
            Transform(dots_l, dots_s),
            Transform(lines_l, lines_s),
            Transform(dt_lbl, dt_lbl2)
        )
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.wait(0.2)


# ═══════════════════════════════════════════════════════════════
# Shared integrator base — Scenes 4-7
# ═══════════════════════════════════════════════════════════════
class _IntegratorBase(Scene):
    algo_lines  = []
    phase_color = C2
    dt          = 0.25
    n_steps     = 80
    omega       = 1.0
    A           = 1.0

    def step_fn(self, x, v, dt):
        raise NotImplementedError

    def _precompute(self):
        xs, vs = [self.A], [0.0]
        for _ in range(self.n_steps):
            xn, vn = self.step_fn(xs[-1], vs[-1], self.dt)
            xs.append(xn); vs.append(vn)
        return np.array(xs), np.array(vs)

    def construct(self):
        omega   = self.omega
        A       = self.A
        dt      = self.dt
        n_steps = self.n_steps

        xs, vs = self._precompute()
        Es = np.array([total_energy(xs[i], vs[i], omega) for i in range(len(xs))])
        ts = np.arange(len(xs)) * dt

        x0_eq = -2.5

        # ── oscillator ──
        wall = Rectangle(width=0.22, height=1.8, color=FG, fill_color=GRAY, fill_opacity=0.4
                         ).move_to(LEFT*5.8)
        hatch = VGroup(*[
            Line(wall.get_right()+UP*(0.5-0.22*i),
                 wall.get_right()+UP*(0.5-0.22*i)+LEFT*0.2+DOWN*0.2,
                 color=GRAY, stroke_width=1.2) for i in range(5)])
        spring = make_spring(-5.55, x0_eq + xs[0] - 0.32)
        mass   = Square(0.55, color=C2, fill_color=C2, fill_opacity=0.85
                        ).move_to(np.array([x0_eq + xs[0], 0, 0]))
        m_lbl  = tex("m", color=WHITE, fs=20).move_to(mass)
        self.play(FadeIn(wall), FadeIn(hatch), Create(spring),
                  FadeIn(mass), Write(m_lbl))

        step_idx = ValueTracker(0)

        def si(cap=n_steps): return min(int(step_idx.get_value()), cap)

        def spring_upd(mob): mob.become(make_spring(-5.55, x0_eq + xs[si()] - 0.32))
        def mass_upd(mob):   mob.move_to(np.array([x0_eq + xs[si()], 0, 0]))
        def mlbl_upd(mob):   mob.move_to(mass.get_center())

        spring.add_updater(spring_upd)
        mass.add_updater(mass_upd)
        m_lbl.add_updater(mlbl_upd)

        # step counter (Text, no LaTeX)
        n_lbl = Text("n =", color=GRAY, font_size=20).shift(LEFT*2.5 + DOWN*1.1)
        n_num = Integer(0, color=GRAY, font_size=20).next_to(n_lbl, RIGHT, buff=0.1)
        n_num.add_updater(lambda m: m.set_value(si()))
        self.add(n_lbl, n_num)

        # ── right-top: algorithm ──
        algo = VGroup(*[tex(l, color=C1, fs=22) for l in self.algo_lines])
        algo.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        algo.shift(RIGHT*2.2 + UP*2.2)
        algo_box = SurroundingRectangle(algo, color=C1, buff=0.15)
        self.play(Create(algo_box), Write(algo))

        # live values — DecimalNumber
        xn_lbl = Text("x =", color=C1, font_size=22).next_to(algo, DOWN, buff=0.35)
        xn_num = DecimalNumber(xs[0], num_decimal_places=4, color=C1, font_size=22
                               ).next_to(xn_lbl, RIGHT, buff=0.1)
        xn_num.add_updater(lambda m: m.set_value(xs[si()]))

        vn_lbl = Text("v =", color=C3, font_size=22).next_to(xn_lbl, DOWN, buff=0.18)
        vn_num = DecimalNumber(vs[0], num_decimal_places=4, color=C3, font_size=22
                               ).next_to(vn_lbl, RIGHT, buff=0.1)
        vn_num.add_updater(lambda m: m.set_value(vs[si()]))

        En_lbl = Text("E =", color=C4, font_size=22).next_to(vn_lbl, DOWN, buff=0.18)
        En_num = DecimalNumber(Es[0], num_decimal_places=5, color=C4, font_size=22
                               ).next_to(En_lbl, RIGHT, buff=0.1)
        En_num.add_updater(lambda m: m.set_value(Es[si()]))

        self.add(xn_lbl, xn_num, vn_lbl, vn_num, En_lbl, En_num)

        # ── right-bottom: phase portrait ──
        ph_xr = [-max(2.0, np.abs(xs).max()*1.1), max(2.0, np.abs(xs).max()*1.1), 1.0]
        ph_yr = [-max(2.0, np.abs(vs).max()*1.1), max(2.0, np.abs(vs).max()*1.1), 1.0]
        ax_ph = phase_axes(2.8, 2.8, xr=ph_xr, yr=ph_yr)
        ax_ph.shift(RIGHT*1.5 + DOWN*1.7)
        ph_xl = tex("x", fs=16, color=FG).next_to(ax_ph.x_axis, RIGHT, buff=0.06)
        ph_vl = tex("v", fs=16, color=FG).next_to(ax_ph.y_axis, UP, buff=0.06)

        theta   = np.linspace(0, 2*np.pi, 200)
        ell_pts = [ax_ph.c2p(A*np.cos(t), -A*omega*np.sin(t)) for t in theta]
        ell     = VMobject(color=LG, stroke_width=1.2).set_points_smoothly(ell_pts)

        # energy axes
        e_min = max(0.1, Es.min()*0.85)
        e_max = Es.max()*1.15 + 0.05
        ax_en = Axes(x_range=[0, n_steps, n_steps//4],
                     y_range=[e_min, e_max, (e_max-e_min)/4],
                     x_length=3.2, y_length=1.6,
                     axis_config={"color": FG, "stroke_width": 1.3}, tips=False)
        ax_en.shift(RIGHT*4.8 + DOWN*1.7)
        en_xl = tex("n", fs=16, color=FG).next_to(ax_en.x_axis, RIGHT, buff=0.06)
        en_yl = tex("E", fs=16, color=FG).next_to(ax_en.y_axis, UP, buff=0.06)

        self.play(Create(ax_ph), Write(ph_xl), Write(ph_vl), Create(ell),
                  Create(ax_en), Write(en_xl), Write(en_yl))

        phase_trail = VMobject(color=self.phase_color, stroke_width=2.2)
        phase_trail.set_points_as_corners([ax_ph.c2p(xs[0], vs[0]),
                                           ax_ph.c2p(xs[0], vs[0])])
        phase_dot = Dot(ax_ph.c2p(xs[0], vs[0]), radius=0.06, color=self.phase_color)

        energy_trail = VMobject(color=C4, stroke_width=2.0)
        energy_trail.set_points_as_corners([ax_en.c2p(0, Es[0]),
                                            ax_en.c2p(0, Es[0])])
        self.add(phase_trail, phase_dot, energy_trail)

        def phase_trail_upd(mob):
            i = si()
            if i >= 1:
                mob.set_points_as_corners([ax_ph.c2p(xs[j], vs[j]) for j in range(i+1)])
        def phase_dot_upd(mob):
            mob.move_to(ax_ph.c2p(xs[si()], vs[si()]))
        def energy_trail_upd(mob):
            i = si()
            if i >= 1:
                mob.set_points_as_corners([ax_en.c2p(j, Es[j]) for j in range(i+1)])

        phase_trail.add_updater(phase_trail_upd)
        phase_dot.add_updater(phase_dot_upd)
        energy_trail.add_updater(energy_trail_upd)

        self.play(step_idx.animate.set_value(n_steps), run_time=8, rate_func=linear)

        for mob in [spring, mass, m_lbl, phase_trail, phase_dot, energy_trail,
                    n_num, xn_num, vn_num, En_num]:
            mob.clear_updaters()
        self.wait(0.6)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.wait(0.2)


# ═══════════════════════════════════════════════════════════════
# Scene 4 — Explicit Euler
# ═══════════════════════════════════════════════════════════════
class Scene4_ExplicitEuler(_IntegratorBase):
    algo_lines  = [
        r"a_n = -\omega^2 x_n",
        r"x_{n+1}=x_n+v_n\Delta t",
        r"v_{n+1}=v_n+a_n\Delta t",
    ]
    phase_color = C2
    dt          = 0.22
    n_steps     = 80

    def step_fn(self, x, v, dt):
        return euler_step(x, v, dt, self.omega)


# ═══════════════════════════════════════════════════════════════
# Scene 5 — Verlet
# ═══════════════════════════════════════════════════════════════
class Scene5_Verlet(_IntegratorBase):
    algo_lines  = [
        r"a_n = -\omega^2 x_n",
        r"x_{n+1}=2x_n-x_{n-1}+a_n\Delta t^2",
    ]
    phase_color = C3
    dt          = 0.22
    n_steps     = 80
    _x_prev     = None

    def step_fn(self, x, v, dt):
        if self._x_prev is None:
            a0 = -self.omega**2 * x
            self._x_prev = x - v*dt + 0.5*a0*dt**2
        x_prev = self._x_prev
        x_new  = verlet_step(x, x_prev, dt, self.omega)
        v_new  = (x_new - x_prev) / (2*dt)
        self._x_prev = x
        return x_new, v_new

    def _precompute(self):
        self._x_prev = None
        return super()._precompute()


# ═══════════════════════════════════════════════════════════════
# Scene 6 — Velocity Verlet
# ═══════════════════════════════════════════════════════════════
class Scene6_VelocityVerlet(_IntegratorBase):
    algo_lines  = [
        r"v_{n+\frac{1}{2}}=v_n+\tfrac{1}{2}a_n\Delta t",
        r"x_{n+1}=x_n+v_{n+\frac{1}{2}}\Delta t",
        r"v_{n+1}=v_{n+\frac{1}{2}}+\tfrac{1}{2}a_{n+1}\Delta t",
    ]
    phase_color = C1
    dt          = 0.22
    n_steps     = 80

    def step_fn(self, x, v, dt):
        return vv_step(x, v, dt, self.omega)


# ═══════════════════════════════════════════════════════════════
# Scene 7 — Leapfrog
# ═══════════════════════════════════════════════════════════════
class Scene7_Leapfrog(_IntegratorBase):
    algo_lines  = [
        r"v_{n+\frac{1}{2}}=v_{n-\frac{1}{2}}+a_n\Delta t",
        r"x_{n+1}=x_n+v_{n+\frac{1}{2}}\Delta t",
    ]
    phase_color = C4
    dt          = 0.22
    n_steps     = 80
    _v_half     = None

    def step_fn(self, x, v, dt):
        if self._v_half is None:
            a0 = -self.omega**2 * x
            self._v_half = v - 0.5*a0*dt
        x_new, v_half_new = lf_step(x, self._v_half, dt, self.omega)
        a_new = -self.omega**2 * x_new
        v_full = v_half_new - 0.5*a_new*dt
        self._v_half = v_half_new
        return x_new, v_full

    def _precompute(self):
        self._v_half = None
        return super()._precompute()
