"""
Part 1: Molecular Dynamics — From Newton's Second Law to Numerical Integrators
Manim Community Edition animation, white background.

Render command:
    manim -pqh part1/integrators.py MDIntegrators

Or render individually:
    manim -pqh part1/integrators.py Scene0_MechanicsFoundations
    manim -pqh part1/integrators.py Scene1_NewtonSpring
    manim -pqh part1/integrators.py Scene2_ManyBody
    manim -pqh part1/integrators.py Scene3_NumericalPitfalls
    manim -pqh part1/integrators.py Scene4_ExplicitEuler
    manim -pqh part1/integrators.py Scene5_Verlet
    manim -pqh part1/integrators.py Scene6_VelocityVerlet
    manim -pqh part1/integrators.py Scene7_Leapfrog
    manim -pqh part1/integrators.py Scene8_Comparison
"""

from manim import *
import numpy as np

# ─────────────────────────────────────────────
# Global style constants
# ─────────────────────────────────────────────
BG = WHITE
FG = BLACK
ACCENT1 = "#1565C0"   # deep blue
ACCENT2 = "#B71C1C"   # deep red
ACCENT3 = "#1B5E20"   # deep green
ACCENT4 = "#E65100"   # deep orange
GRAY = "#555555"
LIGHT_GRAY = "#BBBBBB"

config.background_color = BG


# ─────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────
def styled_tex(*args, color=FG, font_size=30, **kwargs):
    return MathTex(*args, color=color, font_size=font_size, **kwargs)


def styled_text(s, color=FG, font_size=28, **kwargs):
    return Text(s, color=color, font_size=font_size, **kwargs)


def section_title(text):
    return Text(text, color=ACCENT1, font_size=40, weight=BOLD)


def small_note(text, color=GRAY):
    return Text(text, color=color, font_size=20)


def fade_all(scene):
    if scene.mobjects:
        scene.play(*[FadeOut(mob) for mob in list(scene.mobjects)])
        scene.wait(0.2)


# ─────────────────────────────────────────────
# Numerical helper data
# ─────────────────────────────────────────────
def compute_exact_phase(A=1.0, omega=1.0, n=240):
    theta = np.linspace(0, 2 * np.pi, n)
    x = A * np.cos(theta)
    v = -A * omega * np.sin(theta)
    return np.column_stack([x, v])


def compute_euler_phase(dt=0.22, steps=120, A=1.0, omega=1.0):
    x, v = A, 0.0
    pts = [(x, v)]
    for _ in range(steps):
        a = -(omega ** 2) * x
        x = x + v * dt
        v = v + a * dt
        pts.append((x, v))
    return np.array(pts)


def compute_verlet_phase(dt=0.22, steps=120, A=1.0, omega=1.0):
    x0 = A
    v0 = 0.0
    a0 = -(omega ** 2) * x0
    x_prev = x0 - v0 * dt + 0.5 * a0 * dt * dt
    x = x0
    pts = [(x0, v0)]
    for _ in range(steps):
        a = -(omega ** 2) * x
        x_next = 2 * x - x_prev + a * dt * dt
        v_est = (x_next - x_prev) / (2 * dt)
        pts.append((x_next, v_est))
        x_prev, x = x, x_next
    return np.array(pts)


def compute_velocity_verlet_phase(dt=0.22, steps=120, A=1.0, omega=1.0):
    x, v = A, 0.0
    pts = [(x, v)]
    for _ in range(steps):
        a = -(omega ** 2) * x
        v_half = v + 0.5 * a * dt
        x = x + v_half * dt
        a_new = -(omega ** 2) * x
        v = v_half + 0.5 * a_new * dt
        pts.append((x, v))
    return np.array(pts)


def compute_leapfrog_phase(dt=0.22, steps=120, A=1.0, omega=1.0):
    x = A
    v_half = 0.0 - 0.5 * (omega ** 2) * x * dt
    pts = [(x, 0.0)]
    for _ in range(steps):
        a = -(omega ** 2) * x
        v_half = v_half + a * dt
        x = x + v_half * dt
        a_new = -(omega ** 2) * x
        v_full = v_half - 0.5 * a_new * dt
        pts.append((x, v_full))
    return np.array(pts)


def compute_energy_curve(method, dt=0.22, steps=80, A=1.0, omega=1.0, m=1.0):
    k = m * omega ** 2
    if method == "euler":
        pts = compute_euler_phase(dt=dt, steps=steps, A=A, omega=omega)
    elif method == "verlet":
        pts = compute_verlet_phase(dt=dt, steps=steps, A=A, omega=omega)
    elif method == "velocity_verlet":
        pts = compute_velocity_verlet_phase(dt=dt, steps=steps, A=A, omega=omega)
    else:
        pts = compute_leapfrog_phase(dt=dt, steps=steps, A=A, omega=omega)

    x = pts[:, 0]
    v = pts[:, 1]
    e = 0.5 * m * v ** 2 + 0.5 * k * x ** 2
    t = np.arange(len(e)) * dt
    return t, e


# ─────────────────────────────────────────────
# Visual helpers
# ─────────────────────────────────────────────
def phase_axes(x_length=3.4, y_length=3.4):
    return Axes(
        x_range=[-1.6, 1.6, 0.8],
        y_range=[-1.6, 1.6, 0.8],
        x_length=x_length,
        y_length=y_length,
        axis_config={"color": FG, "stroke_width": 1.5},
        tips=False,
    )


def add_phase_labels(ax, x_label="x", y_label="v", size=20):
    xl = styled_tex(x_label, font_size=size, color=FG).next_to(ax.x_axis, RIGHT, buff=0.1)
    yl = styled_tex(y_label, font_size=size, color=FG).next_to(ax.y_axis, UP, buff=0.1)
    return VGroup(xl, yl)


def phase_path(ax, pts, color, stroke_width=2.5):
    mapped = [ax.c2p(x, v) for x, v in pts]
    return VMobject(color=color, stroke_width=stroke_width).set_points_smoothly(mapped)


def traced_dot(ax, pts, color):
    dot = Dot(ax.c2p(*pts[0]), radius=0.05, color=color)
    path = phase_path(ax, pts, color=color)
    return dot, path


def energy_axes(x_length=7.5, y_length=1.8, y_range=None):
    if y_range is None:
        y_range = [0.35, 0.95, 0.1]
    return Axes(
        x_range=[0, 18, 3],
        y_range=y_range,
        x_length=x_length,
        y_length=y_length,
        axis_config={"color": FG, "stroke_width": 1.3},
        tips=False,
    )


def make_spring(x_left, x_right, y=0, n_coils=8, color=ACCENT1):
    pts = [np.array([x_left, y, 0])]
    coil_w = (x_right - x_left) / (n_coils + 1)
    for i in range(n_coils):
        cx = x_left + coil_w * (i + 0.5)
        pts.append(np.array([cx, y + 0.22 * (1 if i % 2 == 0 else -1), 0]))
    pts.append(np.array([x_right, y, 0]))
    return VMobject(color=color, stroke_width=2.5).set_points_as_corners(pts)


# ─────────────────────────────────────────────
# Scene 0: Mechanics Foundations
# ─────────────────────────────────────────────
class Scene0_MechanicsFoundations(Scene):
    def construct(self):
        title = section_title("0. Mechanics Foundations")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        newton_block = VGroup(
            styled_text("Newtonian mechanics", color=ACCENT1, font_size=28, weight=BOLD),
            styled_tex(r"\mathbf{F}=m\mathbf{a}", font_size=34, color=FG),
            styled_tex(r"m\ddot{x}=-kx", font_size=34, color=ACCENT2),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        newton_block.to_edge(LEFT, buff=1.0).shift(UP * 0.8)
        self.play(Write(newton_block))
        self.wait(0.4)

        lag_block = VGroup(
            styled_text("Lagrangian", color=ACCENT3, font_size=28, weight=BOLD),
            styled_tex(r"L=T-V=\frac{1}{2}m\dot{q}^2-\frac{1}{2}kq^2", font_size=28, color=FG),
            styled_tex(r"\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right)-\frac{\partial L}{\partial q}=0", font_size=24, color=FG),
            styled_tex(r"m\ddot{q}+kq=0", font_size=30, color=ACCENT3),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        lag_block.to_edge(LEFT, buff=1.0).shift(DOWN * 1.4)
        self.play(Write(lag_block))
        self.wait(0.5)

        ham_block = VGroup(
            styled_text("Hamiltonian", color=ACCENT4, font_size=28, weight=BOLD),
            styled_tex(r"H=\frac{p^2}{2m}+\frac{1}{2}kq^2", font_size=28, color=FG),
            styled_tex(r"\dot{q}=\frac{\partial H}{\partial p},\quad \dot{p}=-\frac{\partial H}{\partial q}", font_size=24, color=FG),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        ham_block.to_edge(RIGHT, buff=1.0).shift(UP * 1.8)
        self.play(Write(ham_block))

        ax = phase_axes(3.5, 3.5)
        ax.to_edge(RIGHT, buff=1.0).shift(DOWN * 1.1)
        labels = add_phase_labels(ax, "q", "p", 20)
        exact_pts = compute_exact_phase()
        exact_curve = phase_path(ax, exact_pts, ACCENT3, 2.8)
        orbit_dot = Dot(ax.c2p(*exact_pts[0]), radius=0.06, color=ACCENT3)
        orbit_label = styled_text("phase space: closed orbit", color=ACCENT3, font_size=20)
        orbit_label.next_to(ax, DOWN, buff=0.15)

        self.play(Create(ax), Write(labels))
        self.play(Create(exact_curve), FadeIn(orbit_dot), Write(orbit_label))

        tracker = ValueTracker(0)
        orbit_dot.add_updater(lambda m: m.move_to(ax.c2p(*exact_pts[min(int(tracker.get_value()), len(exact_pts) - 1)])))
        self.play(tracker.animate.set_value(len(exact_pts) - 1), run_time=3.0, rate_func=linear)
        orbit_dot.clear_updaters()

        transition = styled_text("Good integrators should preserve this structure.", color=ACCENT2, font_size=24)
        transition.to_edge(DOWN, buff=0.35)
        self.play(Write(transition))
        self.wait(1.3)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 1: Newton + Spring Mass
# ─────────────────────────────────────────────
class Scene1_NewtonSpring(Scene):
    def construct(self):
        title = section_title("1. Spring-Mass System")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        x_mass = -2.1
        wall = Rectangle(width=0.28, height=2.3, color=FG, fill_color=GRAY, fill_opacity=0.45).move_to(LEFT * 5.7)
        hatch = VGroup(*[
            Line(
                wall.get_right() + UP * (0.72 - 0.24 * i),
                wall.get_right() + UP * (0.72 - 0.24 * i) + LEFT * 0.26 + DOWN * 0.26,
                color=GRAY, stroke_width=1.2,
            )
            for i in range(7)
        ])
        spring = make_spring(-5.35, x_mass - 0.34, y=0.1)
        mass = Square(side_length=0.68, color=ACCENT2, fill_color=ACCENT2, fill_opacity=0.85).move_to(RIGHT * x_mass + UP * 0.1)
        mass_label = styled_tex("m", color=WHITE, font_size=24).move_to(mass.get_center())
        eq_line = DashedLine(mass.get_center() + UP * 1.2, mass.get_center() + DOWN * 1.2, color=LIGHT_GRAY, dash_length=0.1)
        eq_label = small_note("equilibrium")
        eq_label.next_to(eq_line, DOWN, buff=0.1)

        spring_group = VGroup(wall, hatch, spring, mass, mass_label, eq_line, eq_label)
        spring_group.shift(LEFT * 0.1 + DOWN * 0.3)
        self.play(FadeIn(wall), FadeIn(hatch), Create(spring), FadeIn(mass), Write(mass_label), Create(eq_line), FadeIn(eq_label))

        right_block = VGroup(
            styled_tex(r"F=ma", font_size=34, color=ACCENT1),
            styled_tex(r"F_{\rm spring}=-kx", font_size=30, color=ACCENT2),
            styled_tex(r"m\ddot{x}=-kx", font_size=32, color=FG),
            styled_tex(r"x(t)=A\cos(\omega t+\varphi)", font_size=30, color=ACCENT3),
            styled_tex(r"\omega=\sqrt{\frac{k}{m}}", font_size=28, color=ACCENT3),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        right_block.to_edge(RIGHT, buff=0.9).shift(UP * 0.5)
        self.play(Write(right_block))

        energy_box = VGroup(
            styled_text("Conserved quantity", color=ACCENT4, font_size=22, weight=BOLD),
            styled_tex(r"E=\frac{1}{2}mv^2+\frac{1}{2}kx^2=\mathrm{const}", font_size=28, color=ACCENT4),
        ).arrange(DOWN, buff=0.15)
        energy_frame = SurroundingRectangle(energy_box, color=ACCENT4, buff=0.18)
        energy_panel = VGroup(energy_frame, energy_box)
        energy_panel.next_to(right_block, DOWN, buff=0.45)
        self.play(Create(energy_frame), Write(energy_box))

        ax = phase_axes(2.7, 2.7)
        ax.next_to(energy_panel, DOWN, buff=0.3)
        phase_labels = add_phase_labels(ax, "x", "v", 18)
        exact_pts = compute_exact_phase(A=1.0)
        exact_curve = phase_path(ax, exact_pts, ACCENT3, 2.2)
        phase_caption = small_note("exact motion: closed ellipse", color=ACCENT3).next_to(ax, DOWN, buff=0.08)
        self.play(Create(ax), Write(phase_labels), Create(exact_curve), Write(phase_caption))

        tracker = ValueTracker(0.0)
        A = 1.15
        omega_val = 1.55
        y0 = 0.1

        def mass_x(t):
            return x_mass + A * np.cos(omega_val * t)

        def spring_updater(mob):
            xm = mass_x(tracker.get_value())
            mob.become(make_spring(-5.35, xm - 0.34, y=y0))

        def mass_updater(mob):
            mob.move_to(np.array([mass_x(tracker.get_value()), y0, 0]))

        def label_updater(mob):
            mob.move_to(mass.get_center())

        spring.add_updater(spring_updater)
        mass.add_updater(mass_updater)
        mass_label.add_updater(label_updater)
        self.play(tracker.animate.set_value(4 * PI / omega_val), run_time=4.5, rate_func=linear)
        spring.clear_updaters()
        mass.clear_updaters()
        mass_label.clear_updaters()

        self.wait(1.0)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 2: Many-body problem
# ─────────────────────────────────────────────
class Scene2_ManyBody(Scene):
    def construct(self):
        title = section_title("2. Why Many-Body Systems Need Numerics")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        label_2body = styled_text("2-body: reducible and often solvable", color=ACCENT3, font_size=24, weight=BOLD)
        label_2body.to_edge(LEFT, buff=0.9).shift(UP * 2.0)
        p1 = Dot(LEFT * 4.2 + UP * 0.9, radius=0.22, color=ACCENT1)
        p2 = Dot(LEFT * 2.2 + UP * 0.9, radius=0.22, color=ACCENT2)
        bond = Line(p1.get_center(), p2.get_center(), color=GRAY)
        mu = styled_tex(r"\mu=\frac{m_1m_2}{m_1+m_2}", font_size=28, color=ACCENT3)
        mu.next_to(bond, DOWN, buff=0.45)
        self.play(Write(label_2body), FadeIn(p1), FadeIn(p2), Create(bond), Write(mu))

        divider = Line(UP * 2.5 + RIGHT * 0.2, DOWN * 2.8 + RIGHT * 0.2, color=LIGHT_GRAY, stroke_width=1.2)
        self.play(Create(divider))

        label_3body = styled_text("3-body and beyond: coupled equations", color=ACCENT2, font_size=24, weight=BOLD)
        label_3body.to_edge(RIGHT, buff=1.1).shift(UP * 2.0)
        p3 = Dot(RIGHT * 2.5 + UP * 0.8, radius=0.19, color=ACCENT1)
        p4 = Dot(RIGHT * 4.5 + UP * 0.6, radius=0.19, color=ACCENT2)
        p5 = Dot(RIGHT * 3.4 + DOWN * 0.7, radius=0.19, color=ACCENT4)
        links = VGroup(
            Line(p3.get_center(), p4.get_center(), color=GRAY),
            Line(p3.get_center(), p5.get_center(), color=GRAY),
            Line(p4.get_center(), p5.get_center(), color=GRAY),
        )
        eqs = VGroup(
            styled_tex(r"m_1\ddot{\mathbf r}_1=\mathbf F_{12}+\mathbf F_{13}", font_size=22, color=FG),
            styled_tex(r"m_2\ddot{\mathbf r}_2=\mathbf F_{21}+\mathbf F_{23}", font_size=22, color=FG),
            styled_tex(r"m_3\ddot{\mathbf r}_3=\mathbf F_{31}+\mathbf F_{32}", font_size=22, color=FG),
            styled_tex(r"m_i\ddot{\mathbf r}_i=\sum_{j\neq i}\mathbf F_{ij}", font_size=24, color=ACCENT1),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        eqs.next_to(links, DOWN, buff=0.55)
        eqs.shift(RIGHT * 0.45)

        self.play(Write(label_3body), FadeIn(p3), FadeIn(p4), FadeIn(p5), Create(links))
        self.play(Write(eqs))

        note = styled_text("No general closed form → integrate trajectories numerically", color=ACCENT2, font_size=24)
        note.to_edge(DOWN, buff=0.45)
        self.play(Write(note))
        self.wait(1.3)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 3: Numerical integration idea
# ─────────────────────────────────────────────
class Scene3_NumericalPitfalls(Scene):
    def construct(self):
        title = section_title("3. Numerical Integration: Approximate in Time")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        idea = styled_text("Replace derivatives by finite differences", color=FG, font_size=26)
        idea.shift(UP * 2.35)
        fd_eq = styled_tex(r"\frac{dx}{dt}\approx\frac{x(t+\Delta t)-x(t)}{\Delta t}", font_size=34, color=ACCENT1)
        fd_eq.shift(UP * 1.55)
        self.play(Write(idea), Write(fd_eq))

        ax = Axes(
            x_range=[0, 10, 2],
            y_range=[-1.4, 1.4, 0.7],
            x_length=8.2,
            y_length=3.4,
            axis_config={"color": FG, "stroke_width": 1.5},
            tips=False,
        ).shift(DOWN * 0.55)
        labels = VGroup(
            styled_tex("t", font_size=22, color=FG).next_to(ax.x_axis, RIGHT),
            styled_tex("x(t)", font_size=22, color=FG).next_to(ax.y_axis, UP),
        )
        exact = ax.plot(lambda t: np.cos(t), color=ACCENT3, stroke_width=2.5)
        exact_lbl = small_note("exact trajectory", color=ACCENT3).next_to(ax, RIGHT, buff=0.12).shift(UP * 0.35)
        self.play(Create(ax), Write(labels), Create(exact), Write(exact_lbl))

        dt_large = 1.5
        t_pts_large = np.arange(0, 10.01, dt_large)
        dots_large = VGroup(*[Dot(ax.c2p(t, np.cos(t)), radius=0.065, color=ACCENT2) for t in t_pts_large])
        lines_large = VGroup(*[
            Line(ax.c2p(t_pts_large[i], np.cos(t_pts_large[i])), ax.c2p(t_pts_large[i + 1], np.cos(t_pts_large[i + 1])), color=ACCENT2, stroke_width=1.7)
            for i in range(len(t_pts_large) - 1)
        ])
        dt_label = styled_tex(r"\Delta t=1.5", font_size=24, color=ACCENT2).next_to(ax, LEFT, buff=0.25).shift(UP * 0.9)
        self.play(FadeIn(dots_large), Create(lines_large), Write(dt_label))
        self.wait(0.6)

        dt_small = 0.6
        t_pts_small = np.arange(0, 10.01, dt_small)
        dots_small = VGroup(*[Dot(ax.c2p(t, np.cos(t)), radius=0.05, color=ACCENT1) for t in t_pts_small])
        lines_small = VGroup(*[
            Line(ax.c2p(t_pts_small[i], np.cos(t_pts_small[i])), ax.c2p(t_pts_small[i + 1], np.cos(t_pts_small[i + 1])), color=ACCENT1, stroke_width=1.3)
            for i in range(len(t_pts_small) - 1)
        ])
        dt_small_label = styled_tex(r"\Delta t=0.6", font_size=24, color=ACCENT1).move_to(dt_label)
        self.play(Transform(dots_large, dots_small), Transform(lines_large, lines_small), Transform(dt_label, dt_small_label))

        tradeoff = styled_text("Smaller Δt improves accuracy, but costs more force evaluations.", color=ACCENT4, font_size=23)
        tradeoff.to_edge(DOWN, buff=0.42)
        self.play(Write(tradeoff))
        self.wait(1.3)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 4: Explicit Euler
# ─────────────────────────────────────────────
class Scene4_ExplicitEuler(Scene):
    def construct(self):
        title = section_title("4. Explicit Euler")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        algo = VGroup(
            styled_text("Algorithm", color=ACCENT1, font_size=24, weight=BOLD),
            styled_tex(r"a(t)=F(x(t))/m", font_size=28, color=FG),
            styled_tex(r"x(t+\Delta t)=x(t)+v(t)\Delta t", font_size=28, color=FG),
            styled_tex(r"v(t+\Delta t)=v(t)+a(t)\Delta t", font_size=28, color=FG),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        algo_box = SurroundingRectangle(algo, color=ACCENT1, buff=0.2)
        algo_panel = VGroup(algo_box, algo)
        algo_panel.to_edge(LEFT, buff=0.8).shift(UP * 0.65)
        self.play(Create(algo_box), Write(algo))

        ax = phase_axes(4.0, 4.0)
        ax.to_edge(RIGHT, buff=0.8).shift(UP * 0.55)
        labels = add_phase_labels(ax, "x", "v", 20)
        exact_pts = compute_exact_phase(n=240)
        euler_pts = compute_euler_phase(dt=0.22, steps=120)
        exact_curve = phase_path(ax, exact_pts, ACCENT3, 2.2)
        euler_curve = phase_path(ax, euler_pts, ACCENT2, 2.5)
        euler_dot = Dot(ax.c2p(*euler_pts[0]), radius=0.06, color=ACCENT2)
        exact_lbl = small_note("exact: closed ellipse", color=ACCENT3).next_to(ax, DOWN, buff=0.08).shift(LEFT * 1.2)
        euler_lbl = small_note("Euler: spirals outward", color=ACCENT2).next_to(ax, DOWN, buff=0.08).shift(RIGHT * 1.0)

        self.play(Create(ax), Write(labels), Create(exact_curve))
        self.play(Create(euler_curve), Write(exact_lbl), Write(euler_lbl), FadeIn(euler_dot))
        tracker = ValueTracker(0)
        euler_dot.add_updater(lambda m: m.move_to(ax.c2p(*euler_pts[min(int(tracker.get_value()), len(euler_pts) - 1)])))
        self.play(tracker.animate.set_value(len(euler_pts) - 1), run_time=3.0, rate_func=linear)
        euler_dot.clear_updaters()

        t_e, e_e = compute_energy_curve("euler", dt=0.22, steps=80)
        e_ax = energy_axes(9.0, 1.6, [0.35, 3.1, 0.55])
        e_ax.to_edge(DOWN, buff=0.55)
        e_labels = VGroup(
            styled_tex("t", font_size=20, color=FG).next_to(e_ax.x_axis, RIGHT, buff=0.08),
            styled_tex("E(t)", font_size=20, color=FG).next_to(e_ax.y_axis, UP, buff=0.08),
        )
        e_curve = VMobject(color=ACCENT2, stroke_width=2.5).set_points_smoothly([e_ax.c2p(t, y) for t, y in zip(t_e, e_e)])
        e_note = styled_text("Energy drifts upward → not symplectic", color=ACCENT2, font_size=23)
        e_note.next_to(e_ax, UP, buff=0.18)
        self.play(Create(e_ax), Write(e_labels), Create(e_curve), Write(e_note))
        self.wait(1.3)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 5: Verlet
# ─────────────────────────────────────────────
class Scene5_Verlet(Scene):
    def construct(self):
        title = section_title("5. Verlet Integrator")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        deriv_title = styled_text("Derive by adding Taylor expansions", color=FG, font_size=24)
        deriv_title.shift(UP * 2.55)
        fwd = styled_tex(
            r"x(t+\Delta t)=x(t)+v(t)\Delta t+\frac{1}{2}a(t)\Delta t^2+\cdots",
            font_size=24,
            color=FG,
        ).shift(UP * 1.7)
        bwd = styled_tex(
            r"x(t-\Delta t)=x(t)-v(t)\Delta t+\frac{1}{2}a(t)\Delta t^2+\cdots",
            font_size=24,
            color=FG,
        ).shift(UP * 1.0)
        add_arrow = Arrow(UP * 1.45 + LEFT * 0.5, UP * 0.55 + LEFT * 0.5, color=ACCENT4, buff=0)
        add_label = styled_text("add", color=ACCENT4, font_size=22).next_to(add_arrow, LEFT, buff=0.1)
        result = styled_tex(
            r"x(t+\Delta t)=2x(t)-x(t-\Delta t)+a(t)\Delta t^2+\mathcal{O}(\Delta t^4)",
            font_size=28,
            color=ACCENT1,
        ).shift(DOWN * 0.15)
        result_box = SurroundingRectangle(result, color=ACCENT1, buff=0.18)
        beat1 = VGroup(deriv_title, fwd, bwd, add_arrow, add_label, result, result_box)
        self.play(Write(deriv_title), Write(fwd), Write(bwd))
        self.play(Create(add_arrow), Write(add_label))
        self.play(Write(result), Create(result_box))
        self.wait(1.0)
        self.play(FadeOut(beat1))

        props = VGroup(
            styled_text("✓ time-reversible", color=ACCENT3, font_size=24),
            styled_text("✓ symplectic / bounded energy error", color=ACCENT3, font_size=24),
            styled_text("✓ good position accuracy", color=ACCENT3, font_size=24),
            styled_text("✗ velocity not directly available", color=ACCENT2, font_size=24),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        props.to_edge(LEFT, buff=0.85).shift(UP * 0.55)

        algo = VGroup(
            styled_text("Algorithm", color=ACCENT1, font_size=24, weight=BOLD),
            styled_tex(r"a(t)=F(x(t))/m", font_size=26, color=FG),
            styled_tex(r"x(t+\Delta t)=2x(t)-x(t-\Delta t)+a(t)\Delta t^2", font_size=24, color=FG),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        algo_box = SurroundingRectangle(algo, color=ACCENT1, buff=0.18)
        algo_panel = VGroup(algo_box, algo)
        algo_panel.to_edge(RIGHT, buff=0.75).shift(UP * 1.2)

        ax = phase_axes(3.5, 3.5)
        ax.to_edge(RIGHT, buff=0.85).shift(DOWN * 1.0)
        labels = add_phase_labels(ax, "x", "v", 18)
        verlet_pts = compute_verlet_phase(dt=0.22, steps=120)
        curve = phase_path(ax, verlet_pts, ACCENT3, 2.5)
        note = small_note("bounded orbit in phase space", color=ACCENT3).next_to(ax, DOWN, buff=0.08)

        self.play(Write(props))
        self.play(Create(algo_box), Write(algo))
        self.play(Create(ax), Write(labels), Create(curve), Write(note))
        self.wait(1.3)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 6: Velocity Verlet
# ─────────────────────────────────────────────
class Scene6_VelocityVerlet(Scene):
    def construct(self):
        title = section_title("6. Velocity Verlet")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        def step_box(step_title, eq, color):
            label = styled_text(step_title, color=color, font_size=22, weight=BOLD)
            formula = styled_tex(eq, font_size=22, color=FG)
            content = VGroup(label, formula).arrange(DOWN, buff=0.12)
            box = RoundedRectangle(corner_radius=0.12, width=5.8, height=1.2, color=color, stroke_width=2)
            content.move_to(box.get_center())
            return VGroup(box, content)

        s1 = step_box("Step 1: Half-kick", r"v(t+\tfrac{\Delta t}{2})=v(t)+\tfrac{1}{2}a(t)\Delta t", ACCENT4)
        s2 = step_box("Step 2: Full drift", r"x(t+\Delta t)=x(t)+v(t+\tfrac{\Delta t}{2})\Delta t", ACCENT1)
        s3 = step_box("Step 3: Recompute force", r"a(t+\Delta t)=F(x(t+\Delta t))/m", ACCENT2)
        s4 = step_box("Step 4: Half-kick again", r"v(t+\Delta t)=v(t+\tfrac{\Delta t}{2})+\tfrac{1}{2}a(t+\Delta t)\Delta t", ACCENT3)
        steps = VGroup(s1, s2, s3, s4).arrange(DOWN, buff=0.28).scale(0.88)
        steps.shift(LEFT * 1.95 + DOWN * 0.12)

        arrows = VGroup(*[
            Arrow(steps[i].get_bottom() + DOWN * 0.03, steps[i + 1].get_top() + UP * 0.03, color=GRAY, buff=0.02, stroke_width=2)
            for i in range(3)
        ])

        for i, box in enumerate(steps):
            self.play(FadeIn(box, shift=UP * 0.15))
            if i < len(arrows):
                self.play(Create(arrows[i]))
        
        ax = phase_axes(3.3, 3.3)
        ax.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.15)
        labels = add_phase_labels(ax, "x", "v", 18)
        vv_pts = compute_velocity_verlet_phase(dt=0.22, steps=120)
        curve = phase_path(ax, vv_pts, ACCENT3, 2.6)
        exact = phase_path(ax, compute_exact_phase(n=240), LIGHT_GRAY, 1.8)
        note = VGroup(
            styled_text("Why this is good", color=ACCENT3, font_size=22, weight=BOLD),
            small_note("time-reversible • symplectic • explicit v(t+Δt)", color=ACCENT3),
        ).arrange(DOWN, buff=0.12)
        note.next_to(ax, DOWN, buff=0.12)
        self.play(Create(ax), Write(labels), Create(exact), Create(curve), Write(note))
        self.wait(1.4)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 7: Leapfrog
# ─────────────────────────────────────────────
class Scene7_Leapfrog(Scene):
    def construct(self):
        title = section_title("7. Leapfrog Integrator")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        grid_title = styled_text("Staggered time grid", color=FG, font_size=26)
        grid_title.shift(UP * 2.55)
        self.play(Write(grid_title))

        x_line = NumberLine(x_range=[0, 5, 1], length=7.2, color=ACCENT1, include_numbers=False, tick_size=0.09)
        x_line.shift(UP * 1.35)
        x_ticks = VGroup(*[
            VGroup(
                Dot(x_line.n2p(i), radius=0.07, color=ACCENT1),
                styled_tex(rf"t_{i}", font_size=18, color=ACCENT1).next_to(x_line.n2p(i), UP, buff=0.08),
            )
            for i in range(6)
        ])
        x_label = styled_tex(r"x:", font_size=24, color=ACCENT1).next_to(x_line, LEFT, buff=0.3)

        v_line = NumberLine(x_range=[0, 5, 1], length=7.2, color=ACCENT2, include_numbers=False, tick_size=0.09)
        v_line.shift(UP * 0.35)
        v_ticks = VGroup(*[
            VGroup(
                Dot(v_line.n2p(i + 0.5), radius=0.07, color=ACCENT2),
                styled_tex(rf"t_{i}+\tfrac{{1}}{{2}}", font_size=16, color=ACCENT2).next_to(v_line.n2p(i + 0.5), UP, buff=0.08),
            )
            for i in range(5)
        ])
        v_label = styled_tex(r"v:", font_size=24, color=ACCENT2).next_to(v_line, LEFT, buff=0.3)

        arrows = VGroup(*[
            CurvedArrow(x_line.n2p(i), v_line.n2p(i + 0.5), color=ACCENT4, angle=-PI / 4)
            for i in range(5)
        ] + [
            CurvedArrow(v_line.n2p(i + 0.5), x_line.n2p(i + 1), color=ACCENT3, angle=-PI / 4)
            for i in range(5)
        ])

        self.play(Create(x_line), FadeIn(x_ticks), Write(x_label))
        self.play(Create(v_line), FadeIn(v_ticks), Write(v_label))
        self.play(Create(arrows))

        algo = VGroup(
            styled_text("Algorithm", color=ACCENT1, font_size=24, weight=BOLD),
            styled_tex(r"v(t+\tfrac{\Delta t}{2})=v(t-\tfrac{\Delta t}{2})+a(t)\Delta t", font_size=24, color=FG),
            styled_tex(r"x(t+\Delta t)=x(t)+v(t+\tfrac{\Delta t}{2})\Delta t", font_size=24, color=FG),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        algo_box = SurroundingRectangle(algo, color=ACCENT1, buff=0.18)
        algo_panel = VGroup(algo_box, algo)
        algo_panel.shift(DOWN * 1.9)
        note = styled_text("Equivalent in spirit to Velocity Verlet, but velocities live on half-steps.", color=ACCENT4, font_size=22)
        note.to_edge(DOWN, buff=0.35)
        self.play(Create(algo_box), Write(algo), Write(note))
        self.wait(1.3)
        fade_all(self)


# ─────────────────────────────────────────────
# Scene 8: Comparison payoff
# ─────────────────────────────────────────────
class Scene8_Comparison(Scene):
    def construct(self):
        title = section_title("8. Which Integrator Behaves Best?")
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        methods = [
            ("Euler", ACCENT2, compute_euler_phase(dt=0.22, steps=120)),
            ("Velocity Verlet", ACCENT3, compute_velocity_verlet_phase(dt=0.22, steps=120)),
            ("Leapfrog", ACCENT1, compute_leapfrog_phase(dt=0.22, steps=120)),
        ]
        panel_group = VGroup()
        for name, color, pts in methods:
            ax = phase_axes(2.25, 2.25)
            labels = add_phase_labels(ax, "x", "v", 14)
            curve = phase_path(ax, pts, color, 2.2)
            caption = styled_text(name, color=color, font_size=18, weight=BOLD)
            caption.next_to(ax, DOWN, buff=0.12)
            panel_group.add(VGroup(ax, labels, curve, caption))
        panel_group.arrange(RIGHT, buff=0.55)
        panel_group.shift(UP * 1.45)

        for panel in panel_group:
            self.play(Create(panel[0]), Write(panel[1]), Create(panel[2]), Write(panel[3]), run_time=0.6)

        e_ax = energy_axes(8.6, 1.85, [0.35, 3.1, 0.55])
        e_ax.shift(DOWN * 0.6)
        e_labels = VGroup(
            styled_tex("t", font_size=18, color=FG).next_to(e_ax.x_axis, RIGHT, buff=0.08),
            styled_tex("E(t)", font_size=18, color=FG).next_to(e_ax.y_axis, UP, buff=0.08),
        )
        te, ee = compute_energy_curve("euler", dt=0.22, steps=80)
        tv, ev = compute_energy_curve("velocity_verlet", dt=0.22, steps=80)
        tl, el = compute_energy_curve("leapfrog", dt=0.22, steps=80)
        curve_e = VMobject(color=ACCENT2, stroke_width=2.2).set_points_smoothly([e_ax.c2p(t, y) for t, y in zip(te, ee)])
        curve_v = VMobject(color=ACCENT3, stroke_width=2.2).set_points_smoothly([e_ax.c2p(t, y) for t, y in zip(tv, ev)])
        curve_l = VMobject(color=ACCENT1, stroke_width=2.2).set_points_smoothly([e_ax.c2p(t, y) for t, y in zip(tl, el)])
        legend = VGroup(
            styled_text("Euler", color=ACCENT2, font_size=18),
            styled_text("Vel. Verlet", color=ACCENT3, font_size=18),
            styled_text("Leapfrog", color=ACCENT1, font_size=18),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        legend.next_to(e_ax, RIGHT, buff=0.2)
        self.play(Create(e_ax), Write(e_labels))
        self.play(Create(curve_e), Create(curve_v), Create(curve_l), Write(legend))

        table_data = [
            ["Euler", "1", "✗", "explicit", "energy drifts"],
            ["Verlet", "2-pos", "✓", "implicit", "good positions"],
            ["Vel. Verlet", "2", "✓", "explicit", "MD standard"],
            ["Leapfrog", "2", "✓", "staggered", "VV-like"],
        ]
        table = Table(
            table_data,
            col_labels=[Text(h, color=FG, font_size=16) for h in ["Method", "Order", "Sympl.", "Velocity", "Takeaway"]],
            include_outer_lines=True,
            line_config={"color": GRAY, "stroke_width": 1},
            element_to_mobject_config={"color": FG, "font_size": 16},
        ).scale(0.58)
        table.to_edge(DOWN, buff=0.15)
        self.play(Create(table))

        closing = styled_text("Velocity Verlet is the workhorse of molecular dynamics.", color=ACCENT3, font_size=24, weight=BOLD)
        closing.next_to(table, UP, buff=0.16)
        self.play(Write(closing))
        self.wait(2.0)
        fade_all(self)


# ─────────────────────────────────────────────
# Master scene
# ─────────────────────────────────────────────
class MDIntegrators(Scene):
    def construct(self):
        main_title = VGroup(
            styled_text("Molecular Dynamics", color=ACCENT1, font_size=50, weight=BOLD),
            styled_text("From Mechanics to Structure-Preserving Integrators", color=FG, font_size=28),
        ).arrange(DOWN, buff=0.35)
        self.play(Write(main_title))
        self.wait(1.8)
        self.play(FadeOut(main_title))
        self.wait(0.2)

        for SceneClass in [
            Scene0_MechanicsFoundations,
            Scene1_NewtonSpring,
            Scene2_ManyBody,
            Scene3_NumericalPitfalls,
            Scene4_ExplicitEuler,
            Scene5_Verlet,
            Scene6_VelocityVerlet,
            Scene7_Leapfrog,
            Scene8_Comparison,
        ]:
            scene_instance = SceneClass()
            scene_instance.camera = self.camera
            scene_instance.renderer = self.renderer
            scene_instance.mobjects = self.mobjects
            scene_instance.construct()
            if self.mobjects:
                self.play(*[FadeOut(mob) for mob in list(self.mobjects)])
                self.wait(0.15)

        end_card = VGroup(
            styled_text("Summary", color=ACCENT1, font_size=42, weight=BOLD),
            styled_text("Phase-space structure and energy behavior tell you which integrator to trust.", color=FG, font_size=24),
            styled_text("In MD practice: Velocity Verlet is usually the default choice.", color=ACCENT3, font_size=24),
        ).arrange(DOWN, buff=0.35)
        self.play(Write(end_card))
        self.wait(2.5)
        self.play(FadeOut(end_card))
