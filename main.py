import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm

def get_fancy_arabic_font():
    preferred_fonts = ['Traditional Arabic', 'Amiri', 'Sakkal Majalla', 'Microsoft Sans Serif', 'Arial', 'Tahoma']
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in preferred_fonts:
        if font in system_fonts:
            return font
    return 'sans-serif'

def bezier_curve(p0, p1, p2, p3, num_points=25):
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    points = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return points[:, 0], points[:, 1]

def create_palm():
    paths = []
    scale = 0.82
    
    # 1. Trunk
    left_trunk_x, left_trunk_y = bezier_curve(
        np.array([-0.6 * scale, -2.5 * scale]), np.array([-0.45 * scale, -0.8 * scale]),
        np.array([-0.25 * scale, 0.8 * scale]), np.array([-0.15 * scale, 2.0 * scale]), 20
    )
    right_trunk_x, right_trunk_y = bezier_curve(
        np.array([0.6 * scale, -2.5 * scale]), np.array([0.45 * scale, -0.8 * scale]),
        np.array([0.25 * scale, 0.8 * scale]), np.array([0.15 * scale, 2.0 * scale]), 20
    )
    paths.append((left_trunk_x, left_trunk_y))
    paths.append((right_trunk_x, right_trunk_y))
    
    # Trunk rings
    for y_val in np.linspace(-2.1 * scale, 1.6 * scale, 7):
        w = (1.8 * scale - y_val) / (3.9 * scale) * 0.35 * scale + 0.15 * scale
        rx, ry = bezier_curve(
            np.array([-w, y_val - 0.08 * scale]), np.array([-w/2, y_val + 0.08 * scale]),
            np.array([w/2, y_val + 0.08 * scale]), np.array([w, y_val - 0.08 * scale]), 10
        )
        paths.append((rx, ry))

    # 2. Crown Base
    cx, cy = bezier_curve(
        np.array([-0.3 * scale, 1.9 * scale]), np.array([0.0, 1.75 * scale]),
        np.array([0.0, 1.75 * scale]), np.array([0.3 * scale, 1.9 * scale]), 10
    )
    paths.append((cx, cy))

    # 3. Fronds & Leaflets
    frond_configs = [
        (np.array([0, 2.0 * scale]), np.array([0.2 * scale, 4.5 * scale]), np.array([0.1 * scale, 5.8 * scale]), np.array([0.0, 6.4 * scale])),
        (np.array([0, 2.0 * scale]), np.array([1.0 * scale, 4.2 * scale]), np.array([2.0 * scale, 5.2 * scale]), np.array([2.5 * scale, 5.6 * scale])),
        (np.array([0, 2.0 * scale]), np.array([-1.0 * scale, 4.2 * scale]), np.array([-2.0 * scale, 5.2 * scale]), np.array([-2.5 * scale, 5.6 * scale])),
        (np.array([0, 2.0 * scale]), np.array([1.8 * scale, 3.8 * scale]), np.array([3.2 * scale, 4.2 * scale]), np.array([4.2 * scale, 4.0 * scale])),
        (np.array([0, 2.0 * scale]), np.array([-1.8 * scale, 3.8 * scale]), np.array([-3.2 * scale, 4.2 * scale]), np.array([-4.2 * scale, 4.0 * scale])),
        (np.array([0, 2.0 * scale]), np.array([2.0 * scale, 3.2 * scale]), np.array([3.8 * scale, 3.0 * scale]), np.array([4.6 * scale, 2.2 * scale])),
        (np.array([0, 2.0 * scale]), np.array([-2.0 * scale, 3.2 * scale]), np.array([-3.8 * scale, 3.0 * scale]), np.array([-4.6 * scale, 2.2 * scale])),
        (np.array([0, 2.0 * scale]), np.array([1.8 * scale, 2.4 * scale]), np.array([3.4 * scale, 1.8 * scale]), np.array([4.2 * scale, 0.8 * scale])),
        (np.array([0, 2.0 * scale]), np.array([-1.8 * scale, 2.4 * scale]), np.array([-3.4 * scale, 1.8 * scale]), np.array([-4.2 * scale, 0.8 * scale])),
    ]

    for p0, p1, p2, p3 in frond_configs:
        fx, fy = bezier_curve(p0, p1, p2, p3, 20)
        paths.append((fx, fy))
        
        num_leaves = 14
        for i in range(2, num_leaves):
            t = i / float(num_leaves)
            idx = int(t * (len(fx) - 1))
            bx, by = fx[idx], fy[idx]
            
            dx = fx[min(idx + 1, len(fx) - 1)] - fx[max(idx - 1, 0)]
            dy = fy[min(idx + 1, len(fy) - 1)] - fy[max(idx - 1, 0)]
            
            leaf_len = (np.sin(t * np.pi) * 0.75 + 0.25) * 0.9 * scale
            
            lx = bx + (dx * 0.1 - dy * 0.7) * leaf_len
            ly = by + (dy * 0.1 + dx * 0.7) * leaf_len - 0.15 * scale * t
            paths.append((np.array([bx, lx]), np.array([by, ly])))
            
            rx_l = bx + (dx * 0.1 + dy * 0.7) * leaf_len
            ry_l = by + (dy * 0.1 - dx * 0.7) * leaf_len - 0.15 * scale * t
            paths.append((np.array([bx, rx_l]), np.array([by, ry_l])))
            
    return paths

def create_single_sword():
    paths = []
    
    blade_x, blade_y = bezier_curve(
        np.array([0.0, -4.5]), np.array([1.5, -3.8]), np.array([3.2, -2.8]), np.array([4.2, -1.2]), 15
    )
    blade_back_x, blade_back_y = bezier_curve(
        np.array([0.2, -4.5]), np.array([1.6, -3.7]), np.array([3.0, -2.7]), np.array([4.2, -1.2]), 15
    )
    paths.append((blade_x, blade_y))
    paths.append((blade_back_x, blade_back_y))
    
    guard_x, guard_y = bezier_curve(
        np.array([-0.6, -4.3]), np.array([0.1, -4.5]), np.array([0.1, -4.5]), np.array([0.8, -4.7]), 6
    )
    paths.append((guard_x, guard_y))
    
    hilt_x, hilt_y = bezier_curve(
        np.array([0.1, -4.5]), np.array([-0.4, -4.9]), np.array([-0.7, -5.2]), np.array([-0.9, -5.5]), 8
    )
    paths.append((hilt_x, hilt_y))
    
    px, py = bezier_curve(
        np.array([-1.1, -5.5]), np.array([-0.8, -5.7]), np.array([-0.7, -5.4]), np.array([-1.1, -5.5]), 6
    )
    paths.append((px, py))
    
    return paths

def create_swords():
    single_sword = create_single_sword()
    all_swords_paths = []
    
    angle1 = np.radians(-25)
    cos1, sin1 = np.cos(angle1), np.sin(angle1)
    for x, y in single_sword:
        cy = y + 3.5
        rx = x * cos1 - cy * sin1 - 0.2
        ry = x * sin1 + cy * cos1 - 3.5
        all_swords_paths.append((rx, ry))
        
    angle2 = np.radians(25)
    cos2, sin2 = np.cos(angle2), np.sin(angle2)
    for x, y in single_sword:
        mx = -x
        cy = y + 3.5
        rx = mx * cos2 - cy * sin2 + 0.2
        ry = mx * sin2 + cy * cos2 - 3.5
        all_swords_paths.append((rx, ry))
        
    return all_swords_paths

def main():
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.axis('off')
    
    arabic_font = get_fancy_arabic_font()
    
    all_paths = create_palm() + create_swords()
    
    glow_params = [
        {'lw': 14.0, 'base_alpha': 0.05},
        {'lw': 9.0,  'base_alpha': 0.12},
        {'lw': 5.5,  'base_alpha': 0.30},
        {'lw': 2.5,  'base_alpha': 0.70},
        {'lw': 1.2,  'base_alpha': 1.00, 'color': '#FFFFFF'}
    ]
    
    path_lines = []
    for x, y in all_paths:
        layers = []
        for gp in glow_params:
            c = gp.get('color', '#00FF66')
            line, = ax.plot([], [], color=c, lw=gp['lw'], alpha=0.0, solid_capstyle='round')
            layers.append((line, gp['base_alpha']))
        path_lines.append((x, y, layers))
        
    extended_text = "المملكة العربيــــــــة السعوديــــــــة"
    
    text_title = ax.text(0, -6.8, extended_text, color='#00FF66', fontsize=24,
                         ha='center', va='center', fontweight='bold', family=arabic_font, alpha=0.0)

    total_paths = len(all_paths)
    draw_frames = 12
    fade_frames = 5
    pulse_frames = 40
    total_frames = draw_frames + fade_frames + pulse_frames

    all_artist_objects = []
    for _, _, layers in path_lines:
        for l, _ in layers:
            all_artist_objects.append(l)
    all_artist_objects.append(text_title)

    def animate(frame):
        if frame <= draw_frames:
            visible_count = int((frame / draw_frames) * total_paths)
            for idx in range(total_paths):
                x, y, layers = path_lines[idx]
                if idx <= visible_count:
                    for line, base_alpha in layers:
                        line.set_data(x, y)
                        line.set_alpha(base_alpha)
                else:
                    for line, _ in layers:
                        line.set_data([], [])

        elif frame <= (draw_frames + fade_frames):
            progress = (frame - draw_frames) / float(fade_frames)
            text_title.set_alpha(progress)

        else:
            pulse_progress = (frame - draw_frames - fade_frames) / float(pulse_frames)
            pulse = 0.65 + 0.35 * np.sin(pulse_progress * 2 * np.pi)
            
            for _, _, layers in path_lines:
                for line, base_alpha in layers:
                    line.set_alpha(base_alpha * pulse)
                    
            text_title.set_alpha(1.0)

        return all_artist_objects

    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=20, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
