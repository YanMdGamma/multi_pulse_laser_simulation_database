import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D  #

if __name__ == "__main__":
    # =============================
    #
    # =============================
    filename = "./RMlloweng_other/RMlloweng_other_NV.txt"   #
    eps = 1.7                  #
    min_samples = 2            #
    output_file = "cluster_result_oneotherC.txt"  #
    count_file_type = "cluster_count_RMlloweng.txt"

    # =============================
    # =============================
    data = []
    atom_ids = []
    atom_types = []

    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                atom_id = parts[0]
                atom_type = int(parts[1])       #
                x, y, z = map(float, parts[2:5])
            except ValueError:
                continue
            data.append([x, y, z])
            atom_ids.append(atom_id)
            atom_types.append(atom_type)

    data = np.array(data)
    atom_types = np.array(atom_types)

    # =============================
    # 1. DBSCAN clustering
    # =============================
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

    print(f"detected cluster number: {n_clusters}")

    # =============================
    # 3. sort results
    # =============================
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append((atom_ids[idx], atom_types[idx], *data[idx]))

    # =============================
    # 4. write down results
    # =============================
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DBSCAN results (eps={eps}, min_samples={min_samples})\n")
        f.write(f"detected cluster number: {n_clusters}\n\n")

        for label in sorted(clusters.keys()):
            if label == -1:
                f.write("=== noise ===\n")
            else:
                f.write("cluster label id type x y z\n")

            coords = np.array([c[2:] for c in clusters[label]])
            for atom_id, atom_type, x, y, z in clusters[label]:
                f.write(f"{label} {atom_id} {atom_type} {x:.3f} {y:.3f} {z:.3f}\n")

            if label != -1:
                mean_x, mean_y, mean_z = coords.mean(axis=0)
                f.write(f"average location: {mean_x:.3f} {mean_y:.3f} {mean_z:.3f}\n")

            f.write("\n")

    print(f"✅ results is saved: {output_file}")

    # =============================
    # 5. cluster files
    # =============================

    with open(count_file_type, "w", encoding="utf-8") as f:
        f.write("cluster number\ttype 1 number\ttype 2 number\n")
        f.write("============================\n")
        for label in sorted(clusters.keys()):
            type1_count = sum(1 for _, t, *_ in clusters[label] if t == 1)
            type2_count = sum(1 for _, t, *_ in clusters[label] if t == 2)
            f.write(f"{label}\t{type1_count}\t{type2_count}\n")

    print(f"📊 number is sorted by labels: {count_file_type}")

    # =============================
    # 6. 3D 可视化
    # =============================
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cluster_labels = [label for label in unique_labels if label != -1]
    label_min = 0
    label_max = max(cluster_labels) if cluster_labels else 0
    norm = mcolors.Normalize(vmin=label_min, vmax=label_max)
    cmap = cm.get_cmap("rainbow")

    #
    phi, theta = np.mgrid[0:np.pi:10j, 0:2 * np.pi:10j]

    for label in unique_labels:
        xyz = data[labels == label]
        types = atom_types[labels == label]

        if label == -1:
            color = 'gray'
        else:
            color = cmap(norm(label))

        for (x0, y0, z0), t in zip(xyz, types):
            #
            if t == 1:
                r = 1.0
                xs = r * np.sin(phi) * np.cos(theta) + x0
                ys = r * np.sin(phi) * np.sin(theta) + y0
                zs = r * np.cos(phi) + z0
                ax.plot_surface(xs, ys, zs, color=color, linewidth=0, antialiased=True, alpha=0.6)
            elif t == 2:
                r = 2.2
                xs = r * np.sin(phi) * np.cos(theta) + x0
                ys = r * np.sin(phi) * np.sin(theta) + y0
                zs = r * np.cos(phi) + z0
                ax.plot_surface(xs, ys, zs, color='grey', linewidth=0, antialiased=True)

    # layout
    ax.set_xlabel(r"X ($\mathrm{\AA}$)", fontsize=20, fontname="Arial")
    ax.set_ylabel(r"Y ($\mathrm{\AA}$)", fontsize=20, fontname="Arial")
    ax.set_zlabel(r"Z ($\mathrm{\AA}$)", fontsize=20, fontname="Arial")
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)

    z_min, z_max = ax.get_zlim()  #
    ax.set_zticks(np.arange(np.floor(0 / 20) * 20, np.ceil(z_max / 20) * 20 + 1, 20))

    ax.xaxis.labelpad = 10  #
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10
    ax.set_zlim(0, 85)

    ax.xaxis.line.set_linewidth(2)
    ax.yaxis.line.set_linewidth(2)
    ax.zaxis.line.set_linewidth(2)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.07)
    cbar.set_label("Cluster label", rotation=270, labelpad=25, fontname="Arial", fontsize=20)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontsize(18)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    tick_step = max(1, (label_max - label_min) // 10)
    ticks = np.arange(label_min, label_max + 1, tick_step)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(t)) for t in ticks])

    ax.view_init(elev=90, azim=-90)

    plt.tight_layout()

    output_image = "DBSCAN_RMllowengt.png"
    plt.savefig(output_image, dpi=600, bbox_inches='tight')
    print(f"✅ figure is saved: {output_image}（600 dpi）")

    plt.show()

