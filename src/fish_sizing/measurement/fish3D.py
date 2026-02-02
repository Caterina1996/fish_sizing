import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy import stats

import os
from termcolor import colored
from pyntcloud import PyntCloud

import cloudpickle as pickle
from sklearn.cluster import DBSCAN
import hdbscan
import collections

from fish_sizing.utils import tools
from fish_sizing.detection.fish2D import Fish2D, FrameScene

class Fish3D(Fish2D):
    def __init__(self, fish2d, pointcloud_raw, base_path):
        # Llama al constructor de la clase base
        super().__init__(fish2d.fish_frame, fish2d.color_id, fish2d.class_name,fish2d.model_classes_dict,
                         fish2d.mask,fish2d.bbox, fish2d.class_colours_dict,fish2d.does_overlap, 
                         fish2d.overlapping_ids, fish2d.model_used,fish2d.track_id, fish2d.in_image_borders)
        
        self.__dict__.update(fish2d.__dict__)
        
        # Atributos nuevos o extendidos de Fish3D        
        
        self.fish_direction = None
        self.azimuth_deg = -1
        self.elevation_deg = -1

        self.pointcloud_raw = pointcloud_raw  # numpy array (N x 4)
        self.pointcloud_filtered = None
        self.pointcloud_size_ok = False

        self.length = -1
        self.filtered_length = -1
        self.base_path = base_path
        self.pointcloud_dir = os.path.join(base_path, f"frame_{self.fish_frame}")
        self.fish_dist_from_camera = -1
        
        self.pc_error_log = []
            
    def get_distance_camera_fish(self):
            #TEncontrar la distancia del pez a la camara

            z_vals = self.pointcloud_filtered[:, 2]
            self.fish_dist_from_camera = np.mean(z_vals)
        
    
    def measure_fish_length_ply_with_angles_and_plot(self, plot_fish_direction=True,filtered=True):
        
        if filtered:
            file_path = os.path.join(self.pointcloud_dir,"object_"+str(self.color_id)+"_filtered.ply")
        else:
            
            file_path = os.path.join(self.pointcloud_dir,"object_"+str(self.color_id)+".ply")
        
        cloud = PyntCloud.from_file(file_path)
        points = cloud.points[["x", "y", "z"]].values
        pca = PCA(n_components=3)

        if not self.pointcloud_size_ok:
            return -1, None, None, None

        pca.fit(points)
        principal_components = pca.components_
        fish_direction = principal_components[np.argmax(pca.explained_variance_)]
        
        # Project to get length
        projections = np.dot(points, fish_direction)
        fish_length = np.max(projections) - np.min(projections)

        # --- ANGLE CALCULATION ---
        fish_direction = fish_direction / np.linalg.norm(fish_direction)
        azimuth = np.arctan2(fish_direction[1], fish_direction[0])
        elevation = np.arctan2(fish_direction[2], np.linalg.norm(fish_direction[:2]))
        azimuth_deg = np.degrees(azimuth)
        elevation_deg = np.degrees(elevation)


        print(f"🐟 Fish ID {self.color_id}:")
        if filtered:
            # --- VISUALIZE ---
            if plot_fish_direction and fish_length>0:
                # tools.plot_fish_with_dual_cameras(points, fish_direction, fish_length, color_id,self.base_path)
                tools.plot_fish_with_dual_cameras_plotly(points, fish_direction, fish_length, self.color_id, self.base_path)
        
            self.filtered_fish_direction = fish_direction
            self.filtered_length = fish_length
            self.azimuth_deg = azimuth_deg
            self.elevation_deg = elevation_deg
            print(f"    ➤ filtered length = {self.filtered_length * 100:.2f} cm")
        else:
            self.fish_direction = fish_direction
            self.length = fish_length
            print(f"    ➤ raw length = {self.length * 100:.2f} cm")
            
        print(f"    ➤ Azimuth = {self.azimuth_deg:.2f}°")
        print(f"    ➤ Elevation = {self.elevation_deg:.2f}°")
            
        # return fish_length, fish_direction, azimuth_deg, elevation_deg

    def save_fish_pointcloud(self, filtered=False):
        pc = self.pointcloud_filtered if filtered else self.pointcloud_raw
        
        if pc is None or pc.shape[0] == 0:
            msg = "EMPTY POINTCLOUD — cannot store or measure"
            print(colored(msg, 'red'))
            self.pc_error_log.append({
                "frame_id": self.fish_frame,
                "color_id": self.color_id,
                "issue": msg
            })
            return False
        
        pcd = tools.create_pointcloud(pc)
        if filtered:
            save_path = os.path.join(self.pointcloud_dir,"object_"+str(self.color_id)+"_filtered.ply")
        else:
            save_path = os.path.join(self.pointcloud_dir,"object_"+str(self.color_id)+".ply")
        
        if os.path.exists(self.pointcloud_dir)==False:
            os.makedirs(self.pointcloud_dir)
            print("Saving pointcloud to: ",self.pointcloud_dir)
            
        print("Saving pointcloud to: ",save_path)
        o3d.io.write_point_cloud(save_path, pcd)
        return True
    
        
    def filter_outliers_HDBSCAN_adaptive(
        self,
        min_cluster_size=10, # tamaño mínimo que debe tener un grupo de puntos para ser considerado un "cluster válido"
        min_samples=2, #  Controla cuántos vecinos necesita un punto para ser considerado central o bien conectado.
        z_jump_threshold_abs=0.07,
        min_points_for_jump=20,
        max_jump_thr = 0.75,
        debug_plot=False):
        """
        1) Agrupa con HDBSCAN (incluye ruido como clusters individuales).
        2) Ordena clusters por z_mean y detecta saltos.
        3) Para cada par de clusters vecinos con salto grande:
            - Si ambos son más pequeños que min_points_for_jump: elimina sólo el que esté
            más alejado de la z_mean global.
            - En el resto de casos: elimina sólo los clusters CON tamaño < min_points_for_jump.
            (Así nunca se eliminan ambos por tamaño.)
        """
        import numpy as np
        import hdbscan
        from sklearn.preprocessing import StandardScaler

        pts = self.pointcloud_raw
        xyz = pts[:, :3]
        
        # mean_z_coords =np.mean(self.pointcloud_raw[:, 2])
        # std_z_coords = np.std(self.pointcloud_raw[:, 2])
        
        # xyz = self.pointcloud_raw[:, :3]
        # scaler = StandardScaler()
        # xyz_norm = scaler.fit_transform(xyz)

        # z_weight = 1.0
        # xyz_norm[:, 2] *= z_weight
        

        # 1) Clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                    min_samples=min_samples,
                                    cluster_selection_epsilon=0.01,
                                    allow_single_cluster = True,
                                    alpha=0.5)
        labels = clusterer.fit_predict(xyz)

        # 2) Construir clusters (ruido como individuales)
        clusters = []
        for lbl in set(labels):
            if lbl != -1:
                clusters.append(np.where(labels == lbl)[0])
        for idx in np.where(labels == -1)[0]:
            clusters.append(np.array([idx], dtype=int))
        
        # 3) Calcular medias y tamaños
        z_means = np.array([pts[c,2].mean() for c in clusters])
        sizes = np.array([len(c) for c in clusters])
        # z_mean_global_clusters = z_means.mean()
            
        # Encuentra el cluster más grande
        largest_idx = np.argmax(sizes)
        z_mean_main = np.mean(pts[clusters[largest_idx], 2])

        # # Desviación típica usando clusters grandes -> Recuperar para evitar saltos raros?
        # El threshold ya cumple bastante la funcion
        # big_clusters = [c for c, sz in zip(clusters, sizes) if sz > 10]
        # z_big = np.concatenate([pts[c, 2] for c in big_clusters])
        # std_z_big = np.std(z_big)

        # 4) Ordenar por z_mean
        order = np.argsort(z_means)
        ordered_clusters = [clusters[i] for i in order]
        ordered_means = z_means[order]
        ordered_sizes = sizes[order]

        # 5) Detectar saltos
        z_diffs = np.diff(ordered_means)
        to_keep = np.ones(len(ordered_clusters), dtype=bool)

        # mean_jump = np.mean(z_diffs)
        # jump_threshold = 2 * mean_jump
        print("z_diffs: ",z_diffs)
        if len(z_diffs) >= 4:
            median_jump = np.median(z_diffs)
            jump_threshold = 2 * median_jump
        else:
            jump_threshold = z_jump_threshold_abs
        
        print("Sorted:", np.sort(z_diffs))
        print("Median:", np.median(z_diffs)) 
        
        print("jump thrs: ",jump_threshold)
        print("jump thrs z_jump_threshold_abs: ",z_jump_threshold_abs)
        
        print("shape:", z_diffs.shape)

        # 6) Si hay un salto grande descartar el mñas alejado de la media
        # si tiene pocos puntos o está alejado de la media
        for i, jump in enumerate(z_diffs):
            if (jump > jump_threshold and jump > z_jump_threshold_abs) or jump > max_jump_thr:
                print("SALTO GRANDE EN Z!!")
                
                # size_i, size_j = ordered_sizes[i], ordered_sizes[i+1]
                dist_i = abs(ordered_means[i]   - z_mean_main)
                dist_j = abs(ordered_means[i+1] - z_mean_main)

                # Decide qué cluster eliminar: el más alejado de la media global de z
                if dist_i > dist_j:
                    to_keep[:i+1] = False
                    print(f"Eliminado cluster {order[i]} por salto en z ({jump:.3f} > {jump_threshold:.3f}) y estar más lejos de la media")
                else:
                    to_keep[i+1:] = False
                    print(f"Eliminado cluster {order[i+1]} por salto en z ({jump:.3f} > {jump_threshold:.3f}) y estar más lejos de la media")
            else:
                print("Salto aceptado: ",jump)

        # 7) Reconstruir filtered
        kept_idxs = np.concatenate([ordered_clusters[k] for k in range(len(to_keep)) if to_keep[k]])
        self.pointcloud_filtered = pts[kept_idxs]
        self.pointcloud_size_ok = len(kept_idxs) > 10

        # 8) Visualización clara
        if debug_plot:
            import open3d as o3d
            import matplotlib.cm as cm

            cmap = cm.get_cmap("tab20")
            geoms_kept = []
            geoms_out  = []
            for k, idxs in enumerate(ordered_clusters):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts[idxs, :3])
        
                if to_keep[k]:
                    # pcd.paint_uniform_color(cmap(k % 20)[:3])
                    pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Azul
                    geoms_kept.append(pcd)
                else:
                    # pcd.paint_uniform_color(cmap(k % 20)[:3])
                    pcd.paint_uniform_color([1.0, 0.0, 0.0]) # rojo
                    geoms_out.append(pcd)
            
            # Crear un eje de coordenadas en el origen, ajusta el tamaño según tu escala
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

            print(f"→ Conservados: {sum(to_keep)}, Eliminados: {len(to_keep) - sum(to_keep)}")
            o3d.visualization.draw_geometries(geoms_kept + geoms_out)
            
            
#--------------------------------------------------------------------------------------------------
    


