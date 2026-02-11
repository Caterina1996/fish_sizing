import numpy as np
import os
from termcolor import colored

import open3d as o3d
from sklearn.decomposition import PCA
from scipy import stats
from pyntcloud import PyntCloud
import cloudpickle as pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hdbscan
import collections


from fish_sizing.utils import tools
from fish_sizing.detection.fish2D import Fish2D, FrameScene

class Fish3D(Fish2D):
    def __init__(self, fish2d, pointcloud_raw, base_path,colors):
        # Llama al constructor de la clase base
        super().__init__(fish2d.fish_frame, fish2d.color_id, fish2d.class_name,fish2d.model_classes_dict,
                         fish2d.mask,fish2d.bbox, fish2d.class_colours_dict,fish2d.does_overlap, 
                         fish2d.overlapping_ids, fish2d.model_used,fish2d.track_id, fish2d.in_image_borders)
        
        self.__dict__.update(fish2d.__dict__)
               
        self.fish_direction = None
        self.azimuth_deg = -1
        self.elevation_deg = -1

        self.pointcloud_raw = np.asarray(pointcloud_raw)  # numpy array (N x 4)
        self.pointcloud_filtered = None
        self.pointcloud_size_ok = False

        self.length = -1
        self.filtered_length = -1
        self.base_path = base_path
        self.pointcloud_dir = os.path.join(base_path, f"frame_{self.fish_frame}")
        self.fish_dist_from_camera = -1
        
        self.colors= colors
        
        self.pc_error_log = []
            
    def get_distance_camera_fish(self):
        #Encontrar la distancia del pez a la camara
        z_vals = self.pointcloud_filtered[:, 2]
        self.fish_dist_from_camera = np.mean(z_vals)
        
    
    def measure_fish_length_ply_with_angles_and_plot(self, 
                                                     plot_fish_direction=True,
                                                     filtered=True):
        """
        Mide el pez usando PCA directamente sobre los datos en memoria (Numpy).
        """
        
        # 1. Seleccionar nube de puntos
        if filtered:
            points = self.pointcloud_filtered
            label_type = "Filtered"
        else:
            points = self.pointcloud_raw
            label_type = "Raw"
        
        # Validaciones
        if points is None or len(points) < 10:
            print(f"⚠️ No hay puntos suficientes para medir ({label_type}).")
            return
            
        # 2. PCA (Principal Component Analysis)
        pca = PCA(n_components=3)
        pca.fit(points)

        # El componente principal (eigenvector con mayor varianza) es la dirección del pez
        principal_components = pca.components_
        fish_direction = principal_components[np.argmax(pca.explained_variance_)]
        
        # Project points over the eigenvector to get length
        projections = np.dot(points, fish_direction)
        fish_length = np.max(projections) - np.min(projections)

        # --- ANGLE CALCULATION ---
        # Normalizar vector
        fish_direction = fish_direction / np.linalg.norm(fish_direction)
        # azimuth: angle with the YZ plane?
        azimuth = np.arctan2(fish_direction[1], fish_direction[0])
        azimuth_deg = np.degrees(azimuth)

        # elevation: angle with the XY plane -> 
        # Ideal case 0 so the fish is in the xy plane
        elevation = np.arctan2(fish_direction[2], np.linalg.norm(fish_direction[:2]))
        elevation_deg = np.degrees(elevation)

        # Save object
        print(f"🐟 Fish {self.track_id}, {self.color_id}, ({label_type}):")
        if filtered:
            # --- VISUALIZE ---
            if plot_fish_direction and fish_length>0:
                # tools.plot_fish_with_dual_cameras(points, fish_direction, fish_length, color_id,self.base_path)
                tools.plot_fish_with_dual_cameras_plotly(points, fish_direction, fish_length, self.color_id, self.base_path)

            self.filtered_fish_direction = fish_direction
            self.filtered_length = fish_length
            self.azimuth_deg = azimuth_deg
            self.elevation_deg = elevation_deg
            print(f"FISH  ➤ filtered length = {self.filtered_length * 100:.2f} cm")
        else:
            self.fish_direction = fish_direction
            self.length = fish_length
            print(f"    ➤ raw length = {self.length * 100:.2f} cm")
            
        print(f"        ➤ Azimuth = {self.azimuth_deg:.2f}°")
        print(f"        ➤ Elevation = {self.elevation_deg:.2f}°")
            
        # return fish_length, fish_direction, azimuth_deg, elevation_deg

    def save_fish_pointcloud(self, filtered=False):
        """
        Save the pc to disk using Open3D
        """
        pc = self.pointcloud_filtered if filtered else self.pointcloud_raw
        colors = self.colors_filtered if filtered else self.colors
        if pc is None or pc.shape[0] == 0:
            msg = "EMPTY POINTCLOUD — cannot store or measure"
            print(colored(msg, 'red'))
            self.pc_error_log.append({
                "frame_id": self.fish_frame,
                "color_id": self.color_id,
                "issue": msg
            })
            return False
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        # BGR (OpenCV) -> RGB (Open3D)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1] / 255.0)
                
        
        if filtered:
            save_path = os.path.join(self.pointcloud_dir,"object_"+str(self.color_id)+"_filtered.ply")
        else:
            save_path = os.path.join(self.pointcloud_dir,"object_"+str(self.color_id)+".ply")
        
        if os.path.exists(self.pointcloud_dir)==False:
            os.makedirs(self.pointcloud_dir)
                       
        print("Saving pointcloud to: ",save_path)
        o3d.io.write_point_cloud(save_path, pcd)
        return True
    
    def filter_outliers_HDBSCAN_adaptive(self, 
                                         min_cluster_size=40, 
                                         max_fish_thickness_meters=0.07, # 5 cm de grosor máximo (ajustable)
                                         debug_plot=False):
        """
        Filtrado de Plano de Pez (Fish-Plane Clipping) + Pre-filtro SOR:
        1. SOR para limpiar la "niebla" flotante (Naranja).
        2. HDBSCAN para aislar el cuerpo principal del pez.
        3. PCA para encontrar el eje lateral del pez (Componente 3 = Grosor).
        4. Recorte de cualquier punto que exceda el grosor físico del pez (Rojo).
        """

        
        if self.pointcloud_raw is None or len(self.pointcloud_raw) < 50:
            print(colored(f"Nube demasiado pequeña para filtrar (Pez {self.track_id}).", "yellow"))
            self.pointcloud_filtered = self.pointcloud_raw
            self.colors_filtered = self.colors
            self.pointcloud_size_ok = False
            return

        pts_raw = self.pointcloud_raw
        colors_raw = self.colors

        # ==========================================================
        # 1. SOR (Statistical Outlier Removal) - Pre-filtro de niebla
        # ==========================================================
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_raw[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors_raw[:, ::-1] / 255.0)

        # Filtro Suave para no dañar las aletas/cola
        cl, ind_sor = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        
        if len(ind_sor) < 20:
            print(colored(f"⚠️ SOR eliminó demasiados puntos en Pez {self.track_id}.", "red"))
            self.pointcloud_filtered = pts_raw
            self.colors_filtered = colors_raw
            self.pointcloud_size_ok = False
            return

        # Guardamos los puntos eliminados por el SOR para pintarlos de naranja luego
        mask_sor = np.zeros(len(pts_raw), dtype=bool)
        mask_sor[ind_sor] = True
        pts_removed_sor = pts_raw[~mask_sor, :3]

        # A partir de aquí, usamos SOLO los puntos que sobrevivieron al SOR
        pts = pts_raw[ind_sor]
        colors = colors_raw[ind_sor]
        xyz = pts[:, :3]

        # ==========================================================
        # 2. HDBSCAN para encontrar la "masa principal"
        # ==========================================================
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
        labels = clusterer.fit_predict(xyz)

        unique_labels = set(labels)
        unique_labels.discard(-1) # Ignorar ruido puro

        if not unique_labels:
            print(colored(f"HDBSCAN no encontró clusters densos para el Pez {self.track_id}.", "red"))
            self.pointcloud_filtered = pts
            self.colors_filtered = colors
            self.pointcloud_size_ok = False
            return

        largest_label = max(unique_labels, key=lambda l: np.sum(labels == l))
        main_cluster_pts = xyz[labels == largest_label]

        if len(main_cluster_pts) < 10:
            self.pointcloud_filtered = pts
            self.colors_filtered = colors
            self.pointcloud_size_ok = False
            return

        # ==========================================================
        # 3. PCA: ENCONTRAR EL EJE DEL GROSOR
        # ==========================================================
        pca = PCA(n_components=3)
        pca.fit(main_cluster_pts)
        
        thickness_vector = pca.components_[2]
        centroid = np.median(main_cluster_pts, axis=0)

        # --- SEGURO DE VIDA (Sanity Check) ---
        cos_angle_with_z = np.abs(np.dot(thickness_vector, [0, 0, 1]))
        if cos_angle_with_z < 0.5: 
            print(colored(f"⚠️ AVISO: PCA dudoso en Pez {self.track_id}. (cos={cos_angle_with_z:.2f}). Aumentando margen preventivo.", "yellow"))
            # max_fish_thickness_meters *= 1.5 

        # ==========================================================
        # 4. FILTRADO POR GROSOR FÍSICO
        # ==========================================================
        vecs_to_centroid = xyz - centroid
        thickness_distances = np.abs(np.dot(vecs_to_centroid, thickness_vector))
        half_thickness = max_fish_thickness_meters / 2.0
        
        mask_keep = thickness_distances < half_thickness

        # ==========================================================
        # 5. RECONSTRUIR NUBE
        # ==========================================================
        self.pointcloud_filtered = pts[mask_keep]
        self.colors_filtered = colors[mask_keep]
        self.pointcloud_size_ok = np.sum(mask_keep) > 10

        # ==========================================================
        # 6. DEBUG VISUAL UNIFICADO
        # ==========================================================
        if debug_plot:
            geoms_to_draw = []

            # 1. Azul (Pez Mantenido)
            pcd_kept = o3d.geometry.PointCloud()
            pcd_kept.points = o3d.utility.Vector3dVector(xyz[mask_keep])
            pcd_kept.paint_uniform_color([0, 0, 1]) 
            geoms_to_draw.append(pcd_kept)
            
            # 2. Rojo (Cortado por el Grosor del PCA)
            if np.sum(~mask_keep) > 0:
                pcd_removed_pca = o3d.geometry.PointCloud()
                pcd_removed_pca.points = o3d.utility.Vector3dVector(xyz[~mask_keep])
                pcd_removed_pca.paint_uniform_color([1, 0, 0]) 
                geoms_to_draw.append(pcd_removed_pca)

            # 3. Naranja (Eliminado inicialmente por el SOR)
            if len(pts_removed_sor) > 0:
                pcd_removed_sor = o3d.geometry.PointCloud()
                pcd_removed_sor.points = o3d.utility.Vector3dVector(pts_removed_sor)
                pcd_removed_sor.paint_uniform_color([1, 0.5, 0]) # NARANJA
                geoms_to_draw.append(pcd_removed_sor)
            
            # 4. Línea Verde (Vector del grosor/Sándwich)
            v3_end = centroid + thickness_vector * half_thickness
            v3_start = centroid - thickness_vector * half_thickness
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([v3_start, v3_end]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
            geoms_to_draw.append(line_set)
            
            print(f"📊 Resumen Pez {self.track_id}:")
            print(f"  - Conservados (Azul): {np.sum(mask_keep)}")
            print(f"  - Eliminados por Grosor PCA (Rojo): {np.sum(~mask_keep)}")
            print(f"  - Eliminados por SOR (Naranja): {len(pts_removed_sor)}")
            
            o3d.visualization.draw_geometries(geoms_to_draw)
            
        
    def filter_outliers_HDBSCAN_adaptive_saltos_imnproved(self, min_cluster_size=20, z_jump_threshold_abs=0.03, debug_plot=False):
        pts = self.pointcloud_raw
        xyz = pts[:, :3]

        # 1. HDBSCAN
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
        labels = clusterer.fit_predict(xyz)

        # 2. Construir clusters (IGNORAMOS EL RUIDO -1 como clusters individuales)
        # Dejamos que el ruido desaparezca de la ecuación de los saltos
        clusters = []
        for lbl in set(labels):
            if lbl != -1:
                clusters.append(np.where(labels == lbl)[0])

        if not clusters:
            print("No se encontraron clusters válidos.")
            return

        # 3. Calcular medias y aislar el pez
        z_means = np.array([pts[c, 2].mean() for c in clusters])
        sizes = np.array([len(c) for c in clusters])
        
        largest_idx = np.argmax(sizes)
        z_mean_main = z_means[largest_idx] # El centro de gravedad del pez

        # 4. Ordenar clusters por Z
        order = np.argsort(z_means)
        ordered_clusters = [clusters[i] for i in order]
        ordered_means = z_means[order]
        
        z_diffs = np.diff(ordered_means)
        to_keep = np.ones(len(ordered_clusters), dtype=bool)

        jump_threshold = max(z_jump_threshold_abs, np.median(z_diffs) * 2 if len(z_diffs) >= 4 else z_jump_threshold_abs)

        # 5. Aplicar TU lógica de saltos, pero solo sobre clusters densos reales
        for i, jump in enumerate(z_diffs):
            if jump > jump_threshold:
                dist_i = abs(ordered_means[i] - z_mean_main)
                dist_j = abs(ordered_means[i+1] - z_mean_main)

                if dist_i > dist_j:
                    to_keep[:i+1] = False
                else:
                    to_keep[i+1:] = False

        # 6. Reconstruir
        kept_idxs = np.concatenate([ordered_clusters[k] for k in range(len(to_keep)) if to_keep[k]])
        
        if len(kept_idxs) > 10:
            self.pointcloud_filtered = pts[kept_idxs]
            self.colors_filtered = self.colors[kept_idxs]
            self.pointcloud_size_ok = True
        else:
            self.pointcloud_filtered = self.pointcloud_raw
            self.colors_filtered = self.colors
            self.pointcloud_size_ok = False
            
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
        
    def filter_outliers_HDBSCAN_adaptive_old(self,
        min_cluster_size=10, # tamaño mínimo que debe tener un grupo de puntos para ser considerado un "cluster válido"
        min_samples=2,     # Controla cuántos vecinos necesita un punto para ser considerado central o bien conectado.
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
        

        pts = self.pointcloud_raw
        xyz = pts[:, :3]
        
        # mean_z_coords =np.mean(self.pointcloud_raw[:, 2])
        # std_z_coords = np.std(self.pointcloud_raw[:, 2])
        
        # xyz = self.pointcloud_raw[:, :3]
        # scaler = StandardScaler()
        # xyz_norm = scaler.fit_transform(xyz)

        # z_weight = 1.0
        # xyz_norm[:, 2] *= z_weight
        

        # 1) Clustering HDBSCAN
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
            # Crea un array de numpy que contiene SOLO ese índice: ej. np.array([45])
             # Y lo añade a la lista de clusters como si fuera un grupo más
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
        self.colors_filtered = self.colors[kept_idxs]
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
    


