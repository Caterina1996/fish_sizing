================================================================================
GUÍA DE PARÁMETROS DE ESTEREOVISIÓN (OPENCV SGBM + WLS)
================================================================================

--- 1. PARÁMETROS PRINCIPALES (SGBM) ---

minDisparity (int)
------------------
- Qué es: La disparidad mínima posible (normalmente 0).
- Valor típico: 0
- Cuándo cambiarlo: Si las cámaras están convergentes (mirando hacia adentro) o si quieres ignorar objetos muy lejanos.
- Efecto: Desplaza el rango de búsqueda.

numDisparities (int)
--------------------
- Qué es: El rango de búsqueda de píxeles. Cuántos píxeles a la izquierda busca el algoritmo para encontrar la pareja.
- Regla de oro: DEBE ser divisible por 16 (16, 32, 48, ..., 128, 192).
- Valor típico: 128 (para resoluciones HD).
- Efecto: 
    - Más alto: Detecta objetos más cercanos a la cámara. Más lento (consume mucha CPU).
    - Más bajo: Solo detecta objetos lejos. Más rápido.
- Nota: Si ves que el pez se corta cuando se acerca a la cámara, SUBE este valor.

blockSize (int)
---------------
- Qué es: Tamaño del "cuadrado" que se compara entre imágenes (siempre impar: 3, 5, 7...).
- Valor típico: 3 o 5 (SGBM), 5-11 (BM).
- Efecto:
    - Pequeño (3): Detecta detalles finos, pero más ruido (grano).
    - Grande (7+): Más suave, menos ruido, pero pierde detalles pequeños y bordes.
- Recomendación: 5 es el punto dulce.

P1 (int)
--------
- Qué es: Penalización por cambios pequeños en la disparidad (suavidad suave).
- Fórmula: 8 * number_of_image_channels * blockSize * blockSize
- Efecto: Controla cuán suave es la superficie.

P2 (int)
--------
- Qué es: Penalización por cambios grandes en la disparidad (saltos de profundidad).
- Fórmula: 32 * number_of_image_channels * blockSize * blockSize
- Regla: P2 debe ser siempre > P1.
- Efecto: 
    - P2 muy alto: La disparidad será muy suave, ignorará saltos bruscos.
    - P2 bajo: Permitirá superficies muy rugosas.

disp12MaxDiff (int)
-------------------
- Qué es: Máxima diferencia permitida en la comprobación Izquierda-Derecha.
- Valor típico: 1 (Estricto) o -1 (Desactivado).
- Efecto: Si es 1, elimina píxeles falsos (outliers). Vital activarlo si usas WLS luego.

uniquenessRatio (int)
---------------------
- Qué es: Margen de confianza (%). Si la mejor coincidencia no es X% mejor que la segunda, se descarta.
- Valor típico: 5 a 15.
- Efecto:
    - Alto (15+): Filtra mucho ruido, pero deja huecos negros en zonas de textura repetitiva (arena, agua).
    - Bajo (5): Rellena más, pero puede meter errores.
- Recomendación: 10.

speckleWindowSize (int)
-----------------------
- Qué es: Filtro de "manchas". Borra regiones de disparidad pequeñas y aisladas (ruido).
- Valor típico: 50 a 200. (0 para desactivar).
- Efecto: Si una "isla" de profundidad tiene menos de estos píxeles, la borra. Limpia mucho la "nieve".

speckleRange (int)
------------------
- Qué es: Cuánto debe variar la profundidad para considerar que un píxel es parte de la misma mancha.
- Valor típico: 1 o 2.
- Nota: Si usas SGBM, a veces hay que multiplicarlo por 16.

mode (constante)
----------------
- Qué es: Algoritmo interno.
- Valores: 
    - MODE_SGBM: Rápido, 5 caminos.
    - MODE_HH: Lento, 8 caminos, muy preciso.
    - MODE_SGBM_3WAY: Equilibrio perfecto. 5 caminos + 1 extra. Recomendado.


--- 2. PARÁMETROS DE FILTRADO (WLS - Weighted Least Squares) ---

lambda (float)
--------------
- Qué es: Fuerza de la regularización (cuánto suaviza).
the amount of regularization during filtering. 
Larger values force filtered disparity map edges to adhere more to source image edges.
- Valor típico: 8000.0
- Efecto:
    - Alto (10000+): Superficies muy lisas, puede perder textura real.
    - Bajo (1000): Más fiel al cálculo original, menos suavizado.

sigma (float)
-------------
- Qué es: Sensibilidad a los bordes de color. Determina cómo el filtro respeta los bordes de la imagen original.
    Defining how sensitive the filtering process is to source image edges. 
    large values can lead to disparity leakage through low-contrast edges. 
    Small values can make the filter too sensitive to noise and textures in the source image

- Typical values range from 0.8 to 2.0. 
- Efecto:
    - Alto: Suaviza incluso a través de bordes de color.
    - Bajo: Se detiene bruscamente en los bordes de color (mantiene la silueta del pez nítida).