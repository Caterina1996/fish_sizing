# En tu scripts/run_pipeline.py

for topic, msg, t in bag.read_messages(...):
    # 1. Convertir a CV2
    img_l = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # 2. Inferencia (Batch=1 implícito)
    # Al ser la misma instancia 'detector', el tracker recuerda el frame anterior
    scene, debug_data = detector.process_frame(img_l, frame_id=str(t))

    # 3. Lógica condicional (Esto es lo que ahorra tiempo real)
    if scene.has_fish(): 
        # Solo gastas tiempo de cálculo estéreo si vale la pena
        img_r = ... # conseguir imagen derecha
        disp = stereo.compute(img_l, img_r)
        # ...