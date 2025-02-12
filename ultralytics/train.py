train_results = model.train(
    data = "data.yaml",
    epochs = 70,
    imgsz = 640,
    batch = 16,
    device = [0,1],
    amp = True,
    workers = 2,
    scale = 0.0,
    fliplr = 0.5,
    flipud = 0.5,
    lr0=0.005,
    lrf = 0.01,
    cos_lr = True,
    # mode = "ddp_notebook"
    project="runs/train",  # Specify project directory
    name="efficientnetb3_training"  # Give a descriptive name

)
